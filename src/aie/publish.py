"""Publish one sector release: build layers -> curated/sector=<s>/release=<r>/ (L4).

  vacancies.parquet  adverts + sector columns + weights (primary `weight`, other `weight_<spec>`)
  tasks.parquet      tasks + primary score + `score_<model>` for every other model
  metrics.parquet    exposure moments, quadrants and £ per FTE; thresholds in _meta.json

metrics is computed here (aie.metrics), from the task scores, using the shared methods and
the CURRENT methodology.toml only, for:
  * the sector's primary scoring model (CS: 8-model ensemble), and
  * the cross-sector model (gpt-oss-120b), so every sector can be compared on one model.

Rebuilding metrics after a methodology change = re-running publish; nothing upstream changes.
Past versions and sensitivity scenarios are recomputed on demand (aie.metrics.from_curated).
"""

from __future__ import annotations

import pandas as pd

from . import catalog, contract, settings, storage
from .metrics import compute_metrics
from .sectors import Sector


def _read(sector: Sector, layer: str, release: str, name: str, columns=None) -> pd.DataFrame:
    uri = storage.join(settings.build_dir(sector.slug, layer, release), name)
    return storage.read_parquet(uri, columns=columns)


def _write(df: pd.DataFrame, table: str, sector: Sector, release: str, extra=None) -> None:
    folder = settings.curated_dir(sector.slug, release)
    inputs = [
        settings.build_dir(sector.slug, layer, release) for layer in ("clean", "scored", "weights")
    ]
    uri = storage.write_table(
        df, folder, f"{table}.parquet", sector=sector.slug, layer="curated", release=release,
        inputs=inputs, extra=extra,
    )  # fmt: skip
    print(f"  wrote {uri}  ({len(df):,} rows)")


def finalise_population(adverts: pd.DataFrame, n_tasks: pd.Series, weights: pd.DataFrame):
    """Fill n_tasks and set in_scope / exclusion_reason against the PRIMARY weight spec.

    in_scope means "in the weighted population". Reasons, first that applies:
    sector rule (from clean) > no task scores > fewer than MIN_TASKS > no raking target.
    """
    min_tasks = settings.methodology()["population"]["min_tasks"]
    a = adverts.copy()
    a["n_tasks"] = a["vacancy_id"].map(n_tasks).fillna(0).astype("Int64")
    primary = weights[weights["is_primary"].fillna(False).astype(bool)].set_index("vacancy_id")
    has_w = a["vacancy_id"].map(primary["weight"]).notna()

    reason = a["exclusion_reason"].astype("string")
    sector_ok = reason.isna()
    reason = reason.mask(sector_ok & (a["n_tasks"] == 0), "no task scores")
    reason = reason.mask(
        reason.isna() & (a["n_tasks"] < min_tasks), f"fewer than {min_tasks} tasks"
    )
    reason = reason.mask(reason.isna() & ~has_w, "no raking target for its category")
    a["exclusion_reason"] = reason.where(~has_w)
    a["in_scope"] = has_w
    return a


def publish(sector: Sector, release: str) -> None:
    print(f"[{sector.slug}] publish release={release}")
    adverts = _read(sector, "clean", release, "adverts.parquet")
    ext = _read(sector, "clean", release, "adverts_ext.parquet")
    tasks = _read(sector, "scored", release, "tasks.parquet")
    model_scores = _read(sector, "scored", release, "model_scores.parquet")
    weights = _read(sector, "weights", release, "vacancy_weights.parquet")

    known = set(adverts["vacancy_id"])
    orphan = ~tasks["vacancy_id"].isin(known)
    if orphan.any():
        n_vac = tasks.loc[orphan, "vacancy_id"].nunique()
        print(f"  dropping {int(orphan.sum()):,} tasks for {n_vac:,} vacancies not in adverts")
        tasks = tasks[~orphan]
        model_scores = model_scores[model_scores["vacancy_id"].isin(known)]

    n_tasks = tasks.groupby("vacancy_id").size()
    adverts = finalise_population(adverts, n_tasks, weights)

    # vacancies: core + primary weight + other weight specs + sector columns
    wide_w = weights.pivot(index="vacancy_id", columns="weight_spec", values="weight")
    vac = adverts.assign(weight=adverts["vacancy_id"].map(wide_w[sector.primary_weight_spec]))
    for spec in wide_w.columns.drop(sector.primary_weight_spec):
        vac[f"weight_{spec}"] = vac["vacancy_id"].map(wide_w[spec])
    vac = vac.merge(contract.conform_ext(ext), on="vacancy_id", how="left", validate="one_to_one")
    vac = contract.conform(vac, "vacancies")
    targets = (
        weights.drop_duplicates("weight_spec")
        .set_index("weight_spec")[["is_primary", "target_source", "target_date"]]
        .to_dict("index")
    )
    _write(vac, "vacancies", sector, release, extra={"weight_specs": targets})

    # tasks: primary score + one score_<model> column per other model
    other = model_scores[model_scores["model"] != sector.primary_score_model]
    t = tasks
    if not other.empty:
        wide_s = other.pivot(
            index=["vacancy_id", "task_index"], columns="model", values="exposure_score"
        )
        wide_s.columns = [contract.model_column(c) for c in wide_s.columns]
        t = t.merge(wide_s.reset_index(), on=["vacancy_id", "task_index"], how="left")
    _write(contract.conform(t, "tasks"), "tasks", sector, release)

    roles = vac[["vacancy_id", "salary_annual_gbp", "weight"]].rename(
        columns={"salary_annual_gbp": "salary"}
    )
    cross = settings.methodology()["comparability"]["cross_sector_score_model"]
    all_metrics, all_thr = [], []
    for model in dict.fromkeys([sector.primary_score_model, cross]):
        if model == sector.primary_score_model:
            scores = tasks[["vacancy_id", "task_index", "exposure_score"]]
        else:
            scores = model_scores.loc[
                model_scores["model"] == model, ["vacancy_id", "task_index", "exposure_score"]
            ]
            if scores.empty:
                print(f"  no {model} scores: skipping cross-sector metrics")
                continue
        metrics, thr = compute_metrics(
            scores.rename(columns={"exposure_score": "score"}), roles, model
        )
        all_metrics.append(metrics)
        all_thr.append(thr.assign(weight_spec=sector.primary_weight_spec))
        print(
            f"  {model}: fixed mean_hi={thr.iloc[0]['mean_hi']:.4f} | sector quartiles "
            f"{thr.iloc[1]['mean_hi']:.4f} / {thr.iloc[1]['high_var_lo']:.4f} / "
            f"{thr.iloc[1]['low_var_hi']:.4f}"
        )
    _write(
        contract.conform(pd.concat(all_metrics), "metrics"),
        "metrics",
        sector,
        release,
        extra={"thresholds": pd.concat(all_thr).to_dict("records")},
    )
    catalog.record_stage(sector, release, "published")
