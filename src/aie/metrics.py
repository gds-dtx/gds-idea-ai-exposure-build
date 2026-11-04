"""Per-vacancy metrics: exposure moments, quadrants and £ per FTE, for one methodology.

`publish` uses `compute_metrics` to write curated metrics.parquet for the CURRENT methodology.
`from_curated` recomputes the same table from a published release's tasks + vacancies for any
other methodology: a past version (`settings.methodology("m1")`) or a sensitivity scenario
(`settings.with_changes(...)`). Releases never change once published, so a past version is
reproduced exactly; nothing but the current version needs storing.

    from aie import metrics, settings
    m = settings.with_changes({"decay.primary": 0.9}, label="decay_090")
    rows, thresholds = metrics.from_curated("nhs", "v1", m)
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from . import read, settings
from .methods import decay, quadrants, valuation

REFERENCE_SECTOR = "civil-service"  # the fixed thresholds are this sector's ensemble quartiles


def compute_metrics(
    tasks: pd.DataFrame,
    roles: pd.DataFrame,
    score_model: str,
    m: dict | None = None,
    fixed: quadrants.Thresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """metrics rows + the thresholds used, for one scoring model.

    `tasks`: vacancy_id, task_index, score. `roles`: vacancy_id, weight, salary.
    `fixed` replaces the methodology's fixed threshold values (used for re-derived ones).
    """
    m = m or settings.methodology()
    mom = decay.vacancy_moments(tasks, "score", m["decay"]["primary"], m["decay"]["comparison"])
    r = mom.merge(roles, on="vacancy_id", how="left")

    # Task-level automatable cut-off: weighted by the vacancy's weight, weighted roles only.
    tw = tasks.merge(roles[["vacancy_id", "weight"]], on="vacancy_id", how="inner").dropna(
        subset=["weight", "score"]
    )
    t_hi = quadrants.task_hi(tw["score"], tw["weight"], m)
    auto = (tasks["score"] >= t_hi).groupby(tasks["vacancy_id"]).mean()

    thr_fixed = (
        dataclasses.replace(fixed, task_hi=t_hi) if fixed else quadrants.fixed_thresholds(t_hi, m)
    )
    thr_sq = quadrants.sector_quartile_thresholds(r, t_hi, m=m)
    r["quadrant"] = quadrants.classify(r, thr_fixed, m)
    r["quadrant_sector_quartiles"] = quadrants.classify(r, thr_sq, m)
    v = valuation.per_fte_values(r["salary"], r["mean_exposure"], r["quadrant"], m)
    v_sq = valuation.per_fte_values(
        r["salary"], r["mean_exposure"], r["quadrant_sector_quartiles"], m
    )

    metrics = pd.DataFrame(
        {
            "vacancy_id": r["vacancy_id"],
            "methodology_version": m["version"],
            "score_model": score_model,
            "n_tasks": r["n_tasks"],
            "decay": r["decay"],
            "mean_exposure": r["mean_exposure"],
            "var_exposure": r["var_exposure"],
            "mean_exposure_uniform": r["mean_exposure_uniform"],
            "var_exposure_uniform": r["var_exposure_uniform"],
            "auto_task_share": r["vacancy_id"].map(auto),
            "quadrant": r["quadrant"],
            "quadrant_sector_quartiles": r["quadrant_sector_quartiles"],
            "opportunity_per_fte_gbp": v["opportunity"],
            "cashable_per_fte_gbp": v["cashable"],
            "cashable_share_per_fte_gbp": v["cashable_share"],
            "non_cashable_per_fte_gbp": v["non_cashable"],
            "cashable_sq_per_fte_gbp": v_sq["cashable"],
            "non_cashable_sq_per_fte_gbp": v_sq["non_cashable"],
        }
    )
    thresholds = pd.DataFrame(
        [
            {**t.as_dict(), "methodology_version": m["version"], "score_model": score_model}
            for t in (thr_fixed, thr_sq)
        ]
    )
    return metrics, thresholds


def _inputs(sector: str, release: str, score_model: str, weight_spec: str):
    col = read.score_column(sector, release, score_model)
    tasks = read.partition("tasks", sector, release, ["vacancy_id", "task_index", col])
    tasks = tasks[["vacancy_id", "task_index", col]].rename(columns={col: "score"})
    wcol = read.weight_column(sector, release, weight_spec)
    vac = read.partition("vacancies", sector, release, ["vacancy_id", "salary_annual_gbp", wcol])
    roles = pd.DataFrame(
        {"vacancy_id": vac["vacancy_id"], "weight": vac[wcol], "salary": vac["salary_annual_gbp"]}
    )
    return tasks, roles


_REFERENCE_CACHE: dict[tuple, quadrants.Thresholds] = {}


def reference_thresholds(m: dict, release: str = "latest") -> quadrants.Thresholds:
    """Fixed thresholds re-derived for `m` ("option b"): the reference sector's ensemble
    quartiles (primary model, primary weights) at m's decay and quartile percentiles."""
    release, _ = read.resolve_release(REFERENCE_SECTOR, release)
    q = m["quadrants"]
    key = (
        release, m["decay"]["primary"], q["high_mean_top"], q["disp_low_bottom"],
        m["population"]["min_tasks"],
    )  # fmt: skip
    if key not in _REFERENCE_CACHE:
        model = read.sector_info(REFERENCE_SECTOR)["primary_score_model"]
        tasks, roles = _inputs(REFERENCE_SECTOR, release, model, "primary")
        mom = decay.vacancy_moments(tasks, "score", m["decay"]["primary"], 1.0)
        r = mom.merge(roles, on="vacancy_id", how="left")
        t_hi = float("nan")  # replaced by each sector's own task cut-off in compute_metrics
        thr = quadrants.sector_quartile_thresholds(r, t_hi, m=m)
        # 4 dp, as the published fixed values are, so the primary settings reproduce them
        _REFERENCE_CACHE[key] = quadrants.Thresholds(
            "fixed", *(round(x, 4) for x in (thr.mean_hi, thr.high_var_lo, thr.low_var_hi)), t_hi
        )
    return _REFERENCE_CACHE[key]


def from_curated(
    sector: str,
    release: str = "latest",
    m: dict | None = None,
    score_model: str | None = None,
    weight_spec: str = "primary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute a release's metrics under methodology `m` (default: current) from curated
    tasks + vacancies. Same rows and columns as metrics.parquet, for one scoring model."""
    m = m or settings.methodology()
    release, _ = read.resolve_release(sector, release)
    model = score_model or read.sector_info(sector)["primary_score_model"]
    tasks, roles = _inputs(sector, release, model, weight_spec)
    rederive = m["quadrants"].get("fixed_source") == "rederive"
    fixed = reference_thresholds(m) if rederive else None
    return compute_metrics(tasks, roles, model, m, fixed)


def totals(metrics: pd.DataFrame, sector: str, release: str = "latest") -> dict:
    """Headline £ and FTE for a metrics table (primary weight): a quick check, not a cut."""
    release, _ = read.resolve_release(sector, release)
    w = read.partition("vacancies", sector, release, ["vacancy_id", "weight"])
    d = metrics.merge(w[["vacancy_id", "weight"]], on="vacancy_id").dropna(subset=["weight"])
    out = {"fte": d["weight"].sum()}
    for c in ("opportunity", "cashable", "cashable_share", "non_cashable"):
        out[f"{c}_gbp"] = (d["weight"] * d[f"{c}_per_fte_gbp"]).sum()
    out["total_savings_gbp"] = out["cashable_gbp"] + out["non_cashable_gbp"]
    return out
