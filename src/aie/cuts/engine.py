"""Run a CutSpec against curated/ and return a table plus the metadata needed to trust it.

Reads only the files a cut needs (each sector's vacancies + metrics, projected columns)
through the shared loader (`aie.read`), then aggregates with DuckDB. Files are
cached in-process, so a front end re-cutting the same release stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb
import pandas as pd

from .. import catalog, read, settings, storage
from ..methods import quadrants, valuation
from .spec import CORE_DIMENSIONS, MEASURES, VACANCY_COLUMNS, CustomThresholds, CutSpec

REFERENCE_SECTOR = "civil-service"  # the fixed thresholds are its ensemble quartiles

DIM_COLS = [
    "vacancy_id",
    "salary_annual_gbp",
    *[c for c in CORE_DIMENSIONS if c not in ("sector", "quadrant")],
]


@dataclass
class CutResult:
    data: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def to_csv(self) -> bytes:
        return self.data.to_csv(index=False).encode()

    def save(self, name: str) -> str:
        """Write to cuts/<sector>/release=<r>/<name>.csv (+ .meta.json) and add a manifest row.

        Cross-sector cuts go to cuts/cross-sector/<s1-r1+s2-r2...>/ since they span releases.
        """
        rel = self.meta["releases"]
        if len(rel) == 1:
            ((sector, release),) = rel.items()
            scope, folder = sector, f"release={release}"
        else:
            scope, folder = "cross-sector", "+".join(f"{s}-{r}" for s, r in rel.items())
        base = storage.join(settings.cuts_dir(scope), folder, name)
        storage.write_bytes(self.to_csv(), base + ".csv")
        storage.write_json(self.meta, base + ".meta.json")
        catalog.append_cut(
            {
                "name": name,
                "scope": scope,
                "uri": base + ".csv",
                "created_utc": self.meta["created_utc"],
                "label": self.meta["spec"].get("label") or "",
                "releases": ";".join(f"{k}={v}" for k, v in self.meta["releases"].items()),
                "methodology_version": self.meta["methodology_version"],
                "score_model": self.meta["score_model"],
            }
        )
        return base + ".csv"


def score_model(spec: CutSpec, methodology: dict) -> str:
    cross = methodology["comparability"]["cross_sector_score_model"]
    if spec.cross_sector:
        if spec.score_model not in (None, cross):
            raise ValueError(
                f"cross-sector cuts must use {cross}, the only model every sector has "
                f"(got {spec.score_model})"
            )
        return cross
    return spec.score_model or read.sector_info(spec.sectors[0])["primary_score_model"]


_REFERENCE: dict[tuple, quadrants.Thresholds] = {}


def custom_thresholds(c: CustomThresholds) -> quadrants.Thresholds:
    """Turn the page's limits into threshold values. Percentiles are taken on the reference
    sector's published primary metrics (ensemble roles, primary weights), exactly as the
    fixed thresholds were, so 25 / 25 gives back the fixed values."""
    if not c.by_percentile:
        return quadrants.Thresholds("custom", c.mean_hi, c.high_var_lo, c.low_var_hi, float("nan"))
    release, _ = read.resolve_release(REFERENCE_SECTOR)
    key = (release, c.mean_top_pct, c.var_pct)
    if key not in _REFERENCE:
        model = read.sector_info(REFERENCE_SECTOR)["primary_score_model"]
        mt = read.partition("metrics", REFERENCE_SECTOR, release)
        mt = mt[mt["score_model"] == model]
        w = read.partition("vacancies", REFERENCE_SECTOR, release, ["vacancy_id", "weight"])
        roles = mt.merge(w[["vacancy_id", "weight"]], on="vacancy_id")
        thr = quadrants.sector_quartile_thresholds(
            roles, float("nan"), c.mean_top_pct / 100, c.var_pct / 100
        )
        # 4 dp, as the published fixed values are, so 25 / 25 reproduces the headline exactly
        _REFERENCE[key] = quadrants.Thresholds(
            "custom",
            *(round(x, 4) for x in (thr.mean_hi, thr.high_var_lo, thr.low_var_hi)),
            float("nan"),
        )
    return _REFERENCE[key]


def _apply_custom(mt: pd.DataFrame, salary: pd.DataFrame, thr) -> pd.DataFrame:
    """Reclassify with custom limits and revalue, using the shared methods."""
    mt = mt.merge(salary, on="vacancy_id", how="left")
    q = quadrants.classify(mt, thr)
    v = valuation.per_fte_values(mt["salary_annual_gbp"], mt["mean_exposure"], q)
    return mt.assign(
        quadrant_custom=q,
        cashable_custom_per_fte_gbp=v["cashable"],
        non_cashable_custom_per_fte_gbp=v["non_cashable"],
    ).drop(columns="salary_annual_gbp")


def _filter_sql(f, col_sql: str, params: list) -> str:
    if f.op in ("is_null", "not_null"):
        return f"{col_sql} IS {'NOT ' if f.op == 'not_null' else ''}NULL"
    if f.op == "between":
        params.extend(f.values[:2])
        return f"{col_sql} BETWEEN ? AND ?"
    if f.op == "eq":
        params.append(f.values[0])
        return f"{col_sql} = ?"
    params.extend(f.values)
    marks = ", ".join("?" for _ in f.values) or "NULL"
    return f"{col_sql} {'NOT ' if f.op == 'not_in' else ''}IN ({marks})"


def _column_sql(col: str, spec: CutSpec, sector_cols: set[str]) -> str:
    if col == "quadrant":
        return "m.quadrant_sel"
    if col in CORE_DIMENSIONS:
        return f'v."{col}"'
    if col in sector_cols:
        if spec.cross_sector:
            raise ValueError(f"{col}: sector-specific columns need a one-sector cut")
        return f'v."{col}"'
    raise ValueError(f"cannot group or filter by {col!r}")


# Vacancy-level columns that are not a plain vacancies column.
_VACANCY_SQL = {
    "fte": "v.w",
    "n_tasks": "m.n_tasks",
    "mean_exposure": "m.mean_exposure",
    "var_exposure": "m.var_exposure",
    "quadrant": "m.quadrant_sel",
    "opportunity_per_fte_gbp": "m.opportunity_per_fte_gbp",
    "cashable_per_fte_gbp": "m.cash_sel",
    "non_cashable_per_fte_gbp": "m.non_cash_sel",
}
_VACANCY_ORDER = ["sector", "org_unit", "occupation_group", "job_title", "vacancy_id"]


def _vacancy_sql(col: str, spec: CutSpec, sector_cols: set[str]) -> str:
    if col in _VACANCY_SQL:
        return _VACANCY_SQL[col]
    if col in VACANCY_COLUMNS:
        return f'v."{col}"'
    return _column_sql(col, spec, sector_cols)


def run_cut(spec: CutSpec) -> CutResult:
    m = read.methodology()
    version = spec.methodology_version or m["version"]
    model = score_model(spec, m)
    warnings: list[str] = []

    releases = {}
    for s in spec.sectors:
        want = spec.release.get(s, "latest") if isinstance(spec.release, dict) else spec.release
        releases[s], warn = read.resolve_release(s, want)
        if warn:
            warnings.append(warn)

    thr = custom_thresholds(spec.custom_thresholds) if spec.custom_thresholds else None
    if thr is not None:
        warnings.append("Custom quadrant limits: a sensitivity, not the published headline figures")

    vacancy = spec.level == "vacancy"
    used = (spec.columns if vacancy else spec.group_by) + [f.column for f in spec.filters]
    dims, metrics = [], []
    weight_specs, sector_cols = {}, set()
    for s, r in releases.items():
        info = read.sector_info(s)
        ws = info["primary_weight_spec"] if spec.weight_spec == "primary" else spec.weight_spec
        weight_specs[s] = ws
        if ws == "unweighted":
            warnings.append(f"{s}: no raking target yet, so totals are advert counts, not FTE")
        own = set(read.sector_columns(s, r))
        sector_cols |= own
        wcol = read.weight_column(s, r, ws)
        extra = ["job_title", "posted_date"] if vacancy else []
        cols = [*DIM_COLS, *extra, wcol, *[c for c in used if c in own]]
        v = read.partition("vacancies", s, r, list(dict.fromkeys(cols)))
        dims.append(v.rename(columns={wcol: "w"}) if wcol != "weight" else v.assign(w=v["weight"]))
        mt = read.partition("metrics", s, r)
        mt = mt[(mt["methodology_version"] == version) & (mt["score_model"] == model)]
        if mt.empty:
            raise ValueError(
                f"{s} {r}: no metrics for {version} / {model}. Only the current methodology "
                f"is published; recompute a past one with `aie metrics {s} --methodology ...`"
            )
        if thr is not None:
            mt = _apply_custom(mt, v[["vacancy_id", "salary_annual_gbp"]], thr)
        metrics.append(mt)

    con = duckdb.connect()
    con.register("v", pd.concat(dims, ignore_index=True))
    con.register("m_raw", pd.concat(metrics, ignore_index=True))
    quad, cash, non_cash = {
        "fixed": ("quadrant", "cashable_per_fte_gbp", "non_cashable_per_fte_gbp"),
        "sector_quartiles": (
            "quadrant_sector_quartiles", "cashable_sq_per_fte_gbp", "non_cashable_sq_per_fte_gbp"
        ),
        "custom": (
            "quadrant_custom", "cashable_custom_per_fte_gbp", "non_cashable_custom_per_fte_gbp"
        ),
    }[spec.threshold_basis]  # fmt: skip
    con.execute(
        f"""CREATE VIEW m AS SELECT *,
            {quad} AS quadrant_sel,
            {cash} AS cash_sel,
            {non_cash} AS non_cash_sel,
            CASE WHEN {quad} = 'Displacement-type' THEN opportunity_per_fte_gbp ELSE 0 END
                AS cash_share_sel
            FROM m_raw"""
    )

    params: list = []
    where = ["v.w IS NOT NULL"] + [
        _filter_sql(f, _column_sql(f.column, spec, sector_cols), params) for f in spec.filters
    ]
    if vacancy:
        df = _vacancies(con, spec, sector_cols, where, params)
        warnings.append("One row per advert: figures are the advert's own, so none are blanked")
    else:
        df = _groups(con, spec, sector_cols, where, params, warnings)

    meta = {
        "spec": spec.model_dump(),
        "sectors": spec.sectors,
        "releases": releases,
        "score_model": model,
        "weight_specs": weight_specs,
        "threshold_basis": spec.threshold_basis,
        "custom_thresholds": (
            {
                **spec.custom_thresholds.model_dump(exclude_none=True),
                "mean_hi": thr.mean_hi,
                "high_var_lo": thr.high_var_lo,
                "low_var_hi": thr.low_var_hi,
            }
            if thr is not None
            else None
        ),
        "methodology_version": version,
        "methodology": {k: m[k] for k in ("decay", "population", "quadrants", "valuation")},
        "rows": len(df),
        "warnings": warnings,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return CutResult(df, meta)


def _vacancies(con, spec: CutSpec, sector_cols, where, params) -> pd.DataFrame:
    """One row per advert, in a stable order (organisation, occupation, title)."""
    if not spec.columns:
        raise ValueError("choose at least one column")
    select = [f'{_vacancy_sql(c, spec, sector_cols)} AS "{c}"' for c in spec.columns]
    order = [_vacancy_sql(c, spec, sector_cols) for c in _VACANCY_ORDER]
    sql = f"""
        SELECT {", ".join(select)}
        FROM v
        JOIN m USING (vacancy_id, sector)
        WHERE {" AND ".join(where)}
        ORDER BY {", ".join(order)}
    """
    return con.execute(sql, params).df()


def _groups(con, spec: CutSpec, sector_cols, where, params, warnings) -> pd.DataFrame:
    groups = [_column_sql(c, spec, sector_cols) for c in spec.group_by]
    select_groups = [f'{g} AS "{c}"' for g, c in zip(groups, spec.group_by)]
    sql = f"""
        SELECT {", ".join(select_groups + [""])}
            COUNT(*) AS n_adverts,
            SUM(v.w) AS fte,
            SUM(CASE WHEN v.salary_annual_gbp IS NOT NULL THEN v.w END) AS fte_with_salary,
            SUM(v.w * m.mean_exposure) / NULLIF(SUM(v.w), 0) AS mean_exposure,
            SUM(v.w * v.salary_annual_gbp) AS paybill_gbp,
            SUM(v.w * m.opportunity_per_fte_gbp) AS opportunity_gbp,
            SUM(v.w * m.cash_sel) AS cashable_gbp,
            SUM(v.w * m.cash_share_sel) AS cashable_share_gbp,
            SUM(v.w * m.non_cash_sel) AS non_cashable_gbp
        FROM v
        JOIN m USING (vacancy_id, sector)
        WHERE {" AND ".join(where)}
        {"GROUP BY " + ", ".join(groups) if groups else ""}
        {"ORDER BY " + ", ".join(f'"{c}"' for c in spec.group_by) if groups else ""}
    """
    df = con.execute(sql, params).df()
    df["pct_fte"] = 100 * df["fte"] / df["fte"].sum()
    df["opportunity_pct_of_paybill"] = 100 * df["opportunity_gbp"] / df["paybill_gbp"]
    df["opportunity_per_fte_gbp"] = df["opportunity_gbp"] / df["fte_with_salary"]

    small = df["n_adverts"] < spec.min_cell_adverts
    if small.any():
        df.loc[small, [c for c in MEASURES if c != "n_adverts"]] = None
        warnings.append(
            f"{int(small.sum())} group(s) under {spec.min_cell_adverts} adverts: measures blanked"
        )
    return df[spec.group_by + spec.measures]
