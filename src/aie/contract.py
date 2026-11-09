"""The data contract: the columns every sector must produce.

CURATED (what analysis, the cut engine and the dashboard read), three tables per sector:

    curated/sector=<s>/release=<r>/vacancies.parquet   one row per advert
    curated/sector=<s>/release=<r>/tasks.parquet       one row per task
    curated/sector=<s>/release=<r>/metrics.parquet     one row per advert x methodology x model

`vacancies` and `tasks` have fixed CORE columns (same name and meaning in every sector) and
may carry extra columns after them:
  * vacancies: `weight_<spec>` for any non-primary weighting (CS: weight_rake_3d), then the
    sector's own columns (NHS pay scheme, local-govt is_school_role, police force flags...).
  * tasks: `score_<model>` for each model other than the primary (CS: its 8 ensemble members,
    including score_gpt_oss_120b for cross-sector work).
Cross-sector reads use the core columns. `sector` and `release` are folder partitions.

BUILD layers (internal, build/sector=<s>/release=<r>/<layer>/) use the long-form schemas below
(adverts, task_scores, model_scores, weights); `aie.publish` reshapes them into curated.
"""

from __future__ import annotations

import re

import pandas as pd

# column -> pandas dtype
ADVERT_CORE = {
    "vacancy_id": "string",  # sector-prefixed: cs-, nhs-, edu-, lg-, pol-, pc-
    "source_vacancy_id": "string",  # id in the source system / pre-scaffold outputs
    "source": "string",  # e.g. "Civil Service Jobs", "NHS Jobs", "council job boards"
    "posted_date": "datetime64[ns]",
    "year": "Int64",
    "job_title": "string",
    "employer": "string",  # the organisation that posted the advert
    "org_unit": "string",  # CS department / NHS trust / school / council / force
    "occupation_group": "string",  # CS profession / NHS staff group / DfE category
    "grade": "string",  # CS grade / AfC band / teacher grade
    "region": "string",
    "salary_annual_gbp": "float64",  # the salary valuations use (see salary_source)
    "salary_source": "string",
    "headcount": "Int64",  # posts the advert represents; 1 unless the source says more
    "n_tasks": "Int64",
    "in_scope": "boolean",  # eligible for the weighted population
    "exclusion_reason": "string",  # why not, when in_scope is False
}

TABLES: dict[str, dict[str, str]] = {
    # --- build layers ---------------------------------------------------------------------
    "adverts": ADVERT_CORE,
    # One row per task, with the sector's PRIMARY score (CS: 8-model ensemble mean).
    "task_scores": {
        "vacancy_id": "string",
        "task_index": "Int64",  # 1-based, in advert order (decay depends on it)
        "task_text": "string",
        "exposure_score": "float64",
        "score_model": "string",
        "task_label": "string",  # substantive / peripheral / generic, where labelled
        "scoring_run_id": "string",
    },
    # One row per task x model. Every sector has gpt-oss-120b; CS also has 7 other models.
    "model_scores": {
        "vacancy_id": "string",
        "task_index": "Int64",
        "model": "string",
        "exposure_score": "float64",
    },
    # One row per advert x weighting spec. Null weight = outside the weighted population.
    "weights": {
        "vacancy_id": "string",
        "weight_spec": "string",  # rake_1d, rake_3d, unweighted
        "weight": "float64",  # FTE represented (or headcount, for unweighted)
        "is_primary": "boolean",
        "target_source": "string",
        "target_date": "string",
    },
    # --- curated --------------------------------------------------------------------------
    # One row per advert. Core columns; then weight_<spec> and sector columns may follow.
    "vacancies": {
        **ADVERT_CORE,
        "weight": "float64",  # PRIMARY weight spec (FTE; headcount if unweighted). Null = out
    },
    # One row per task. Core columns; then score_<model> for non-primary models may follow.
    "tasks": {
        "vacancy_id": "string",
        "task_index": "Int64",
        "task_text": "string",
        "task_label": "string",
        "exposure_score": "float64",  # the sector's primary model
        "score_model": "string",
        "scoring_run_id": "string",
    },
    # One row per advert x methodology version x scoring model.
    # £ columns are PER UNIT OF WEIGHT, so cuts multiply by whichever weight spec they use.
    "metrics": {
        "vacancy_id": "string",
        "methodology_version": "string",
        "score_model": "string",
        "n_tasks": "Int64",
        "decay": "float64",
        "mean_exposure": "float64",
        "var_exposure": "float64",
        "mean_exposure_uniform": "float64",
        "var_exposure_uniform": "float64",
        "auto_task_share": "float64",
        "quadrant": "string",  # primary threshold basis (fixed)
        "quadrant_sector_quartiles": "string",  # sensitivity basis
        "opportunity_per_fte_gbp": "float64",
        "cashable_per_fte_gbp": "float64",
        "cashable_share_per_fte_gbp": "float64",
        "non_cashable_per_fte_gbp": "float64",
        "non_cashable_sq_per_fte_gbp": "float64",  # non-cashable under the sensitivity basis
        "cashable_sq_per_fte_gbp": "float64",  # cashable under the sensitivity basis
    },
}

KEYS: dict[str, list[str]] = {
    "adverts": ["vacancy_id"],
    "task_scores": ["vacancy_id", "task_index"],
    "model_scores": ["vacancy_id", "task_index", "model"],
    "weights": ["vacancy_id", "weight_spec"],
    "vacancies": ["vacancy_id"],
    "tasks": ["vacancy_id", "task_index"],
    "metrics": ["vacancy_id", "methodology_version", "score_model"],
}
CURATED_TABLES = ("vacancies", "tasks", "metrics")
WIDE_TABLES = ("vacancies", "tasks")  # core columns first, extra columns allowed after

ID_PREFIX = {
    "civil-service": "cs-",
    "nhs": "nhs-",
    "education": "edu-",
    "local-govt": "lg-",
    "police": "pol-",
    "public-corporations": "pc-",
}

# The quadrant labels in display order.
QUADRANTS = [
    "Displacement-type",
    "Augmentation-type",
    "Concentrated",
    "Insulated",
    "Unclassified",
    "Unscored",
]


class ContractError(ValueError):
    pass


def prefixed_id(sector: str, ids: pd.Series) -> pd.Series:
    """Add the sector prefix to ids that don't already carry it."""
    p = ID_PREFIX[sector]
    s = ids.astype("string")
    return s.where(s.str.startswith(p), p + s)


def model_column(model: str) -> str:
    """Column name for a model's task score in curated tasks: gpt-oss-120b -> score_gpt_oss_120b."""
    return "score_" + re.sub(r"[^0-9a-z]+", "_", model.lower()).strip("_")


def conform(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return `df` with the contract columns first, in order, with contract dtypes.

    Missing columns are added as nulls. Extra columns are kept after the core ones for the
    wide curated tables (vacancies, tasks) and are an error otherwise. Duplicate keys are an
    error.
    """
    spec = TABLES[table]
    extra = [c for c in df.columns if c not in spec]
    if extra and table not in WIDE_TABLES:
        raise ContractError(f"{table}: columns not in the contract: {sorted(extra)}")
    out = pd.DataFrame(index=df.index)
    for col, dtype in spec.items():
        s = df[col] if col in df.columns else pd.Series(None, index=df.index, dtype="object")
        if dtype.startswith("datetime"):
            s = pd.to_datetime(s, errors="coerce")
        elif dtype == "float64":
            s = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            s = s.astype(dtype)
        out[col] = s
    for col in extra:
        out[col] = df[col]
    dup = out.duplicated(KEYS[table])
    if dup.any():
        raise ContractError(f"{table}: {int(dup.sum()):,} duplicate keys on {KEYS[table]}")
    return out.reset_index(drop=True)


def conform_ext(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-specific advert columns: free-form, keyed uniquely on vacancy_id, and never
    reusing a core column name."""
    if "vacancy_id" not in df.columns:
        raise ContractError("adverts_ext: vacancy_id missing")
    if df["vacancy_id"].duplicated().any():
        raise ContractError("adverts_ext: duplicate vacancy_id")
    clash = (set(df.columns) - {"vacancy_id"}) & set(TABLES["vacancies"])
    if clash or any(c.startswith("weight") for c in df.columns):
        raise ContractError(f"adverts_ext: reuses reserved column names {sorted(clash)}")
    out = df.copy()
    out["vacancy_id"] = out["vacancy_id"].astype("string")
    return out.reset_index(drop=True)
