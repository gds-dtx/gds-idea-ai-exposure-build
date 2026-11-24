"""Salary estimates for each department × grade × profession in GRID.

Constructs a three-tier lookup, taking the most specific published figure
available for each combination:

    1. dept_grade_prof_mean_salary  — organisation x grade x profession
    2. dept_grade_mean_salary       — organisation x grade
    3. grade_mean_salary            — grade, whole Civil Service

All three are read directly from published CS Statistics tables rather than
derived, so each is the Statistics Authority's own figure for that grain. Mean
salary is suppressed independently of FTE — a cell may report headcount but not
pay — so the tiers exist to fall back gracefully rather than to aggregate.

The final `estimated_salary` takes the first non-missing tier, and
`salary_source` records which one was used.

Copied from gds-idea-ai-exposure (src/salary.py at commit e02cbd4) so the build no longer depends
on that repo. Unchanged except the imports.
"""

import numpy as np
import pandas as pd

from . import cs_stats
from .stats_settings import constants

DIMS = ["cs_org", "cs_grade", "cs_profession"]

# Tier name -> the dimensions it is keyed on, most specific first.
SALARY_TIERS: list[tuple[str, list[str]]] = [
    ("dept_grade_prof_mean_salary", DIMS),
    ("dept_grade_mean_salary", ["cs_org", "cs_grade"]),
    ("grade_mean_salary", ["cs_grade"]),
]


# ---------------------------------------------------------------------------
# Tier accessors
# ---------------------------------------------------------------------------


def get_salary_by_dept_grade_profession() -> pd.DataFrame:
    """Published mean salary by organisation x grade x profession.

    Each source row is already a cell at this grain. Aggregation applies
    where profession harmonisation (remap + bundling) or
    reassign_unknown_grade_by_salary merges published rows together (the
    latter matching within the same (org, profession) pair only), in which
    case the salary is FTE-weighted.
    """
    df = (
        cs_stats.load_margin(
            constants.CS_STATS_BROWSER_URLS["salary_by_dept_grade_prof"]["filename"]
        )
        .pipe(cs_stats.filter_for_orgs)
        .rename(
            columns={
                "Organisation": "cs_org",
                "Responsibility_level_ungrouped": "cs_grade",
                "Profession_of_post": "cs_profession",
                "Mean_salary": "dept_grade_prof_mean_salary",
            }
        )
        .pipe(cs_stats.normalise_professions, profession_col="cs_profession")
        .pipe(
            cs_stats.reassign_unknown_grade_by_salary,
            salary_col="dept_grade_prof_mean_salary",
            additional_group_cols=["cs_profession"],
        )
    )
    return cs_stats.fte_weighted_salary(df, DIMS, salary_col="dept_grade_prof_mean_salary")


def get_salary_by_dept_grade() -> pd.DataFrame:
    """Published mean salary by organisation x grade.

    reassign_unknown_grade_by_salary merges rows here (nearest-salary,
    department-specific), so aggregation may combine an Unknown row into
    whichever known grade its salary is closest to for that department.
    """
    df = (
        cs_stats.load_margin(constants.CS_STATS_BROWSER_URLS["salary"]["filename"])
        .pipe(cs_stats.filter_for_orgs)
        .rename(
            columns={
                "Organisation": "cs_org",
                "Responsibility_level_ungrouped": "cs_grade",
                "Mean_salary": "dept_grade_mean_salary",
            }
        )
        .pipe(cs_stats.reassign_unknown_grade_by_salary, salary_col="dept_grade_mean_salary")
    )
    return cs_stats.fte_weighted_salary(
        df, ["cs_org", "cs_grade"], salary_col="dept_grade_mean_salary"
    )


def get_salary_by_grade() -> pd.DataFrame:
    """Published mean salary by grade, across the whole Civil Service.

    The source table has no organisation dimension, so this cannot be
    restricted to the estimation population, and Unknown is folded into AO
    via the BLANKET rule (fold_unknown_into_ao) -- consistent with
    cs_stats.get_grade_distribution, which reads the SAME source table for
    the raking margin. Both must apply identically, or this reintroduces
    exactly the raking/salary population mismatch this fix was meant to
    resolve (just for the national tier instead of the department tier).
    """
    df = (
        cs_stats.load_margin(constants.CS_STATS_BROWSER_URLS["grade"]["filename"])
        .rename(
            columns={
                "Responsibility_level_ungrouped": "cs_grade",
                "Mean_salary": "grade_mean_salary",
            }
        )
        .pipe(cs_stats.fold_unknown_into_ao)
    )
    return cs_stats.fte_weighted_salary(df, ["cs_grade"], salary_col="grade_mean_salary")


# ---------------------------------------------------------------------------
# Lookup construction
# ---------------------------------------------------------------------------


def build_salary_lookup(combos: pd.DataFrame) -> pd.DataFrame:
    """Attach all three salary tiers to a frame of dimension combinations.

    Args:
        combos: unique (cs_org, cs_grade, cs_profession) combinations, as
            plain strings.

    Returns:
        `combos` with three tier columns added, one per level of specificity.
    """
    tiers = [
        (get_salary_by_dept_grade_profession(), DIMS, "dept_grade_prof_mean_salary"),
        (get_salary_by_dept_grade(), ["cs_org", "cs_grade"], "dept_grade_mean_salary"),
        (get_salary_by_grade(), ["cs_grade"], "grade_mean_salary"),
    ]
    out = combos.copy()
    for tier_df, keys, col in tiers:
        out = out.merge(tier_df[keys + [col]], on=keys, how="left")
    return out


def coalesce_salary(lookup: pd.DataFrame) -> pd.DataFrame:
    """Take the most specific non-missing tier, recording which was used.

    Adds `estimated_salary` and `salary_source`. Combinations with no salary at
    any tier retain NaN, which downstream code must handle explicitly rather
    than imputing.
    """
    out = lookup.copy()
    out["estimated_salary"] = np.nan
    out["salary_source"] = pd.NA

    for col, keys in SALARY_TIERS:
        fill = out["estimated_salary"].isna() & out[col].notna()
        out.loc[fill, "estimated_salary"] = out.loc[fill, col]
        out.loc[fill, "salary_source"] = "x".join(k.removeprefix("cs_") for k in keys)
    return out
