"""Civil Service weights (1D IPF and 3D post-stratification) and the salary lookup.

Ported from gds-idea-ai-exposure (src/3_create_raking_weights.py and src/4_get_salaries.py at
commit e02cbd4): the same steps without the file reading and writing. The statistics tables are
read from the folder set by stats_settings.use_stats_dir.

Release v2's weights were built on adverts with at least 3 tasks (the published weights'
metadata records min_tasks_per_vacancy = 3); the sector passes that sample in. Its salary lookup
covers every post-processed advert.
"""

from __future__ import annotations

import pandas as pd

from . import cs_stats, rake, salary
from .stats_settings import constants

DIMS = ["cs_org", "cs_grade", "cs_profession"]


def assert_target_support(target_categories, sample_categories, dim: str) -> None:
    """Every category in a raking target must appear in the sample, and vice versa."""
    unsupported = set(target_categories) - set(sample_categories)
    assert not unsupported, f"{dim}: categories in target but not sample: {sorted(unsupported)}"
    orphaned = set(sample_categories) - set(target_categories)
    assert not orphaned, f"{dim}: categories in sample but not target: {sorted(orphaned)}"


def filter_to_matchable(base: pd.DataFrame) -> pd.DataFrame:
    """Drop adverts whose organisation, grade or profession has no published counterpart (in
    practice the 'Other' and 'Industrial' grades)."""
    norm_org, norm_grade, norm_prof = rake.get_normalised_target_distributions()
    ok = (
        base["cs_org"].isin(norm_org)
        & base["cs_grade"].isin(norm_grade)
        & base["cs_profession"].isin(norm_prof)
    )
    return base[ok].reset_index(drop=True)


def raking_weights(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """weight_1d and weight_3d for every advert in `grid` (NaN where not weighted), and
    diagnostics. Steps as in the original main()."""
    base = rake.build_base(grid)
    base_matched = filter_to_matchable(base)
    excluded_orgs = rake.find_overextrapolated_departments(
        base_matched, constants.EXTRAPOLATION_THRESHOLD
    )
    base_final = base_matched[~base_matched["cs_org"].isin(excluded_orgs)].reset_index(drop=True)
    published_fte = pd.Series(cs_stats.get_department_fte_distribution())
    published_fte = published_fte[~published_fte.index.isin(excluded_orgs)]
    covered_fte = float(published_fte.sum())

    norm_org, norm_grade, norm_prof = rake.get_estimation_margins(excluded_orgs)
    assert_target_support(norm_org.keys(), base_final["cs_org"].unique(), "department")
    assert_target_support(norm_grade.keys(), base_final["cs_grade"].unique(), "grade")
    assert_target_support(norm_prof.keys(), base_final["cs_profession"].unique(), "profession")
    w_1d = rake.calculate_raked_weights(
        base_final,
        norm_org=norm_org,
        norm_grade=norm_grade,
        norm_prof=norm_prof,
        covered_fte=covered_fte,
        max_iter=50,
    )

    joint = cs_stats.get_dept_grade_profession_distribution()
    joint = joint[~joint["cs_org"].isin(excluded_orgs)].copy()
    ps = rake.calculate_post_stratified_weights(
        base_final,
        joint=joint,
        published_fte=published_fte,
        coverage_threshold=constants.COVERAGE_THRESHOLD_3D,
    )
    assigned = base_final[["vacancy_id"]].assign(
        weight_1d=w_1d.to_numpy(), weight_3d=ps.weights.to_numpy()
    )
    weights = grid[["vacancy_id"]].merge(assigned, on="vacancy_id", how="left")
    diagnostics = {
        "excluded_departments": sorted(excluded_orgs),
        "estimation_population_fte": covered_fte,
        "n_weighted_1d": int(weights["weight_1d"].notna().sum()),
        "n_weighted_3d": int(weights["weight_3d"].notna().sum()),
        "fte_1d": float(weights["weight_1d"].sum()),
        "fte_3d": float(weights["weight_3d"].sum()),
        "extrapolation_threshold_fte_per_vacancy": constants.EXTRAPOLATION_THRESHOLD,
        "coverage_threshold_3d_pct": constants.COVERAGE_THRESHOLD_3D,
    }
    return weights, diagnostics


def salary_lookup(grid: pd.DataFrame) -> pd.DataFrame:
    """The three-tier salary lookup for every organisation x grade x profession in `grid`."""
    combos = grid[salary.DIMS].astype(str).drop_duplicates().reset_index(drop=True)
    lookup = salary.coalesce_salary(salary.build_salary_lookup(combos))
    assert not lookup.duplicated(subset=salary.DIMS).any()
    return lookup
