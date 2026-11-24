"""Raking the GRID adverts to Civil Service Statistics: the estimation population, the 1D IPF
weights and the 3D post-stratified weights.

Copied from gds-idea-ai-exposure (src/rake.py at commit e02cbd4, the version release v2's
weights were built with) so the build no longer depends on that repo. Unchanged except the
imports (no downloads here; tables come from stats_settings.use_stats_dir).
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from . import cs_stats
from .stats_settings import constants


def build_base(
    grid: pd.DataFrame,
    dims: list[str] | None = None,
    id_col: str = "vacancy_id",
    njobs_col: str = "number_of_jobs_available",
) -> pd.DataFrame:
    """Vacancy-level frame with the raking dimensions as plain strings.

    Categorical dtypes are cast to str so that groupby keys match the plain
    strings used in the published margin dictionaries, and so `observed=True`
    behaves predictably.
    """
    dims = dims or ["cs_org", "cs_grade", "cs_profession"]
    cols = [id_col, *dims, njobs_col]
    base = grid[cols].copy()
    for c in dims:
        base[c] = base[c].astype(str)
    return base


def get_estimation_margins(
    excluded_orgs: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """1D margins over the estimation population only.

    The published 1D grade and profession files are organisation-agnostic and
    therefore cover the whole Civil Service. To restrict them to the departments
    we can actually estimate for, we collapse the corresponding 2D files
    (which do carry organisation) after excluding out-of-scope departments.

    Returns normalised (org, grade, profession) distributions summing to 1.0
    over the same population.
    """
    excluded = set(excluded_orgs or [])

    org_dist = {
        k: v for k, v in cs_stats.get_department_fte_distribution().items() if k not in excluded
    }

    dg = cs_stats.get_dept_grade_distribution()
    dg = dg[~dg["cs_org"].isin(excluded)]
    grade_dist = dg.groupby("cs_grade", observed=True)["FTE"].sum().to_dict()

    dp = cs_stats.get_dept_profession_distribution()
    dp = dp[~dp["cs_org"].isin(excluded)]
    prof_dist = dp.groupby("cs_profession", observed=True)["FTE"].sum().to_dict()

    return (
        cs_stats.normalise_distribution(org_dist),
        cs_stats.normalise_distribution(grade_dist),
        cs_stats.normalise_distribution(prof_dist),
    )


def get_normalised_target_distributions() -> tuple[
    dict[str, float], dict[str, float], dict[str, float]
]:
    """
    Normalises grade, department, and profession distributions to sum to 1.0.
    """
    # Get raw counts of FTEs per dimension
    org_dist = cs_stats.get_department_fte_distribution()
    grade_dist = cs_stats.get_grade_distribution()
    prof_dist = cs_stats.get_profession_distribution()

    # Normalise
    norm_org = cs_stats.normalise_distribution(org_dist)
    norm_grade = cs_stats.normalise_distribution(grade_dist)
    norm_prof = cs_stats.normalise_distribution(prof_dist)

    return norm_org, norm_grade, norm_prof


def filter_to_target_strata(
    df: pd.DataFrame,
    norm_org: dict[str, float],
    norm_grade: dict[str, float],
    norm_prof: dict[str, float],
) -> pd.DataFrame:
    """
    Filters the sample to remove records whose department/grade/profession
    doesn't appear in the target distributions.
    """
    before = len(df)

    df = df[
        df["cs_org"].isin(norm_org)
        & df["cs_grade"].isin(norm_grade)
        & df["cs_profession"].isin(norm_prof)
    ].reset_index(drop=True)

    after = len(df)
    dropped = before - after
    print(f"Filtered out {dropped:,} rows ({dropped / before:.2%} of data)")
    print(f"Remaining: {after:,} rows ({df['vacancy_id'].nunique():,} unique vacancies)")

    return df


def find_overextrapolated_departments(
    base_matched: pd.DataFrame,
    threshold: float = constants.EXTRAPOLATION_THRESHOLD,
    verbose: bool = True,
) -> list[str]:
    """Departments where each surviving vacancy would represent >= `threshold` FTE.

    Applied once, to the matched sample, so both specifications describe the same
    population.
    """
    fte = pd.Series(cs_stats.get_department_fte_distribution(), name="fte")
    n = base_matched.groupby("cs_org", observed=True).size().rename("n_matched")
    extrap = pd.concat([fte, n], axis=1).fillna({"n_matched": 0})
    extrap["fte_per_vac"] = extrap["fte"] / extrap["n_matched"].replace(0, np.nan)

    excluded = extrap.index[
        (extrap["n_matched"] == 0) | (extrap["fte_per_vac"] >= threshold)
    ].tolist()
    if verbose:
        print(f"\nExcluded at >={threshold:g} FTE per vacancy: {len(excluded)} dept(s)")
        print(extrap.loc[excluded].round(1).to_string())
    return excluded


def calculate_raked_weights(
    df: pd.DataFrame,
    *,
    norm_org: dict[str, float],
    norm_grade: dict[str, float],
    norm_prof: dict[str, float],
    covered_fte: float,
    njobs_col: str = "number_of_jobs_available",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.Series:
    """Iterative Proportional Fitting to the three 1D marginals.

    Weights are scaled so they sum to `covered_fte` — the FTE of the
    organisations matched in GRID, not the whole Civil Service.
    """
    w = df[njobs_col].to_numpy(dtype=float)

    dept_groups = list(df.groupby("cs_org", observed=True).indices.items())
    grade_groups = list(df.groupby("cs_grade", observed=True).indices.items())
    prof_groups = list(df.groupby("cs_profession", observed=True).indices.items())

    for iteration in range(max_iter):
        w_old = w.copy()

        for groups, target in (
            (dept_groups, norm_org),
            (grade_groups, norm_grade),
            (prof_groups, norm_prof),
        ):
            w_sum = w.sum()
            for key, idx in groups:
                g_sum = w[idx].sum()
                if g_sum:
                    w[idx] *= target[key] / (g_sum / w_sum)

        if np.allclose(w, w_old, rtol=tol, atol=0):
            print(f"Converged after {iteration + 1} iterations")
            break
    else:
        print(f"⚠️ Did not converge after {max_iter} iterations")

    return pd.Series(w * (covered_fte / w.sum()), index=df.index)


def get_scoring_coverage() -> dict[str, float]:
    """Coverage of the population we cleaned, batched and scored.
    Fixed across all weighting specifications."""
    total = cs_stats.get_total_ukcs_fte()
    covered = sum(cs_stats.get_department_fte_distribution().values())
    return {
        "total_ukcs_fte": total,
        "scored_fte": covered,
        "share_of_ukcs": covered / total,
    }


class PostStratResult(NamedTuple):
    """Post-stratification weights, with the coverage assessment behind them."""

    weights: pd.Series  # aligned to df.index; NaN where unweightable
    coverage: pd.Series  # % of published FTE weightable, per department
    reportable: list[str]  # departments meeting the coverage threshold
    n_cells: int  # cells with both a target and sample


def calculate_post_stratified_weights(
    df: pd.DataFrame,
    *,
    joint: pd.DataFrame,
    published_fte: pd.Series,
    coverage_threshold: float = 80.0,
    org_col: str = "cs_org",
    grade_col: str = "cs_grade",
    prof_col: str = "cs_profession",
    njobs_col: str = "number_of_jobs_available",
    verbose: bool = True,
) -> PostStratResult:
    """Post-stratify to a published organisation × grade × profession joint.

    Where the full joint distribution is published there is nothing to infer, so
    no iteration is required: each vacancy's weight is its cell's published FTE,
    allocated across the vacancies in that cell in proportion to
    number_of_jobs_available:

        w_ijk(v) = (FTE_ijk / sum_of_njobs_in_cell_ijk) * njobs(v)

    This reproduces every published cell's FTE total exactly and makes no
    independence assumption, but coverage is incomplete: disclosure control
    suppresses small cells, and a vacancy in a suppressed cell has no target
    to be weighted against.

    Because suppression is non-random — it removes a department's atypical
    grade–profession combinations — departments are only weighted where at least
    `coverage_threshold` per cent of their published FTE falls in a cell that
    also contains sample. Vacancies in departments below the threshold, or in
    suppressed cells, receive NaN.

    Args:
        df: vacancy-level frame containing the three dimension columns and
            number_of_jobs_available.
        joint: published joint distribution with the three dimension columns and
            an `FTE` column. Should already be restricted to the estimation
            population (see `get_dept_grade_profession_distribution`).
        coverage_threshold: minimum percentage of departmental FTE that must be
            weightable for the department to be included.

    Returns:
        A Series aligned to `df.index`, with NaN where no weight can be assigned.
    """
    dims = [org_col, grade_col, prof_col]

    # Cells that exist in the published joint
    joint_cells = set(map(tuple, joint[dims].values))
    has_cell = np.array([tuple(row) in joint_cells for row in df[dims].to_numpy()])

    # Cells that have both a published target and at least one sample vacancy.
    # Denominator is now the SUM of number_of_jobs_available in the cell, not
    # a row count -- so a bulk campaign correctly counts as many roles, not one.
    counts = (
        df.loc[has_cell]
        .groupby(dims, observed=True)[njobs_col]
        .sum()
        .rename("n_roles")
        .reset_index()
    )
    cells = joint.merge(counts, on=dims, how="inner")
    cells["fte_per_role"] = cells["FTE"] / cells["n_roles"]

    ## Departmental coverage: share of published FTE sitting in a usable cell
    fte_usable = cells.groupby(org_col, observed=True)["FTE"].sum()
    coverage = (100 * fte_usable / published_fte).reindex(published_fte.index).fillna(0.0)
    reportable = coverage.index[coverage >= coverage_threshold].tolist()

    usable = cells[cells[org_col].isin(reportable)]
    merged = df[dims + [njobs_col]].merge(usable[dims + ["fte_per_role"]], on=dims, how="left")
    weights = merged["fte_per_role"] * merged[njobs_col]
    weights.index = df.index

    if verbose:
        n_dropped_org = len(coverage) - len(reportable)
        n_weighted = int(weights.notna().sum())
        print(
            f"  3D: {len(cells):,} usable cells; "
            f"{len(reportable)} of {len(coverage)} departments at "
            f">={coverage_threshold:g}% coverage"
        )
        if n_dropped_org:
            worst = coverage[coverage < coverage_threshold].nsmallest(5).round(1)
            print(f"      {n_dropped_org} department(s) below threshold, worst: {dict(worst)}")
        print(
            f"      {n_weighted:,} of {len(df):,} vacancies weighted "
            f"({100 * n_weighted / len(df):.1f}%), "
            f"summing to {weights.sum():,.0f} FTE"
        )
        missing_from_joint = set(published_fte.index) - set(joint[org_col].unique())
        print(f"Departments in published_fte but absent from joint: {len(missing_from_joint)}")
        print(sorted(missing_from_joint))

    return PostStratResult(
        weights=weights.rename("weight_3d"),
        coverage=coverage,
        reportable=reportable,
        n_cells=len(cells),
    )


def verify_margin_reproduction(
    weights: pd.Series,
    dims: pd.DataFrame,
    targets: dict[str, dict[str, float]],
) -> pd.Series:
    """Check that weights reproduce each one-way margin.

    Returns max absolute error per dimension, in FTE. Raking guarantees these
    are ~0 on convergence, so a non-trivial value indicates non-convergence or
    a mismatch between the targets used to build and to verify.
    """
    total = weights.sum()
    errors = {}
    for col, target in targets.items():
        actual = weights.groupby(dims[col], observed=True).sum().dropna()
        expected = (pd.Series(target) * total).reindex(actual.index)
        errors[col] = (actual - expected).abs().dropna().max()
    return pd.Series(errors, name="max_abs_error_fte")


def verify_cell_reproduction(
    weights: pd.Series,
    dims: pd.DataFrame,
    joint: pd.DataFrame,
    dim_cols: list[str],
) -> tuple[float, int]:
    """Check post-stratified weights reproduce every cell they cover.

    Returns (max absolute error in FTE, number of cells checked).
    """
    actual = weights.groupby([dims[c] for c in dim_cols], observed=True).sum(min_count=1).dropna()
    target = joint.set_index(dim_cols)["FTE"].reindex(actual.index)
    err = (actual - target).abs().dropna()
    return float(err.max()), len(err)


def effective_sample_size(w: pd.Series) -> float:
    """Kish's effective sample size: (Σw)² / Σw².

    The size of a simple random sample carrying equivalent statistical
    information. Falls below the nominal sample size as weights become more
    dispersed.
    """
    w = w.dropna()
    return float(w.sum() ** 2 / (w**2).sum())


def design_effect(w: pd.Series) -> float:
    """Variance inflation from weighting: n / n_eff.

    Standard errors on weighted estimates scale as sqrt(deff) relative to an
    unweighted sample of the same nominal size.
    """
    w = w.dropna()
    return len(w) / effective_sample_size(w)


if __name__ == "__main__":
    scoring = get_scoring_coverage()

    print("=" * 60)
    print("COVERAGE: scored population")
    print("=" * 60)
    print(f"Total UK Civil Service FTE:   {scoring['total_ukcs_fte']:>10,.0f}")
    print(f"Matched departments FTE:      {scoring['scored_fte']:>10,.0f}")
    print(
        f"Excluded (unmatched orgs):    {scoring['total_ukcs_fte'] - scoring['scored_fte']:>10,.0f}"
    )
    print(f"Coverage:                     {100 * scoring['share_of_ukcs']:>9.2f}%")

    # Which organisations fall outside the cross-walk, and how big are they?
    org_filename = constants.CS_STATS_BROWSER_URLS["organisation"]["filename"]
    fte_df = cs_stats.load_margin(org_filename)
    excluded = fte_df[~fte_df["Organisation"].isin(cs_stats.retrieve_org_list())]

    print(f"\nLargest unmatched organisations ({len(excluded)} total):")
    print(
        excluded[["Organisation", "FTE"]]
        .sort_values("FTE", ascending=False)
        .head(10)
        .to_string(index=False)
    )
