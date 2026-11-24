"""Ingestion layer for the UK Civil Service Statistics data browser.

Downloads, parses and normalises the published margin tables that underpin both
the raking weights (`rake.py`) and the salary lookup (`salary.py`). Neither of
those modules should read the source CSVs directly — everything they consume
passes through here, so that filtering, category bundling and grade handling are
applied identically across all downstream uses.

Source:
    https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/

Conventions applied to every table loaded through `load_margin`:
  - Only in-post staff are retained (excluding, e.g., staff on loan).
  - `[c]` and `[n]` disclosure markers are read as NaN, not as strings.
  - Numeric columns are coerced, so suppressed cells become NaN rather than
    silently forcing a column to object dtype.

Further conventions are applied selectively, and must be applied consistently
wherever the corresponding dimension is used:
  - `PROFESSION_BUNDLE_INTO_OTHER`: niche professions and 'Unknown' are bundled
    into 'Other', matching the treatment applied to the GRID data in
    `data_cleaning.py`.
  - The 'Unknown' responsibility level is handled DIFFERENTLY depending on
    whether the table being built has an organisation dimension, and this
    split is DELIBERATE, not an inconsistency to eliminate:
      - National, org-LESS tables (e.g. the whole-Civil-Service grade margin
        used for the 1D specification): `fold_unknown_into_ao` applies a
        BLANKET 'Unknown' -> AO relabelling. This was empirically validated
        against HM Prison and Probation Service, which accounts for the large
        majority of the national Unknown population (a large share of its
        own workforce, paid slightly BELOW its own published AO figure —
        consistent with genuine unreported operational staff functioning at
        AO level). There is no organisation column on these tables, so there
        is no department-specific evidence to reassign against; the blanket
        fold is the only option available here.
      - Department-level tables (dept x grade, dept x grade x profession —
        used for the 3D specification and for all department-aware salary
        lookups): `reassign_unknown_grade_by_salary` compares each
        organisation's 'Unknown' row against that SAME organisation's own
        known-grade salaries (and, for the profession-joint table, within the
        same profession too), reassigning to whichever known grade its salary
        is closest to. This was introduced after finding that the blanket
        HMPPS-derived assumption does NOT hold for at least five other
        departments, where a small amount of Unknown-grade FTE is paid well
        above the department's own AO figure (e.g. at HM Treasury, clearly
        SCS-level pay, not AO). Blanket-folding these into AO would manufacture implausibly
        highly-paid "AO" staff at those departments. Falls back to AO where a
        department (or department-profession pair) has too few known grades
        to compare against, or where the Unknown row's own salary is itself
        suppressed.
    A consequence of this split is that a department's AO figure at the 1D
    (national) level may diverge slightly from its own AO figure at the 3D
    (department) level, for the handful of departments affected. This is
    analogous to the already-documented 1D-vs-3D divergence pattern seen
    elsewhere (e.g. Ministry of Justice) and should be reported as such, not
    treated as a bug.
  - Whichever tier/table is affected, BOTH the raking-facing accessor
    (cs_stats.py) and the salary-facing accessor (salary.py) that read the
    SAME source table must apply the SAME 'Unknown'-handling function, with
    the same arguments. Diverging between the two silently mismatches the
    population raking weights are built from against the population salary
    estimates are built from — see the salary.py code-review finding this
    module's grade-handling was refactored to fix.

Note that the published tables cover the whole Civil Service. Restriction to
matched departments happens via `filter_for_orgs`; restriction to the estimation
population (which additionally excludes departments failing the extrapolation
threshold) happens downstream in `rake.py`.

Copied from gds-idea-ai-exposure (src/cs_stats.py at commit e02cbd4, the version release v2's
weights and salaries were built with) so the build no longer depends on that repo.
Comments that quoted figures were reworded (public repo); the code is unchanged. Unchanged
except that the statistics tables are read from the folder set by stats_settings.use_stats_dir (the release's
reference/ copies) instead of the repo's data folder, and nothing is downloaded here.
"""

import os
import re

import numpy as np
import pandas as pd

from .stats_settings import constants

# Constants ####################################################################

NICHE_PROFESSIONS = frozenset(
    {
        # All six previously-bundled niche professions removed following a
        # fair, like-for-like LLM reclassification test (see conversation,
        # [2026-08-28]): each showed a plausible, department-consistent concentration
        # pattern once given an equal chance to be identified --
        #   Risk Management: widespread across dozens of departments (genuine
        #     cross-cutting function, largest volume of the six)
        #   Clinical: concentrated in HMPPS/MoD/UKHSA/Social Security Scotland/
        #     DHSC (genuine clinical/medical staff)
        #   Inspector of Education and Training: concentrated in Ofsted
        #   Geography: concentrated in UK Hydrographic Office
        #   Actuary: concentrated in Government Actuary's Department
        #   Occupational Psychology: smaller, scattered across large
        #     departments (weakest case of the six, but still plausible, not
        #     random noise)
        # NICHE_PROFESSIONS is now empty; PROFESSION_BUNDLE_INTO_OTHER below
        # still bundles the genuine non-response placeholder ("Unknown") into
        # "Other".
    }
)

PROFESSION_BUNDLE_INTO_OTHER = NICHE_PROFESSIONS | frozenset({"Unknown", "Other"})

# CS Statistics 2026 introduced several standalone professions
# (Cyber, Debt and Grants) which do not exist in GRID's profession
# taxonomy. They are folded into their parent professions to maintain
# compatibility between the vacancy sample and the Civil Service target
# distributions used for raking.
PROFESSION_REMAP = {
    "Cyber": "Digital and Data",
    "Debt": "Finance",
    "Grants": "Operational Delivery",
}


# Download and load #############################################################


def load_raw_cs_browser_marginal_distribution(filename: str) -> pd.DataFrame:
    """Loads the downloaded CS browser data into memory"""
    return pd.read_csv(os.path.join(constants.RAW_DATA_DIR, filename), na_values=["[c]", "[n]"])


def convert_cols_to_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures that the numeric columns are the correct datatype"""
    numeric_cols = ["Headcount", "FTE", "Mean_salary", "Median_salary"]
    cols_present = [c for c in numeric_cols if c in df.columns]
    df[cols_present] = df[cols_present].apply(pd.to_numeric, errors="coerce")
    return df


def filter_for_in_post_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only roles that are in-post"""
    return df[df["Status"] == "In post"].copy()


def load_margin(filename: str) -> pd.DataFrame:
    """Loads, filters, and converts a CS browser marginal distribution."""
    return (
        load_raw_cs_browser_marginal_distribution(filename)
        .pipe(filter_for_in_post_roles)
        .pipe(convert_cols_to_numerics)
    )


def get_cs_stats_vintage() -> str:
    """Extract the CS Statistics Browser release from the configured URLs.

    The browser is published under a versioned path, e.g.
    .../data_browser_2025/... — so the vintage can be read from the URL rather
    than hardcoded, keeping metadata in step when the source is updated.

    Raises ValueError if no vintage can be found, or if the configured URLs
    span more than one release.
    """
    found = set()
    for config in constants.CS_STATS_BROWSER_URLS.values():
        m = re.search(r"data_browser_(\d{4})", config["url"])
        if m:
            found.add(m.group(0))

    if not found:
        raise ValueError(
            "No 'data_browser_YYYY' segment found in CS_STATS_BROWSER_URLS — check the URL format."
        )
    if len(found) > 1:
        raise ValueError(
            f"CS_STATS_BROWSER_URLS span multiple releases: {sorted(found)}. "
            f"All margins must come from the same vintage."
        )
    return found.pop()


# Filtering and normalisation #####################################################


def retrieve_org_list() -> list[str]:
    """Returns a list of organisations we are concerned with from our constants.py module"""
    return [value["cs_bulletin_org_name"] for value in constants.DEPARTMENTS.values()]


def filter_for_orgs(df: pd.DataFrame) -> pd.DataFrame:
    """Filters our marginal distribution for the orgs we are raking to"""
    org_list = retrieve_org_list()

    # Flag any missing organisations
    missing = set(org_list) - set(df["Organisation"].unique())
    if missing:
        print(f"⚠️ {len(missing)} org(s) not found in data: {sorted(missing)}")

    return df[df["Organisation"].isin(org_list)]


def fold_unknown_into_ao(df: pd.DataFrame, grade_col: str = "cs_grade") -> pd.DataFrame:
    """
    Folds Unknown grades into AO. Most unreported FTEs are HMPPS operational
    staff, who function at AO level.
    """
    df = df.copy()
    df[grade_col] = df[grade_col].replace({"Unknown": "AO"})
    return df


def reassign_unknown_grade_by_salary(
    df: pd.DataFrame,
    org_col: str = "cs_org",
    grade_col: str = "cs_grade",
    salary_col: str = "Mean_salary",
    fallback_grade: str = "AO",
    min_known_grades: int = 3,
    additional_group_cols: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each (organisation [+ additional_group_cols, if given]) group with an
    'Unknown'-grade row in `df`, reassigns that row's grade to whichever
    KNOWN grade has the CLOSEST mean salary within that SAME group.

    additional_group_cols: restrict nearest-salary matching to rows sharing
    BOTH org_col and these additional columns (e.g. ["cs_profession"] for
    the dept x grade x profession joint table), so an Unknown row is never
    matched against a genuinely unrelated profession's salary structure.
    Leave as None for tables with no further dimension (e.g. dept x grade).

    Falls back to `fallback_grade` if the group has fewer than
    `min_known_grades` known-grade rows, or if the Unknown row's own salary
    is missing/suppressed.

    IMPORTANT: this function is for DEPARTMENT-LEVEL tables ONLY (those with
    an org_col). Tables with no organisation dimension (e.g. the national
    grade margin) have no group to match within -- use the blanket
    `fold_unknown_into_ao` for those instead; calling this function on an
    org-less table will raise a KeyError.
    """
    df = df.copy()
    group_cols = [org_col] + (additional_group_cols or [])
    is_unknown = df[grade_col] == "Unknown"

    df["unknown_grade_reassigned_to"] = df[grade_col]
    df["unknown_grade_reassignment_method"] = pd.NA

    n_reassigned_nearest = 0
    n_fallback_insufficient = 0
    n_fallback_no_salary = 0

    for group_key, group_df in df[is_unknown].groupby(group_cols, observed=True):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        group_mask = np.logical_and.reduce([df[c] == v for c, v in zip(group_cols, key_tuple)])
        for idx in group_df.index:
            unknown_salary = df.at[idx, salary_col]
            known_mask = group_mask & ~is_unknown & df[salary_col].notna()
            known_rows = df.loc[known_mask]

            if pd.isna(unknown_salary):
                df.at[idx, "unknown_grade_reassigned_to"] = fallback_grade
                df.at[idx, "unknown_grade_reassignment_method"] = "fallback_ao_no_salary"
                n_fallback_no_salary += 1
                continue

            if len(known_rows) < min_known_grades:
                df.at[idx, "unknown_grade_reassigned_to"] = fallback_grade
                df.at[idx, "unknown_grade_reassignment_method"] = "fallback_ao_insufficient_data"
                n_fallback_insufficient += 1
                continue

            distances = (known_rows[salary_col] - unknown_salary).abs()
            nearest_idx = distances.idxmin()
            df.at[idx, "unknown_grade_reassigned_to"] = known_rows.loc[nearest_idx, grade_col]
            df.at[idx, "unknown_grade_reassignment_method"] = "nearest_salary"
            n_reassigned_nearest += 1

    df.loc[is_unknown, grade_col] = df.loc[is_unknown, "unknown_grade_reassigned_to"]

    if verbose:
        n_unknown_total = int(is_unknown.sum())
        print(f"'Unknown'-grade rows processed: {n_unknown_total}")
        print(f"  Reassigned by nearest salary: {n_reassigned_nearest}")
        print(
            f"  Fallback to {fallback_grade} (insufficient known grades): {n_fallback_insufficient}"
        )
        print(f"  Fallback to {fallback_grade} (salary itself missing):     {n_fallback_no_salary}")

    return df


def normalise_distribution(dist: dict[str, float]) -> dict[str, float]:
    """
    Normalises a dictionary of values so that the sum of all values equals 1.0.
    """
    total = sum(dist.values())
    if total == 0:
        return dist
    return {k: v / total for k, v in dist.items()}


def normalise_professions(
    df: pd.DataFrame,
    profession_col: str = "cs_profession",
) -> pd.DataFrame:
    """
    Applies all profession harmonisation rules used throughout the pipeline.

    - Cyber -> Digital and Data
    - Debt -> Finance
    - Grants -> Operational Delivery
    - Unknown/niche professions -> Other
    """
    df = df.copy()

    # Merge newly introduced CS professions into existing categories
    df[profession_col] = df[profession_col].replace(PROFESSION_REMAP)

    # Bundle niche professions and Unknown into Other
    bundle_mask = df[profession_col].isin(PROFESSION_BUNDLE_INTO_OTHER)
    df.loc[bundle_mask, profession_col] = "Other"

    return df


# Margin accessors #####################################################


def get_department_fte_distribution() -> dict[str, float]:
    """FTE by organisation, restricted to organisations matched in GRID.

    No residual bucket: we rake to the covered population (~95% of CS FTE)
    rather than the whole Civil Service, since unmatched organisations have
    no GRID vacancies to weight.
    """
    org_filename = constants.CS_STATS_BROWSER_URLS["organisation"]["filename"]
    fte_df = load_margin(org_filename)
    covered = filter_for_orgs(fte_df).groupby("Organisation", as_index=False)["FTE"].sum()
    return dict(zip(covered["Organisation"], covered["FTE"]))


def get_grade_distribution() -> dict[str, float]:
    """FTE by grade, whole Civil Service (no department dimension).

    Unknown is folded into AO via the BLANKET rule (fold_unknown_into_ao),
    not the department-aware nearest-salary reassignment -- this table has
    no organisation column, so there is no department-specific evidence to
    reassign against. This is deliberately the ONE place the blanket fold
    is still used, consistent with HMPPS dominating the national Unknown
    population; see module docstring for the full rationale.
    """
    grades_df = (
        load_margin(constants.CS_STATS_BROWSER_URLS["grade"]["filename"])
        .rename(columns={"Responsibility_level_ungrouped": "cs_grade", "FTE": "fte"})
        .pipe(fold_unknown_into_ao)[["cs_grade", "fte"]]
        .groupby("cs_grade", as_index=False)["fte"]
        .sum()
    )
    return dict(zip(grades_df["cs_grade"], grades_df["fte"]))


def get_profession_distribution() -> dict[str, float]:
    """
    Loads and cleans the profession distribution of the UKCS from the Stats Browser data.

    'Other', 'Unknown', and the niche professions are bundled together.
    """
    prof_df = (
        load_margin(constants.CS_STATS_BROWSER_URLS["profession"]["filename"])
        .rename(columns={"Profession_of_post": "cs_profession", "FTE": "fte"})[
            ["cs_profession", "fte"]
        ]
        .pipe(normalise_professions)
    )

    # Sum across the now-bundled labels
    prof_df = prof_df.groupby("cs_profession", as_index=False)["fte"].sum()

    return dict(zip(prof_df["cs_profession"], prof_df["fte"]))


def get_dept_grade_distribution() -> pd.DataFrame:
    """Dept x grade FTE.

    Unknown is reassigned to the nearest-salary KNOWN grade for that same
    department (reassign_unknown_grade_by_salary), NOT blanket-folded to AO
    -- validated against HMPPS's own dominant Unknown population (correctly
    lands on AO there) and against five departments where blanket-AO would
    have manufactured implausibly high-paid "AO" staff (see conversation
    for the evidence). This means dept-level AO FTE MAY diverge slightly
    from the national (1D) AO margin for a handful of departments -- an
    intentional divergence, analogous to the already-documented 1D-vs-3D
    pattern (e.g. MoJ), not an inconsistency to eliminate.

    Fully-suppressed cells return NaN (not 0) and are dropped, so they don't
    become spurious zero targets during raking.
    """
    out = (
        load_margin(constants.CS_STATS_BROWSER_URLS["salary"]["filename"])
        .pipe(filter_for_orgs)
        .rename(
            columns={
                "Organisation": "cs_org",
                "Responsibility_level_ungrouped": "cs_grade",
            }
        )
        .pipe(reassign_unknown_grade_by_salary, salary_col="Mean_salary")
        .groupby(["cs_org", "cs_grade"], observed=True)["FTE"]
        .sum(min_count=1)
        .reset_index()
    )
    n_before = len(out)
    out = out.dropna(subset=["FTE"])
    if n_before - len(out):
        print(f"  dept×grade: dropped {n_before - len(out)} fully-suppressed cells")
    return out


def get_dept_profession_distribution() -> pd.DataFrame:
    """Dept × profession FTE, with the same Other/Unknown/niche bundling as 1D."""
    prof_df = (
        load_margin(constants.CS_STATS_BROWSER_URLS["profession_by_dept"]["filename"])
        .pipe(filter_for_orgs)
        .rename(
            columns={
                "Organisation": "cs_org",
                "Profession_of_post": "cs_profession",
            }
        )[["cs_org", "cs_profession", "FTE"]]
    )
    prof_df = normalise_professions(prof_df)

    out = (
        prof_df.groupby(["cs_org", "cs_profession"], observed=True)["FTE"]
        .sum(min_count=1)
        .reset_index()
    )
    n_before = len(out)
    out = out.dropna(subset=["FTE"])
    if n_before - len(out):
        print(f"  dept×profession: dropped {n_before - len(out)} fully-suppressed cells")
    return out


def get_dept_grade_profession_distribution() -> pd.DataFrame:
    """Full dept x grade x profession joint.

    Unknown grade is reassigned to the nearest-salary KNOWN grade WITHIN
    THE SAME (department, profession) pair -- not just the same department
    -- so a profession's Unknown row is never matched against an unrelated
    profession's salary structure. See reassign_unknown_grade_by_salary's
    additional_group_cols.
    """
    joint_df = (
        load_margin(constants.CS_STATS_BROWSER_URLS["salary_by_dept_grade_prof"]["filename"])
        .pipe(filter_for_orgs)
        .pipe(
            reassign_unknown_grade_by_salary,
            org_col="Organisation",
            grade_col="Responsibility_level_ungrouped",
            salary_col="Mean_salary",
            additional_group_cols=["Profession_of_post"],
        )
        .rename(
            columns={
                "Organisation": "cs_org",
                "Responsibility_level_ungrouped": "cs_grade",
                "Profession_of_post": "cs_profession",
            }
        )[["cs_org", "cs_grade", "cs_profession", "FTE"]]
    )
    joint_df = normalise_professions(joint_df)

    out = (
        joint_df.groupby(["cs_org", "cs_grade", "cs_profession"], observed=True)["FTE"]
        .sum(min_count=1)
        .reset_index()
    )
    n_before = len(out)
    out = out.dropna(subset=["FTE"])
    if n_before - len(out):
        print(f"  dept×grade×profession: dropped {n_before - len(out)} fully-suppressed cells")
    return out


def get_total_ukcs_fte() -> float:
    """Total UK Civil Service FTE across all organisations."""
    org_filename = constants.CS_STATS_BROWSER_URLS["organisation"]["filename"]
    return load_margin(org_filename)["FTE"].sum()


def fte_weighted_salary(
    df: pd.DataFrame,
    group_cols: list[str],
    salary_col: str,
    fte_col: str = "FTE",
) -> pd.DataFrame:
    """Combine published rows that share a key after relabelling.

    The CS Statistics Browser publishes each table already aggregated to its own
    grain, so no aggregation across dimensions is required. However, some
    normalisations applied upstream merge rows the Authority kept separate:

      - `fold_unknown_into_ao` (used only for national, org-less tables)
        relabels every 'Unknown' responsibility level as AO, so every (grade)
        group may gain a second row;
      - `reassign_unknown_grade_by_salary` (used for department-level tables)
        relabels each organisation's 'Unknown' row to whichever KNOWN grade
        its salary is closest to for that same organisation (and, for the
        profession-joint table, the same profession too) — so the row a given
        (org, grade[, profession]) group gains is not always AO; it depends on
        that department's own salary evidence, and falls back to AO only where
        there isn't enough of it;
      - `PROFESSION_BUNDLE_INTO_OTHER` collapses niche professions and
        'Unknown' into 'Other', so an (org, grade) may gain several.

    Where this occurs, the combined salary is FTE-weighted:

        sum(salary x fte) / sum(fte)

    over rows where both are present. Salary is suppressed independently of FTE,
    so a row with a masked salary still contributes to the FTE total but not to
    the weighted mean; a group where every salary is masked returns NaN.

    Where a key maps to a single published row — the common case — this returns
    that row's salary unchanged.
    """
    d = df.copy()
    valid = d[salary_col].notna() & d[fte_col].notna()
    d["_num"] = np.where(valid, d[salary_col] * d[fte_col], np.nan)
    d["_den"] = np.where(valid, d[fte_col], np.nan)

    agg = (
        d.groupby(group_cols, observed=True)
        .agg(_num=("_num", "sum"), _den=("_den", "sum"), **{fte_col: (fte_col, "sum")})
        .reset_index()
    )
    agg[salary_col] = np.where(agg["_den"] > 0, agg["_num"] / agg["_den"], np.nan)
    return agg[group_cols + [salary_col, fte_col]]
