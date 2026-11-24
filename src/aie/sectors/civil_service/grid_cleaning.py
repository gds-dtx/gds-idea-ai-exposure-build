"""GRID cleaning (release v2, step 0): raw GRID export + profession file -> cleaned adverts.

Copied from gds-idea-ai-exposure (src/data_cleaning.py at commit e246377, the version whose output
release v2 was scored and post-processed from), so the build no longer depends on that repo.
Comments that quoted figures were reworded (public repo); the code is unchanged. Only
the file loading takes paths now; the cleaning steps are unchanged. Uses the mapping tables as
they were at that commit (mappings.CLEANING_DEPARTMENTS / CLEANING_GRADE_MAPPING).
"""

import html
import re
from collections.abc import Iterable
from types import SimpleNamespace

import pandas as pd

from .mappings import (
    CLEANING_DEPARTMENTS,
    CLEANING_GRADE_MAPPING,
    PROFESSION_MAPPING,
    unwanted_job_descriptions,
)

constants = SimpleNamespace(
    DEPARTMENTS=CLEANING_DEPARTMENTS,
    GRADE_MAPPING=CLEANING_GRADE_MAPPING,
    PROFESSION_MAPPING=PROFESSION_MAPPING,
    unwanted_job_descriptions=unwanted_job_descriptions,
)

# ---------------------------------------------------------------------------
# Small, generic helpers
# ---------------------------------------------------------------------------

_SMART_PUNCTUATION_MAP = {
    "\u2018": "'",
    "\u2019": "'",  # Smart single quotes
    "\u201c": '"',
    "\u201d": '"',  # Smart double quotes
    "\u2013": "-",
    "\u2014": "-",  # En and Em dashes
    "\xa0": " ",  # Non-breaking space
}

# ---------------------------------------------------------------------------
# Patterns for cleaning messy markup in job descriptions.
#
# GRID adverts appear in at least three markup dialects:
#   1. Standard HTML          — <p>...</p>, <strong>...</strong>
#   2. Encoded HTML entities  — &lt;iframe ...&gt;
#   3. BBCode-style           — [p][b]...[/b][/p], [color="black"]...
# ---------------------------------------------------------------------------

_TAG_PATTERN = re.compile(
    r'<strong data-lexical-text="true">\[</strong>.*?'
    r'<strong data-lexical-text="true">\]</strong>'  # legacy lexical wrapper
    r"|&lt;[^&]*?&gt;"  # encoded HTML tags
    r"|<[^>]+>",  # raw HTML tags
    flags=re.DOTALL,
)

# BBCode-style tags: [p], [/p], [b], [color="black"], [p style="..."], etc.
# Requires a letter after the (optional) slash so we don't clobber legitimate
# bracketed content like citation markers or [optional] labels in duty lists.
_BBCODE_PATTERN = re.compile(r"\[/?[a-zA-Z][^\[\]]*\]")


def skewer_department(department_name: str) -> str:
    """Turn a department into kebab-case, e.g. 'Department for Education' -> 'department-for-education'."""
    cleaned_name = department_name.replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-zA-Z0-9_]+", "-", cleaned_name).lower().strip("-")


def clean_job_text(text: str) -> str:
    """Cleans up a job description's messy HTML text."""
    if not isinstance(text, str):
        return text

    # Normalise smart punctuation
    for smart, straight in _SMART_PUNCTUATION_MAP.items():
        text = text.replace(smart, straight)

    # HTML entity decoding (doesn't fix &amp; or &nbsp; via encoding alone)
    text = html.unescape(text)

    # Strip tags (content, not encoding)
    text = _TAG_PATTERN.sub(" ", text)

    # Strip BBCode-style tags
    text = _BBCODE_PATTERN.sub(" ", text)

    # Whitespace normalisation
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# GRID loading
# ---------------------------------------------------------------------------


def load_raw_grid(grid_file, professions_file) -> pd.DataFrame:
    """Loads the raw GRID data from the 2026 update, joining on profession."""
    raw_grid = pd.read_csv(grid_file, encoding="utf-8-sig")
    professions = pd.read_csv(professions_file, encoding="utf-8-sig")
    return raw_grid.merge(professions, how="inner", on="vacancy_id")


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip trailing `_desc` suffix from column names."""
    df = df.copy()
    df.columns = df.columns.str.replace(r"_desc$", "", regex=True)
    return df


def _reassign_known_departments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known department / profession quirks in the raw GRID:
      - FCO was subsumed into FCDO.
      - HMPPS records most vacancies with `profession = 'Other'`, but published
        workforce stats show it's almost entirely Operational Delivery. We reassign
        only the 'Other'-tagged rows and preserve any genuine values.
    """
    df = df.copy()

    df["department"] = df["department"].replace(
        "Foreign and Commonwealth Office",
        "Foreign, Commonwealth & Development Office",
    )

    hmpps_other_mask = (df["department"] == "HM Prison & Probation Service") & (
        df["profession"] == "Other"
    )
    df.loc[hmpps_other_mask, "profession"] = "Operational Delivery Profession"

    return df


def _drop_empty_job_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove vacancies with a missing job description."""
    return df.dropna(subset=["job_description"]).copy()


def _filter_to_matched_departments(df: pd.DataFrame) -> pd.DataFrame:
    """Subset GRID to departments matched in CS stats and attach `cs_org`."""
    matched_depts = [d["grid_name"] for d in constants.DEPARTMENTS.values()]
    grid_to_cs_org = {
        info["grid_name"]: info["cs_bulletin_org_name"] for info in constants.DEPARTMENTS.values()
    }
    mask = df["department"].isin(matched_depts).to_numpy()
    df = df.loc[mask].copy()
    df["cs_org"] = df["department"].map(grid_to_cs_org)
    return df


def _filter_to_matched_professions(df: pd.DataFrame) -> pd.DataFrame:
    """Map GRID professions to CS professions and drop unmapped / niche rows."""
    NICHE_AND_UNKNOWN_PROFESSIONS = {"Clinical", "Occupational Psychology"}

    df = df.copy()
    df["cs_profession"] = df["profession"].map(constants.PROFESSION_MAPPING)
    df.loc[
        df["cs_profession"].isin(NICHE_AND_UNKNOWN_PROFESSIONS),
        "cs_profession",
    ] = "Other"

    keep = (
        df["cs_profession"].notna() & ~df["cs_profession"].isin(["Not reported", "invalid"])
    ).to_numpy()
    return df.loc[keep].copy()


def _filter_to_matched_grades(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Map GRID job grades to CS responsibility levels and drop invalid rows.

    Note on non-standard grades: 'Other' and 'Industrial' are retained as their
    own categories rather than mapped to AO. Many NDPBs and arm's-length bodies
    (Homes England, Student Loans Company, IPSA, House of Commons) advertise on
    their own pay structures and appear entirely as 'Other'; assigning them AO
    would assert a responsibility level the data does not support and severely
    distorts those organisations' grade profiles.

    These vacancies are retained here — the decision about whether they can be
    weighted is taken at the raking stage, where the published CS Statistics
    margins (which have no 'Other' or 'Industrial' level) determine coverage.
    """
    df = df.copy()
    df["cs_grade"] = df["job_grade"].map(constants.GRADE_MAPPING)

    n_invalid = (df["cs_grade"] == "invalid").sum()
    n_unmapped = df["cs_grade"].isna().sum()

    df = df.loc[(df["cs_grade"] != "invalid").to_numpy()].copy()
    df = df.dropna(subset=["cs_grade"]).copy()

    if verbose and (n_invalid or n_unmapped):
        print(f"     - Grades: dropped {n_invalid:,} 'invalid', {n_unmapped:,} unmapped")
        if n_unmapped:
            missing = set(df["job_grade"].unique()) - set(constants.GRADE_MAPPING)
            if missing:
                print(f"       unmapped job_grade values: {sorted(missing)[:10]}")

    return df


# Departments where an 'Other' job_grade denotes operational staff who sit at
# AO level in the CS responsibility structure. HMPPS advertises prison officers
# and operational support grades as 'Other'; these are genuinely AO-equivalent,
# unlike NDPBs on separate corporate pay structures.
OTHER_IS_AO_DEPARTMENTS = frozenset(
    {
        "HM Prison & Probation Service",
    }
)


def _reassign_operational_other_grades(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Map 'Other' to AO for departments where it denotes operational staff.

    Applied only to departments in `OTHER_IS_AO_DEPARTMENTS`. Elsewhere 'Other'
    is retained, since it typically indicates a non-Civil-Service pay structure
    rather than an AO-equivalent role.
    """
    df = df.copy()
    mask = (df["department"].isin(OTHER_IS_AO_DEPARTMENTS) & (df["cs_grade"] == "Other")).to_numpy()

    if verbose and mask.sum():
        print(
            f"     - Reassigned {mask.sum():,} 'Other' grades to AO in "
            f"{len(OTHER_IS_AO_DEPARTMENTS)} operational department(s)"
        )

    df.loc[mask, "cs_grade"] = "AO"
    return df


def _clean_html_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Apply `clean_job_text` to each named column."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].apply(clean_job_text)
    return df


def _strip_boilerplate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blank out job_description / job_summary cells that match any known boilerplate
    entry in `constants.unwanted_job_descriptions`. Matched using the first 100
    chars of each cleaned template as a signature.
    """
    df = df.copy()
    for unwanted in constants.unwanted_job_descriptions:
        signature = clean_job_text(unwanted.strip())[:100]
        for col in ("job_description", "job_summary"):
            hit = df[col].str.contains(signature, regex=False, na=False)
            df.loc[hit, col] = ""
    return df


def _filter_short_adverts(df: pd.DataFrame, n_chars: int) -> pd.DataFrame:
    """
    Drop vacancies whose job_description is shorter than `n_chars`. Where the
    description is short but a longer summary exists, promote the summary into
    the description before applying the filter.
    """
    df = df.copy()
    desc_short = (df["job_description"].str.len() < n_chars).to_numpy()
    df["_summary_promoted"] = desc_short  # remember which rows were promoted
    df.loc[desc_short, "job_description"] = df.loc[desc_short, "job_summary"]
    keep = (df["job_description"].str.len() >= n_chars).to_numpy()
    return df.loc[keep].copy()


def _combine_description_and_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate description + summary, description first. (skipped for promoted rows)."""
    df = df.copy()
    # Only append the summary where it wasn't already promoted into the description
    append_summary = ~df["_summary_promoted"]
    df["job_description"] = df["job_description"].fillna("")
    df.loc[append_summary, "job_description"] = (
        df.loc[append_summary, "job_description"]
        + "\n\nJob Summary:\n"
        + df.loc[append_summary, "job_summary"].fillna("")
    )
    return df.drop(columns="_summary_promoted")


def _finalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their final, semantically meaningful dtypes."""
    df = df.copy()

    # Identifiers → nullable string
    df["vacancy_id"] = df["vacancy_id"].astype("string")

    # Ordered categorical for grades — matters for raking
    grade_order = ["Other", "Industrial", "AA", "AO", "EO", "HEO", "SEO", "G7", "G6", "SCS level"]
    df["cs_grade"] = pd.Categorical(df["cs_grade"], categories=grade_order, ordered=True)

    # Unordered categoricals
    for col in ("cs_profession", "cs_org", "department"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def clean_raw_grid_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Runs the full sequence of cleaning steps on the raw GRID DataFrame."""
    N_original = len(df)

    df = _standardise_columns(df)
    df = _reassign_known_departments(df)

    df = _drop_empty_job_descriptions(df)
    N_after_empty_filter = len(df)

    df = _filter_to_matched_departments(df)
    N_after_dept_filter = len(df)

    df = _filter_to_matched_professions(df)
    N_after_profession_filter = len(df)

    df = _filter_to_matched_grades(df)
    df = _reassign_operational_other_grades(df)
    N_after_grade_filter = len(df)

    df = _clean_html_columns(df, columns=["job_description", "job_summary"])
    df = _strip_boilerplate(df)
    df = _filter_short_adverts(df, n_chars=200)
    N_after_short_filter = len(df)

    df = _combine_description_and_summary(df)

    df = _finalise_dtypes(df)

    if verbose:
        _print_pipeline_summary(
            N_original=N_original,
            N_after_empty_filter=N_after_empty_filter,
            N_after_dept_filter=N_after_dept_filter,
            N_after_profession_filter=N_after_profession_filter,
            N_after_grade_filter=N_after_grade_filter,
            N_after_short_filter=N_after_short_filter,
            df_final=df,
        )

    return df


def _print_pipeline_summary(
    *,
    N_original: int,
    N_after_empty_filter: int,
    N_after_dept_filter: int,
    N_after_profession_filter: int,
    N_after_grade_filter: int,
    N_after_short_filter: int,
    df_final: pd.DataFrame,
) -> None:
    """Print a step-by-step retention diagnostic for the cleaning pipeline."""

    def pct(n: int) -> float:
        return 100 * n / N_original

    print(f"We begin with {N_original:,} observations. After each filtering step we retain...\n")
    print(
        f"     - Removing missing job descriptions:                     {pct(N_after_empty_filter):.2f}%"
    )
    print(
        f"     - Subsetting to the matched departments:                 {pct(N_after_dept_filter):.2f}%"
    )
    print(
        f"     - Subsetting to the matched professions:                 {pct(N_after_profession_filter):.2f}%"
    )
    print(
        f"     - Subsetting to the matched grades:                      {pct(N_after_grade_filter):.2f}%"
    )
    print(
        f"     - Stripping known boilerplate + removing short adverts:  {pct(N_after_short_filter):.2f}%"
    )
    print(f"     - Concatenating job summaries onto job descriptions...\n")
    print(f"Remaining roles to pass through the LLM: {len(df_final):,}")
    print(
        f"Total number of departments matched between GRID and CS Stats: {df_final['department'].nunique()}"
    )
