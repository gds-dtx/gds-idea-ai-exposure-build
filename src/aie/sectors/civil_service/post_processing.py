"""GRID post-processing (release v2, step 1c).

Copied from gds-idea-ai-exposure (src/data_quality.py at commit c2a10a5, the version release v2
was post-processed with) so the build no longer depends on that repo.
Comments that quoted figures were reworded (public repo); the code is unchanged. The functions are
unchanged; the sector calls them in the order the original notebook did
(1c_grid_post_processing.ipynb). The export_* helpers write spot-check files and are not used.


data_quality.py

Post-processing data-quality module for Module 1c, applied AFTER LLM task
extraction/scoring has already run on the Module 0 (data_cleaning.py)
output. Handles everything triggered by the 2026-08 GRID data drop:

  1. Cleaning the new fields themselves (number_of_jobs_available,
     salary_minimum/maximum_optional, location/business-area fields,
     most_recent_posting_live).
  2. Title-based data-quality flags (test/placeholder data, GRID's own
     "(Copy)"/"(Discard)" markers, junk "error"/"please delete"/"delete -"
     titles).
  3. Duplicate-posting detection: flags (never silently drops) rows that
     look like duplicate/erroneous postings of the same underlying role,
     while protecting genuine bulk recruitment campaigns and correctly
     preferring a properly-run advert over a short-lived erroneous one.
  4. Diagnostic/export helpers for manual review of flagged subsets.
  5. Final filtering: applies the flags above to produce the deduplicated
     dataset ready for downstream raking/monetisation.

Deliberate separation from data_cleaning.py: data_cleaning.py governs what
goes INTO the LLM extraction/scoring step (Module 0) and must not change
now that scoring has already run on its output. This module governs what
happens AFTER scoring.

USAGE:

    from data_cleaning import load_cleaned_grid
    from data_quality import (
        remove_invalid_jobs_available, clean_new_salary_fields,
        clean_new_location_and_business_fields, ensure_most_recent_posting_live,
        flag_test_and_copy_titles, flag_error_titles, assign_duplicate_groups,
        export_duplicate_review_subset, export_njobs_ambiguous_review,
        find_content_key_mismatches, explain_content_key_mismatch,
        inspect_content_key_group, apply_final_filters,
    )

    joint_df = grid.merge(new_df[cols_to_join], on="vacancy_id", how="left")

    joint_df = remove_invalid_jobs_available(joint_df)
    joint_df = clean_new_salary_fields(joint_df)
    joint_df = clean_new_location_and_business_fields(joint_df)
    joint_df = ensure_most_recent_posting_live(joint_df)

    joint_df = flag_test_and_copy_titles(joint_df)
    joint_df = flag_error_titles(joint_df)
    joint_df = assign_duplicate_groups(joint_df)

    export_duplicate_review_subset(joint_df, path="temp.csv")   # optional, for spot-checking

    final_df = apply_final_filters(joint_df)

CHANGELOG (high level -- see individual function docstrings for full detail):
  - Tightened is_test_data / added flag_error_titles to avoid false
    positives on "NHS Test and Trace" / "Fraud and Error" style genuine
    role names.
  - assign_duplicate_groups: exact content-key match -> date-gap
    subgroup split -> title-similarity split -> same-timestamp bulk-batch
    protection -> longest-advertising-window canonical selection.
  - FIXED: imputed posting dates no longer trigger/anchor a false
    "genuine re-advertisement" date-gap split.
  - FIXED: _make_content_key rebuilt to be immune to a pandas gotcha
    where `.str.cat()` silently propagates NA across an ENTIRE
    concatenated key if any single input column has a missing value.
  - FIXED: _normalize_categorical_for_matching forces plain string dtype
    output, since `.map()` on a Categorical-dtype column (e.g.
    `department`, cast to "category" by data_cleaning.py) otherwise
    preserves the Categorical dtype, which breaks string concatenation.
  - FIXED: inspect_content_key_group now checks for a genuine >gap_days
    date-gap explanation (mirroring assign_duplicate_groups' own logic)
    before raising an "unexplained anomaly" warning, in addition to the
    title-similarity check it already had.
  - FIXED: duplicate_group_id is now only assigned to clusters with more
    than one member -- singletons (including ones split off from a
    larger content-key group by date-gap/title-similarity) correctly
    keep duplicate_group_id == -1, consistent with is_duplicate == False.
    This was previously inflating "in a duplicate group" counts (e.g.
    export_duplicate_review_subset) far beyond the true count of rows
    genuinely involved in a duplicate relationship.
  - ADDED (this version): flag_error_titles now also catches titles that
    START with "Delete"/"DELETE" followed by a dash or colon (e.g.
    "Delete - Team Leader: Invoicing...", "DELETE - Performance
    Analyst..."), found via manual review of spot_check_manual_review.csv
    -- the same kind of explicit self-declared "this record should be
    removed" marker as "(Discard)", just phrased as a title prefix
    instead of a suffix.

NOTEBOOK HYGIENE NOTE: if you redefine any function from this module
directly in a notebook cell while iterating, make sure you re-run that
cell (and re-import) before re-running your pipeline -- Python will
silently keep using whichever version was executed last.
"""

import difflib
import re

import pandas as pd

# ===========================================================================
# 1. Cleaning the new GRID fields (2026-08 data drop)
# ===========================================================================


def remove_invalid_jobs_available(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Removes rows with an explicit invalid (zero/negative) number_of_jobs_available
    -- these look like test/placeholder entries, not real postings.

    Rows with a MISSING (NaN) number_of_jobs_available are NOT removed here:
    these are typically pre-existing GRID vacancies that predate this new
    field, or vacancy_ids that didn't match the new-data join, rather than
    test placeholders. They are left as NaN for apply_final_filters to
    default to 1 at the very end (matching pre-2026 implicit behaviour).
    """
    df = df.copy()
    if "number_of_jobs_available" not in df.columns:
        return df

    n_jobs = pd.to_numeric(df["number_of_jobs_available"], errors="coerce")
    n_missing = int(n_jobs.isna().sum())
    invalid_mask = n_jobs.notna() & (n_jobs < 1)
    n_invalid = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    if verbose:
        print(
            f"Role multiplier: removed {n_invalid:,} rows with an explicit "
            f"zero/negative number_of_jobs_available"
        )
        print(
            f"Role multiplier: {n_missing:,} rows have a MISSING "
            f"number_of_jobs_available (not removed -- likely unmatched to new data)"
        )
    return df


def clean_new_salary_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-cast salary_minimum / salary_maximum_optional (arrive clean per GRID team)."""
    df = df.copy()
    for col in ("salary_minimum", "salary_maximum_optional"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_new_location_and_business_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Light trim/normalise for business_area(_detail), vacancy_city/region, approach, postcode."""
    df = df.copy()
    for col in (
        "business_area",
        "business_area_detail",
        "vacancy_city",
        "vacancy_region",
        "approach",
    ):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    if "vacancy_postcode" in df.columns:
        df["vacancy_postcode"] = (
            df["vacancy_postcode"]
            .astype("string")
            .str.upper()
            # ASCII whitespace only: under pandas 3.0 (the original run) `\s` did not match
            # no-break or narrow no-break spaces in this string column; newer pandas does. The
            # explicit class keeps release v2's postcodes the same whichever pandas runs this.
            .str.replace(r"[ \t\n\r\f\v]+", " ", regex=True)
            .str.strip()
        )
    return df


def ensure_most_recent_posting_live(
    df: pd.DataFrame,
    date_col: str = "most_recent_posting_live",
    fallback_col: str = "closing_date",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Guarantee `date_col` is populated for every row, falling back to
    `fallback_col` where missing.

    Coerces both `date_col` and `fallback_col` to proper datetime dtype
    FIRST -- required for downstream date arithmetic (assign_duplicate_groups,
    advertising-window calculations) to work.

    CAVEAT: closing_date is a downstream-in-time proxy, not a true
    "last live" date. Imputed rows are flagged via `{date_col}_imputed`
    so they can be treated differently downstream -- see
    assign_duplicate_groups, which treats an imputed date as unreliable
    evidence for splitting a genuine re-advertisement cycle apart from a
    duplicate.

    MUST be called on the SAME dataframe object (or its direct descendant
    via reassignment) that is later passed into assign_duplicate_groups --
    if assign_duplicate_groups runs on a dataframe that never had this
    function applied, `{date_col}_imputed` won't exist and ALL rows will
    be treated as non-imputed, silently disabling that protection.

    Rows where BOTH date_col and fallback_col are missing/unparseable
    remain null (NaT), and are counted separately in the verbose summary.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[fallback_col] = pd.to_datetime(df[fallback_col], errors="coerce")

    missing = df[date_col].isna()
    df[f"{date_col}_imputed"] = missing & df[fallback_col].notna()
    df.loc[missing, date_col] = df.loc[missing, fallback_col]

    if verbose:
        n_imputed = int(df[f"{date_col}_imputed"].sum())
        n_still_missing = int(df[date_col].isna().sum())
        print(
            f"{date_col}: {n_imputed:,} rows imputed from {fallback_col}; "
            f"{n_still_missing:,} rows still missing (no fallback available)"
        )
    return df


def cap_imputed_posting_dates(
    df: pd.DataFrame,
    date_col: str = "most_recent_posting_live",
    imputed_flag_col: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Caps any IMPUTED `date_col` value at the maximum `date_col` value among
    NON-imputed rows -- i.e. the latest date we have genuine evidence for a
    real posting event. Non-imputed rows are never modified.

    Args:
        df: dataframe containing `date_col` (and, unless
            `imputed_flag_col` is given, `f"{date_col}_imputed"`).
        date_col: the date column to cap.
        imputed_flag_col: name of the boolean imputed-flag column. Defaults
            to `f"{date_col}_imputed"`, matching
            ensure_most_recent_posting_live's own naming convention.

    Raises:
        ValueError if `imputed_flag_col` is not present in `df` -- this
        function REQUIRES ensure_most_recent_posting_live (or an
        equivalent) to have already run on this exact dataframe; it
        cannot distinguish imputed from genuine dates otherwise, and
        capping a GENUINE date would silently discard real information.
    """
    df = df.copy()
    imputed_flag_col = imputed_flag_col or f"{date_col}_imputed"

    if imputed_flag_col not in df.columns:
        raise ValueError(
            f"'{imputed_flag_col}' not found in df -- cap_imputed_posting_dates "
            f"must run AFTER ensure_most_recent_posting_live (or an equivalent "
            f"function producing this flag) on this exact dataframe. Capping "
            f"cannot safely proceed without knowing which dates are imputed."
        )

    is_imputed = df[imputed_flag_col].fillna(False)
    genuine_dates = df.loc[~is_imputed, date_col]

    if genuine_dates.notna().sum() == 0:
        if verbose:
            print(
                f"WARNING: no non-imputed {date_col} values found -- cannot "
                f"determine a ceiling. No capping applied."
            )
        return df

    max_observed_date = genuine_dates.max()

    exceeds_ceiling = is_imputed & (df[date_col] > max_observed_date)
    n_capped = int(exceeds_ceiling.sum())

    df.loc[exceeds_ceiling, date_col] = max_observed_date

    if verbose:
        n_imputed_total = int(is_imputed.sum())
        print(f"{date_col}: latest genuinely-observed date is {max_observed_date}")
        print(
            f"  {n_imputed_total:,} rows are imputed in total; "
            f"{n_capped:,} of these exceeded the observed ceiling and were capped to it "
            f"({100 * n_capped / len(df):.3f}% of all rows)"
        )

    return df


# ===========================================================================
# 2. Title-based data-quality flags
# ===========================================================================

_TEST_TITLE_PATTERN = re.compile(r"\btests?\b", flags=re.IGNORECASE)

_REPEATED_TEST_TITLE_PATTERN = re.compile(r"(?:\btests?\b.*){2,}", flags=re.IGNORECASE)

_LEGITIMATE_TEST_CONTEXT = re.compile(
    r"test\s*(and|&)\s*trace"
    r"|test\s*(and|&)\s*evaluation"
    r"|test\s*(and|&)\s*measur"
    r"|test\s*(and|&)\s*learn"
    r"|test.{0,25}?(engineer|manager|analyst|lead|technician|automation)"
    r"|in\s*test\b"
    r"|test\s*development"
    r"|quality\s*assurance.{0,15}test"
    r"|test\s*delivery"
    r"|test\s*close\s*date"
    r"|online\s*test"
    r"|reading\s*tests?"
    r"|driving\s*test"
    r"|theory\s*test",
    flags=re.IGNORECASE,
)

_EXPLICIT_TEST_MARKERS = re.compile(
    r"do\s*not\s*apply"
    r"|do\s*not\s*use"
    r"|test\s*only"
    r"|test\s*vacancy"
    r"|test\s*advert"
    r"|test\s*external"
    r"|for\s*wordcount"
    r"|tests?\s*closed"
    r"|please\s*delete",
    flags=re.IGNORECASE,
)

_COPY_TITLE_PATTERN = re.compile(r"\(\s*copy\s*\)", flags=re.IGNORECASE)
_DISCARD_TITLE_PATTERN = re.compile(r"\(\s*discard\s*\)", flags=re.IGNORECASE)

# "Fraud and Error" is a genuine DWP/HMRC function name -- must NOT be
# flagged as junk.
_LEGITIMATE_ERROR_CONTEXT = re.compile(
    r"fraud.{0,15}error"
    r"|error.{0,15}resolution"
    r"|error.{0,15}measurement"
    r"|error.{0,10}debt",
    flags=re.IGNORECASE,
)

# High-confidence junk/placeholder "error" markers.
_EXPLICIT_ERROR_MARKERS = re.compile(
    r"please\s*delete"
    r"|do\s*not?\s*use"
    r"|created\s*in\s*error"
    r"|copied\s*in\s*error"
    r"|due\s*to\s*error"
    r"|system\s*error"
    r"|error\s*in\s*system"
    r"|^\s*error\s*$",
    flags=re.IGNORECASE,
)

# NEW: titles that START with "Delete"/"DELETE" followed by a dash or
# colon (e.g. "Delete - Team Leader: Invoicing...", "DELETE - Performance
# Analyst..."). Found via manual review of spot_check_manual_review.csv --
# the same kind of explicit self-declared "this record should be removed"
# marker as "(Discard)", just phrased as a title PREFIX instead of a
# suffix. Anchored to the START of the title (allowing leading
# whitespace/punctuation) specifically so this does NOT match genuine
# role content that happens to mention "delete" elsewhere in a longer
# title (e.g. a hypothetical "Data Retention and Delete Policy Officer"
# would NOT match, since "Delete" is not the first substantive word).
_DELETE_PREFIX_PATTERN = re.compile(r"^\s*delete\s*[-:]", flags=re.IGNORECASE)

# NEW: titles containing "(Dummy)" as a bracketed suffix (e.g. "Ad Hoc EOI
# Analyst Manager(Dummy)", "AD HOC EOI - Project Support Pensions
# Programme (DUMMY)"). Found via manual review of adhoc_review.csv --
# HMRC in particular uses this as an explicit self-declared marker for
# placeholder/test EOI (Expression of Interest) postings, the same kind
# of signal as "(Copy)" or "(Discard)". Deliberately requires the
# parenthesised form specifically -- an unbracketed use of the word
# "dummy" elsewhere in a title (e.g. a hypothetical genuine "Crash Test
# Dummy Design Engineer" role) is too ambiguous to safely treat as a
# placeholder marker without the explicit bracketing convention.
_DUMMY_TITLE_PATTERN = re.compile(r"\(\s*dummy\s*\)", flags=re.IGNORECASE)


def flag_test_and_copy_titles(df: pd.DataFrame, title_col: str = "vacancy_title") -> pd.DataFrame:
    """
    Add `is_test_data`, `title_marked_copy`, and `title_marked_discard`
    flags based on vacancy_title.

    is_test_data (tightened after finding many false positives in an
    earlier word-boundary-only version):
      1. No "test"/"tests" word at all -> False.
      2. Matches a legitimate context (NHS Test and Trace, QA/engineering
         job titles, driving/theory tests, etc.) -> False.
      3. Matches an explicit high-confidence placeholder marker -> True.
      4. Otherwise (ambiguous "test" mention) -> True, but MANUAL REVIEW
         territory -- NOT auto-dropped by apply_final_filters.

    title_marked_copy / title_marked_discard: GRID's own explicit
    annotations. title_marked_discard IS dropped by apply_final_filters
    by default (unambiguous self-declared marker); title_marked_copy
    alone is NOT (many "(Copy)"-titled rows have no matching duplicate
    sibling -- see apply_final_filters docstring).
    """
    df = df.copy()
    titles = df[title_col].fillna("")

    def classify_test(title: str) -> bool:
        if not _TEST_TITLE_PATTERN.search(title):
            return False
        if _LEGITIMATE_TEST_CONTEXT.search(title):
            return False
        if _EXPLICIT_TEST_MARKERS.search(title):
            return True
        return True  # ambiguous -- flagged for manual review, not auto-dropped

    df["is_test_data"] = titles.apply(classify_test)
    df["title_marked_copy"] = titles.str.contains(_COPY_TITLE_PATTERN)
    df["title_marked_discard"] = titles.str.contains(_DISCARD_TITLE_PATTERN)
    df["title_marked_repeated_test"] = titles.str.contains(_REPEATED_TEST_TITLE_PATTERN)
    df["title_marked_dummy"] = titles.str.contains(_DUMMY_TITLE_PATTERN)
    return df


def flag_error_titles(df: pd.DataFrame, title_col: str = "vacancy_title") -> pd.DataFrame:
    """
    Add `title_marked_error`: True for GRID's junk/placeholder "error"/
    "please delete"/"delete -" style titles (e.g. "error please delete",
    "SYSTEM ERROR - DO NOT USE", "Copied in Error do not process",
    "Delete - Team Leader: Invoicing...", "DELETE - Performance
    Analyst..."). Narrow, high-confidence flag -- excludes genuine "Fraud
    and Error" function names, and only matches "Delete"/"DELETE" when it
    is the FIRST substantive word in the title (followed by a dash or
    colon), not merely present somewhere in a longer, genuine title.

    Dropped by apply_final_filters by default, given how unambiguous
    these titles are.
    """
    df = df.copy()
    titles = df[title_col].fillna("")

    def classify(title: str) -> bool:
        if _DELETE_PREFIX_PATTERN.search(title):
            return True
        if not re.search(r"\berror\b|please\s*delete", title, flags=re.IGNORECASE):
            return False
        if _LEGITIMATE_ERROR_CONTEXT.search(title):
            return False
        return bool(_EXPLICIT_ERROR_MARKERS.search(title))

    df["title_marked_error"] = titles.apply(classify)
    return df


# ===========================================================================
# 3. Duplicate-posting detection
# ===========================================================================


def _normalize_title_for_matching(title: str) -> str:
    """Lower-case, strip (Copy)/(Discard)/ref-codes/readvertisement phrases/month-year suffixes, punctuation."""
    t = str(title).lower()
    t = re.sub(r"\(\s*copy\s*\)", " ", t)
    t = re.sub(r"\(\s*discard\s*\)", " ", t)
    t = re.sub(r"\(ref\S*[:\)].*?\)", " ", t)
    t = re.sub(r"re[-\s]?advertisement of \d+", " ", t)
    t = re.sub(
        r"\(?\s*\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{2,4}\b\s*\)?",
        " ",
        t,
    )
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


_DIRECTION_WORDS = frozenset(
    {
        "north",
        "south",
        "east",
        "west",
        "northern",
        "southern",
        "eastern",
        "western",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    }
)


def _titles_indicate_same_role(title_a: str, title_b: str, max_editdist: int = 1) -> bool:
    """
    Decide whether two titles sharing an identical content key plausibly
    refer to the SAME underlying role, vs. genuinely different postings
    that happen to share templated boilerplate text.
    """
    norm_a = _normalize_title_for_matching(title_a)
    norm_b = _normalize_title_for_matching(title_b)
    tokens_a, tokens_b = set(norm_a.split()), set(norm_b.split())

    diff_a = tokens_a - tokens_b
    diff_b = tokens_b - tokens_a

    if not diff_a and not diff_b:
        return True

    if len(diff_a) == 1 and len(diff_b) == 1:
        word_a, word_b = next(iter(diff_a)), next(iter(diff_b))
        if word_a in _DIRECTION_WORDS or word_b in _DIRECTION_WORDS:
            return False
        if not (word_a.isalpha() and word_b.isalpha()):
            return False
        return _levenshtein(word_a, word_b) <= max_editdist

    return False


def _cluster_by_title_similarity(indices: list, titles: pd.Series) -> list:
    """Union-find clustering of `indices` via _titles_indicate_same_role."""
    parent = {i: i for i in indices}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    n = len(indices)
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = indices[a], indices[b]
            if _titles_indicate_same_role(titles.loc[ia], titles.loc[ib]):
                union(ia, ib)

    clusters = {}
    for i in indices:
        root = find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


_DEDUP_KEY_EXACT_COLUMNS = ["department", "job_grade"]
_DEDUP_KEY_LOOSE_COLUMNS = ["profession", "business_area"]
_DEDUP_KEY_COLUMNS = (
    _DEDUP_KEY_EXACT_COLUMNS
    + _DEDUP_KEY_LOOSE_COLUMNS
    + ["salary_minimum", "salary_maximum_optional"]
)


def _safe_str_scalar(x) -> str:
    """
    Converts ANY scalar value to a string, GUARANTEED never to return
    NA/None, regardless of source dtype (float64, nullable Float64,
    object, category, pyarrow-backed, etc).
    """
    if pd.isna(x):
        return "<MISSING>"
    return str(x)


def _normalize_categorical_scalar(x) -> str:
    """Scalar strip+casefold+whitespace-collapse. Never returns NA (missing -> '')."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return re.sub(r"\s+", " ", s)


def _normalize_description_scalar(x) -> str:
    """Scalar description normaliser (curly quotes, bullets, '&'/'and', embedded ref numbers). Never returns NA."""
    if pd.isna(x):
        return ""
    t = str(x).lower()
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\xa0", " ")
    t = re.sub(r"job\s*reference\s*\d+", " ", t)
    t = re.sub(r"\bref\S*[:\s]*\d+\b", " ", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_categorical_for_matching(series: pd.Series) -> pd.Series:
    """
    Vectorized (per-column .map, NOT .astype(str)/.str.cat()) categorical
    normaliser. Guaranteed to never produce NA in its output regardless
    of the source column's dtype.

    Explicitly forces plain string dtype via `.astype(str)` after `.map()`
    -- `.map()` on a Categorical-dtype Series (e.g. `department`, cast to
    "category" by data_cleaning.py's _finalise_dtypes) PRESERVES the
    Categorical dtype in its output, which breaks Python `+` string
    concatenation with a plain string. Safe to call after
    _normalize_categorical_scalar has already turned any NA into "".
    """
    return series.map(_normalize_categorical_scalar).astype(str)


def _make_content_key(df: pd.DataFrame) -> pd.Series:
    """
    Content-match key: dept + grade (exact) + profession + business_area
    (casefold/whitespace-normalised) + salary band + normalised
    job_description text.

    Built using per-column `.map()` to explicit scalar functions (never
    `.str.cat()`, which silently propagates NA across an ENTIRE
    concatenated key if any single input has a missing value) and plain
    Python string concatenation (`+`, which does NOT have that
    NA-propagation behaviour once every piece is already a guaranteed
    real string).
    """
    key = _normalize_categorical_for_matching(df["department"])
    key = key + "||" + _normalize_categorical_for_matching(df["job_grade"])
    for col in _DEDUP_KEY_LOOSE_COLUMNS:
        key = key + "||" + _normalize_categorical_for_matching(df[col])
    key = key + "||" + df["salary_minimum"].map(_safe_str_scalar)
    key = key + "||" + df["salary_maximum_optional"].map(_safe_str_scalar)
    key = key + "||" + df["job_description"].map(_normalize_description_scalar)
    return key


def _compute_advertising_window(
    df: pd.DataFrame,
    date_col: str,
    closing_col: str,
    imputed_flag_col: str,
) -> pd.Series:
    """
    advertising_window_days = closing_col - date_col, in days. NaN
    wherever date_col was IMPUTED.
    """
    posting = pd.to_datetime(df[date_col], errors="coerce")
    closing = pd.to_datetime(df[closing_col], errors="coerce")
    window_days = (closing - posting).dt.total_seconds() / 86400

    if imputed_flag_col in df.columns:
        window_days = window_days.where(~df[imputed_flag_col].fillna(False))

    return window_days


def _select_canonical_row(cluster: list, window: pd.Series, date_series: pd.Series) -> tuple:
    """
    Choose which row in `cluster` is canonical (kept). Prefers the row
    with the LONGEST advertising window; falls back to earliest date if
    no window is computable.
    """
    with_window = [i for i in cluster if pd.notna(window.get(i))]
    if with_window:
        keep_idx = max(with_window, key=lambda i: window.at[i])
        return keep_idx, "longest_window"

    dated = [i for i in cluster if pd.notna(date_series.get(i))]
    if dated:
        keep_idx = min(dated, key=lambda i: date_series.at[i])
        return keep_idx, "earliest_date"

    return cluster[0], "earliest_date"


def _compute_date_subgroups(
    df: pd.DataFrame,
    order,
    date_col: str,
    imputed_flag_col: str,
    gap_days: int,
) -> dict:
    """
    Shared helper (used by both assign_duplicate_groups and
    inspect_content_key_group) implementing the date-subgrouping rule:
    rows are split into a new subgroup whenever the gap between
    consecutive (non-imputed) posting dates exceeds `gap_days`. Rows with
    a missing or IMPUTED date join whichever subgroup is "current"
    without triggering a new split or updating the anchor date.

    `order` must be an index sequence already sorted by date_col
    (na_position="first"). Returns a dict {index: subgroup_label}.
    """
    has_imputed_flag = imputed_flag_col in df.columns
    date_subgroup_of = {}
    current_subgroup = 0
    prev_date = None
    for idx in order:
        d = df.at[idx, date_col]
        if pd.isna(d):
            date_subgroup_of[idx] = current_subgroup
            continue

        is_imputed = bool(df.at[idx, imputed_flag_col]) if has_imputed_flag else False
        if is_imputed:
            date_subgroup_of[idx] = current_subgroup
            continue

        if prev_date is not None and (d - prev_date).days > gap_days:
            current_subgroup += 1
        date_subgroup_of[idx] = current_subgroup
        prev_date = d

    return date_subgroup_of


def assign_duplicate_groups(
    df: pd.DataFrame,
    date_col: str = "most_recent_posting_live",
    closing_col: str = "closing_date",
    title_col: str = "vacancy_title",
    gap_days: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Flag (not drop) duplicate postings using an exact content key, a
    date-proximity check, a title-similarity check, a same-timestamp
    bulk-batch protection, and an advertising-window preference for
    canonical-row selection.

    IMPORTANT: requires `{date_col}_imputed` to already exist on `df`
    (i.e. ensure_most_recent_posting_live must have been called on this
    exact dataframe object first). If missing, ALL rows are treated as
    non-imputed (a warning is printed) -- this will not raise an error.

    Grouping pipeline, in order:
      1. Exact content key (see _make_content_key): department + job_grade
         + profession + business_area + salary_minimum +
         salary_maximum_optional + normalised job_description text.
      2. Within each content-key group, split into date-consistent
         subgroups wherever the gap between consecutive posting dates
         exceeds `gap_days` (default 30). Rows whose date is IMPUTED do
         NOT trigger a new split and do NOT update the "previous date"
         anchor.
      3. Within each date-consistent subgroup, further split by TITLE
         similarity -- catches bulk recruitment campaigns advertising many
         genuinely DIFFERENT vacancies using identical templated text.

    Canonical-row selection, for each final cluster:
      - Same-timestamp batch (every member shares the exact same posting
        timestamp AND number_of_jobs_available varies across members):
        NO row is dropped.
      - Otherwise: row with the LONGEST advertising_window_days is kept
        (falls back to earliest-dated row if no member has a computable
        window).

    duplicate_group_id is only assigned (non -1) to clusters with MORE
    THAN ONE member -- singleton clusters (including rows split off from
    a larger content-key group by date-gap/title-similarity) correctly
    keep duplicate_group_id == -1, consistent with is_duplicate == False.

    Adds: duplicate_group_id, is_duplicate, is_duplicate_to_drop,
    duplicate_canonical_method ("protected_batch"/"longest_window"/
    "earliest_date"), duplicate_date_gap_unresolved, duplicate_njobs_differs,
    duplicate_same_timestamp_batch, date_subgroup_uses_imputed_date,
    advertising_window_days, short_advert_window.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[closing_col] = pd.to_datetime(df[closing_col], errors="coerce")

    imputed_flag_col = f"{date_col}_imputed"
    has_imputed_flag = imputed_flag_col in df.columns
    window = _compute_advertising_window(df, date_col, closing_col, imputed_flag_col)
    df["advertising_window_days"] = window
    df["short_advert_window"] = window.notna() & (window >= 0) & (window < 7.0)

    content_key = _make_content_key(df)
    titles = df[title_col].fillna("")

    duplicate_group_id = pd.Series(-1, index=df.index, dtype="int64")
    is_duplicate = pd.Series(False, index=df.index)
    is_duplicate_to_drop = pd.Series(False, index=df.index)
    duplicate_canonical_method = pd.Series("", index=df.index, dtype="object")
    date_gap_unresolved = pd.Series(False, index=df.index)
    njobs_differs = pd.Series(False, index=df.index)
    same_timestamp_batch = pd.Series(False, index=df.index)
    date_subgroup_uses_imputed_date = pd.Series(False, index=df.index)

    has_njobs = "number_of_jobs_available" in df.columns
    group_id_counter = 0

    for key, idxs in content_key.groupby(content_key).groups.items():
        idxs = list(idxs)
        if len(idxs) == 1:
            continue

        sub_dates = df.loc[idxs, date_col]
        order = sub_dates.sort_values(na_position="first").index

        date_subgroup_of = {}
        current_subgroup = 0
        prev_date = None
        for idx in order:
            d = df.at[idx, date_col]
            if pd.isna(d):
                date_subgroup_of[idx] = current_subgroup
                date_gap_unresolved[idx] = True
                continue

            is_imputed = bool(df.at[idx, imputed_flag_col]) if has_imputed_flag else False
            if is_imputed:
                date_subgroup_of[idx] = current_subgroup
                date_subgroup_uses_imputed_date[idx] = True
                continue

            if prev_date is not None and (d - prev_date).days > gap_days:
                current_subgroup += 1
            date_subgroup_of[idx] = current_subgroup
            prev_date = d

        n_date_subgroups = max(date_subgroup_of.values()) + 1
        for sg in range(n_date_subgroups):
            sg_idxs = [i for i in order if date_subgroup_of[i] == sg]

            title_clusters = (
                _cluster_by_title_similarity(sg_idxs, titles) if len(sg_idxs) > 1 else [sg_idxs]
            )

            for cluster in title_clusters:
                # Only assign a real (non -1) duplicate_group_id when the
                # cluster has more than one member -- singleton clusters
                # (including ones split off from a larger content-key
                # group by date-gap/title-similarity) correctly keep
                # duplicate_group_id == -1, consistent with
                # is_duplicate == False.
                if len(cluster) > 1:
                    for i in cluster:
                        duplicate_group_id[i] = group_id_counter
                    is_duplicate.loc[cluster] = True

                    n_unique_njobs = (
                        df.loc[cluster, "number_of_jobs_available"].nunique(dropna=False)
                        if has_njobs
                        else 1
                    )
                    if has_njobs and n_unique_njobs > 1:
                        njobs_differs.loc[cluster] = True

                    is_same_timestamp = df.loc[cluster, date_col].nunique(dropna=False) == 1
                    protect_as_batch = is_same_timestamp and has_njobs and n_unique_njobs > 1

                    if protect_as_batch:
                        same_timestamp_batch.loc[cluster] = True
                        duplicate_canonical_method.loc[cluster] = "protected_batch"
                    else:
                        keep_idx, method = _select_canonical_row(cluster, window, df[date_col])
                        for i in cluster:
                            duplicate_canonical_method[i] = method
                            if i != keep_idx:
                                is_duplicate_to_drop[i] = True
                group_id_counter += 1

    df["duplicate_group_id"] = duplicate_group_id
    df["is_duplicate"] = is_duplicate
    df["is_duplicate_to_drop"] = is_duplicate_to_drop
    df["duplicate_canonical_method"] = duplicate_canonical_method
    df["duplicate_date_gap_unresolved"] = date_gap_unresolved
    df["duplicate_njobs_differs"] = njobs_differs
    df["duplicate_same_timestamp_batch"] = same_timestamp_batch
    df["date_subgroup_uses_imputed_date"] = date_subgroup_uses_imputed_date

    if verbose:
        n_groups = df.loc[df["is_duplicate"], "duplicate_group_id"].nunique()
        n_drop = int(df["is_duplicate_to_drop"].sum())
        n_unresolved = int(df["duplicate_date_gap_unresolved"].sum())
        n_njobs_differs = int(df["duplicate_njobs_differs"].sum())
        n_same_ts = int(df["duplicate_same_timestamp_batch"].sum())
        n_longest_window = int((df["duplicate_canonical_method"] == "longest_window").sum())
        n_earliest_date = int((df["duplicate_canonical_method"] == "earliest_date").sum())
        n_imputed_subgroup = int(df["date_subgroup_uses_imputed_date"].sum())
        if not has_imputed_flag:
            print(
                f"WARNING: '{imputed_flag_col}' column not found -- imputed-date protection "
                f"is DISABLED for this call. Run ensure_most_recent_posting_live on this exact "
                f"dataframe first if you expect it to be active."
            )
        print(
            f"Duplicates: {n_groups:,} duplicate group(s) found; "
            f"{n_drop:,} rows flagged is_duplicate_to_drop; "
            f"{n_unresolved:,} rows had an unresolved (missing) date for gap-checking; "
            f"{n_njobs_differs:,} rows are in a duplicate group with inconsistent number_of_jobs_available; "
            f"{n_same_ts:,} rows are in a same-timestamp batch (NOT flagged to drop)"
        )
        print(
            f"Canonical selection: {n_longest_window:,} rows in clusters resolved by longest-window "
            f"preference; {n_earliest_date:,} rows in clusters resolved by earliest-date fallback "
            f"(no computable window); {n_imputed_subgroup:,} rows joined their subgroup via an "
            f"imputed (unreliable) date rather than triggering a new re-advertisement split"
        )

    return df


# ===========================================================================
# 4. Diagnostic / export helpers
# ===========================================================================

_DEFAULT_REVIEW_COLUMNS = [
    "vacancy_id",
    "vacancy_title",
    "department",
    "cs_org",
    "job_grade",
    "cs_profession",
    "salary_minimum",
    "salary_maximum_optional",
    "most_recent_posting_live",
    "most_recent_posting_live_imputed",
    "closing_date",
    "advertising_window_days",
    "short_advert_window",
    "number_of_jobs_available",
    "vacancy_city",
    "vacancy_region",
    "business_area",
    "duplicate_group_id",
    "is_duplicate",
    "is_duplicate_to_drop",
    "duplicate_canonical_method",
    "duplicate_date_gap_unresolved",
    "duplicate_njobs_differs",
    "duplicate_same_timestamp_batch",
    "date_subgroup_uses_imputed_date",
    "is_test_data",
    "title_marked_copy",
    "title_marked_discard",
    "title_marked_error",
    "job_description",
]


def export_duplicate_review_subset(
    df: pd.DataFrame,
    path: str = "temp.csv",
    description_chars: int = 200,
    n_singleton_sample: int = 25,
    n_unresolved_sample: int = 50,
    columns: list = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build and save a review subset of `df` covering duplicate groups, flagged titles, etc."""
    df = df.copy()
    cols = columns if columns is not None else _DEFAULT_REVIEW_COLUMNS
    cols = [c for c in cols if c in df.columns]

    reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

    in_dup_group = df["duplicate_group_id"] != -1
    reasons.loc[in_dup_group] += "duplicate_group;"

    is_flagged_title = pd.Series(False, index=df.index)
    for flag_col in (
        "is_test_data",
        "title_marked_copy",
        "title_marked_discard",
        "title_marked_error",
    ):
        if flag_col in df.columns:
            is_flagged_title |= df[flag_col].fillna(False)
    reasons.loc[is_flagged_title] = reasons.loc[is_flagged_title] + "test_or_copy_title;"

    unresolved_mask = pd.Series(False, index=df.index)
    if "duplicate_date_gap_unresolved" in df.columns:
        unresolved_pool = df.index[df["duplicate_date_gap_unresolved"] & (reasons == "")]
        if len(unresolved_pool) > 0:
            n = min(n_unresolved_sample, len(unresolved_pool))
            sampled = (
                pd.Index(unresolved_pool).to_series().sample(n=n, random_state=random_state).index
            )
            unresolved_mask.loc[sampled] = True
    reasons.loc[unresolved_mask] += "unresolved_date_sample;"

    already_included = reasons != ""
    singleton_pool = df.index[~already_included]
    n_singletons = min(n_singleton_sample, len(singleton_pool))
    singleton_sample_idx = (
        pd.Index(singleton_pool).to_series().sample(n=n_singletons, random_state=random_state).index
        if n_singletons > 0
        else pd.Index([])
    )
    reasons.loc[singleton_sample_idx] += "negative_control_sample;"

    keep_mask = reasons != ""
    out = df.loc[keep_mask, cols].copy()
    out["subset_reason"] = reasons.loc[keep_mask].str.rstrip(";")

    if "job_description" in out.columns:
        out["job_description"] = (
            out["job_description"].astype("string").str.slice(0, description_chars)
        )

    sort_cols = [c for c in ("duplicate_group_id", "most_recent_posting_live") if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, na_position="last")

    out.to_csv(path, index=False)

    if verbose:
        print(f"Exported {len(out):,} rows to {path}")
        print(f"  - in a duplicate group:        {int(in_dup_group.sum()):,}")
        print(f"  - test/copy/discard/error title: {int(is_flagged_title.sum()):,}")
        print(f"  - unresolved-date sample:       {int(unresolved_mask.sum()):,}")
        print(f"  - negative control sample:      {len(singleton_sample_idx):,}")

    return out


def export_njobs_ambiguous_review(
    df: pd.DataFrame,
    path: str = "temp_njobs_review.csv",
    description_chars: int = 150,
    verbose: bool = True,
) -> pd.DataFrame:
    """Export rows in duplicate groups where number_of_jobs_available varies and it's not a protected batch."""
    df = df.copy()

    mask = df["duplicate_njobs_differs"] & ~df["duplicate_same_timestamp_batch"]
    ambiguous_group_ids = df.loc[mask, "duplicate_group_id"].unique()
    out = df[df["duplicate_group_id"].isin(ambiguous_group_ids)].copy()

    group_stats = out.groupby("duplicate_group_id")["number_of_jobs_available"].agg(
        group_size="count",
        group_njobs_sum="sum",
        group_njobs_min="min",
        group_njobs_max="max",
    )
    out = out.merge(group_stats, on="duplicate_group_id", how="left")

    cols = [
        "duplicate_group_id",
        "group_size",
        "group_njobs_min",
        "group_njobs_max",
        "group_njobs_sum",
        "vacancy_id",
        "vacancy_title",
        "department",
        "job_grade",
        "salary_minimum",
        "salary_maximum_optional",
        "most_recent_posting_live",
        "closing_date",
        "advertising_window_days",
        "number_of_jobs_available",
        "vacancy_city",
        "vacancy_region",
        "business_area",
        "is_duplicate_to_drop",
        "duplicate_canonical_method",
        "job_description",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    if "job_description" in out.columns:
        out["job_description"] = (
            out["job_description"].astype("string").str.slice(0, description_chars)
        )

    out = out.sort_values(
        by=["group_size", "duplicate_group_id", "most_recent_posting_live"],
        ascending=[False, True, True],
    )
    out.to_csv(path, index=False)

    if verbose:
        print(
            f"Exported {len(out):,} rows across {len(ambiguous_group_ids):,} ambiguous "
            f"njobs-differs groups to {path}"
        )
        print(
            f"  Group sizes range from {group_stats['group_size'].min()} to "
            f"{group_stats['group_size'].max()} rows"
        )

    return out


def find_content_key_mismatches(
    df: pd.DataFrame,
    title_col: str = "vacancy_title",
    max_groups: int = None,
    max_pairs_per_group: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    DIAGNOSTIC ONLY. Groups rows by a LOOSER key than assign_duplicate_groups'
    exact content key, and for groups the exact key does NOT tie together,
    reports a text-similarity ratio and first divergence point.

    Default max_groups=None checks every candidate (no silent cap).
    Candidates are prioritised by total number_of_jobs_available at stake.
    """
    df = df.copy()
    loose_title = df[title_col].fillna("").apply(_normalize_title_for_matching)
    loose_key = (
        loose_title
        + "||"
        + _normalize_categorical_for_matching(df["department"])
        + "||"
        + _normalize_categorical_for_matching(df["job_grade"])
        + "||"
        + df["salary_minimum"].map(_safe_str_scalar)
        + "||"
        + df["salary_maximum_optional"].map(_safe_str_scalar)
    )

    candidate_groups = []
    for key, idxs in loose_key.groupby(loose_key).groups.items():
        if len(idxs) < 2:
            continue
        if "duplicate_group_id" in df.columns:
            group_ids = df.loc[idxs, "duplicate_group_id"]
            already_grouped = group_ids.nunique() == 1 and group_ids.iloc[0] != -1
        else:
            already_grouped = False
        if already_grouped:
            continue
        njobs_at_stake = (
            df.loc[idxs, "number_of_jobs_available"].fillna(1).sum()
            if "number_of_jobs_available" in df.columns
            else len(idxs)
        )
        candidate_groups.append((njobs_at_stake, list(idxs)))

    candidate_groups.sort(key=lambda x: x[0], reverse=True)

    if verbose:
        total_candidates = len(candidate_groups)
        n_to_check = total_candidates if max_groups is None else min(max_groups, total_candidates)
        print(
            f"find_content_key_mismatches: {total_candidates:,} candidate mismatch group(s) found; "
            f"checking {n_to_check:,} (sorted by total number_of_jobs_available at stake, descending)"
        )
        if max_groups is not None and total_candidates > max_groups:
            print(
                f"  WARNING: {total_candidates - max_groups:,} lower-impact candidate group(s) "
                f"will NOT be checked due to max_groups={max_groups}. Pass max_groups=None (default) "
                f"to check every candidate."
            )

    results = []
    groups_checked = 0
    for njobs_at_stake, idx_list in candidate_groups:
        if max_groups is not None and groups_checked >= max_groups:
            break
        groups_checked += 1
        n_pairs = 0
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                if n_pairs >= max_pairs_per_group:
                    break
                a, b = idx_list[i], idx_list[j]
                desc_a = (
                    str(df.at[a, "job_description"])
                    if pd.notna(df.at[a, "job_description"])
                    else ""
                )
                desc_b = (
                    str(df.at[b, "job_description"])
                    if pd.notna(df.at[b, "job_description"])
                    else ""
                )

                ratio = difflib.SequenceMatcher(None, desc_a, desc_b).ratio()

                first_diff_pos = None
                for k in range(min(len(desc_a), len(desc_b))):
                    if desc_a[k] != desc_b[k]:
                        first_diff_pos = k
                        break
                if first_diff_pos is None and len(desc_a) != len(desc_b):
                    first_diff_pos = min(len(desc_a), len(desc_b))

                snippet_a = desc_a[max(0, (first_diff_pos or 0) - 20) : (first_diff_pos or 0) + 20]
                snippet_b = desc_b[max(0, (first_diff_pos or 0) - 20) : (first_diff_pos or 0) + 20]

                prof_a = df.at[a, "profession"] if "profession" in df.columns else None
                prof_b = df.at[b, "profession"] if "profession" in df.columns else None
                biz_a = df.at[a, "business_area"] if "business_area" in df.columns else None
                biz_b = df.at[b, "business_area"] if "business_area" in df.columns else None

                results.append(
                    {
                        "vacancy_id_a": df.at[a, "vacancy_id"],
                        "vacancy_id_b": df.at[b, "vacancy_id"],
                        "vacancy_title": df.at[a, title_col],
                        "description_similarity_ratio": round(ratio, 5),
                        "description_len_a": len(desc_a),
                        "description_len_b": len(desc_b),
                        "first_diff_char_position": first_diff_pos,
                        "context_around_diff_a": snippet_a,
                        "context_around_diff_b": snippet_b,
                        "profession_a": prof_a,
                        "profession_b": prof_b,
                        "business_area_a": biz_a,
                        "business_area_b": biz_b,
                        "profession_or_business_area_differs": (prof_a != prof_b)
                        or (biz_a != biz_b),
                    }
                )
                n_pairs += 1

    out = pd.DataFrame(results)
    if len(out):
        out = out.sort_values("description_similarity_ratio", ascending=False)
    return out


def explain_content_key_mismatch(
    df: pd.DataFrame,
    vacancy_id_a,
    vacancy_id_b,
    vacancy_id_col: str = "vacancy_id",
) -> None:
    """
    TARGETED DIAGNOSTIC -- prints, field by field, exactly why two specific
    rows did or did not receive the same assign_duplicate_groups content
    key. Uses the same scalar-safe normalisation as _make_content_key.
    """
    row_a = df.loc[df[vacancy_id_col].astype(str) == str(vacancy_id_a)]
    row_b = df.loc[df[vacancy_id_col].astype(str) == str(vacancy_id_b)]
    if len(row_a) == 0:
        print(f"No row found with {vacancy_id_col} == {vacancy_id_a!r}")
        return
    if len(row_b) == 0:
        print(f"No row found with {vacancy_id_col} == {vacancy_id_b!r}")
        return
    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    print(f"Comparing {vacancy_id_col}={vacancy_id_a!r} vs {vacancy_id_col}={vacancy_id_b!r}\n")

    all_match = True
    for col in _DEDUP_KEY_EXACT_COLUMNS + _DEDUP_KEY_LOOSE_COLUMNS:
        raw_a, raw_b = row_a.get(col), row_b.get(col)
        norm_a = _normalize_categorical_scalar(raw_a)
        norm_b = _normalize_categorical_scalar(raw_b)
        match = norm_a == norm_b
        all_match &= match
        flag = "OK" if match else "DIFFERS"
        print(f"[{flag:7}] {col}:")
        print(f"           a: raw={raw_a!r}  normalised={norm_a!r}")
        print(f"           b: raw={raw_b!r}  normalised={norm_b!r}")

    for col in ("salary_minimum", "salary_maximum_optional"):
        raw_a, raw_b = row_a.get(col), row_b.get(col)
        safe_a, safe_b = _safe_str_scalar(raw_a), _safe_str_scalar(raw_b)
        match = safe_a == safe_b
        all_match &= match
        flag = "OK" if match else "DIFFERS"
        print(
            f"[{flag:7}] {col}: a={raw_a!r} (key-value={safe_a!r})  b={raw_b!r} (key-value={safe_b!r})"
        )

    desc_a = row_a.get("job_description")
    desc_b = row_b.get("job_description")
    norm_desc_a = _normalize_description_scalar(desc_a)
    norm_desc_b = _normalize_description_scalar(desc_b)
    desc_match = norm_desc_a == norm_desc_b
    all_match &= desc_match

    raw_len_a = len(str(desc_a)) if pd.notna(desc_a) else 0
    raw_len_b = len(str(desc_b)) if pd.notna(desc_b) else 0
    print(
        f"\n[{'OK' if desc_match else 'DIFFERS':7}] job_description (after matching-normalisation):"
    )
    print(f"           raw length: a={raw_len_a:,} chars, b={raw_len_b:,} chars")
    print(
        f"           normalised length: a={len(norm_desc_a):,} chars, b={len(norm_desc_b):,} chars"
    )

    if not desc_match:
        first_diff = None
        for k in range(min(len(norm_desc_a), len(norm_desc_b))):
            if norm_desc_a[k] != norm_desc_b[k]:
                first_diff = k
                break
        if first_diff is None:
            first_diff = min(len(norm_desc_a), len(norm_desc_b))
        start = max(0, first_diff - 40)
        print(f"           first normalised divergence at character {first_diff}:")
        print(f"           a: ...{norm_desc_a[start : first_diff + 40]!r}...")
        print(f"           b: ...{norm_desc_b[start : first_diff + 40]!r}...")

    print(f"\n{'=' * 70}")
    if all_match:
        print("RESULT: every field matches after normalisation -- assign_duplicate_groups")
        print("SHOULD place these two rows in the same content-key group. If they still")
        print("show different duplicate_group_id values, use inspect_content_key_group()")
        print("below to see the full set of rows sharing this content key and how the")
        print("date-gap/title-similarity stages split them.")
    else:
        print("RESULT: at least one field genuinely differs after normalisation (see")
        print("'DIFFERS' lines above) -- this is why the two rows are NOT in the same")
        print("content-key group.")


def inspect_content_key_group(
    df: pd.DataFrame,
    vacancy_id,
    vacancy_id_col: str = "vacancy_id",
    title_col: str = "vacancy_title",
    date_col: str = "most_recent_posting_live",
    gap_days: int = 30,
) -> pd.DataFrame:
    """
    Finds every row in `df` sharing the EXACT SAME content key as
    `vacancy_id` (recomputed live via the scalar-safe _make_content_key --
    the identical logic assign_duplicate_groups uses internally), and
    returns them sorted by date with their title, imputed flag, and
    current duplicate_group_id/is_duplicate/is_duplicate_to_drop values.

    Defensive against real issues found during debugging:
      - A non-unique index on `df` is detected and reset internally.
      - Missing required columns are reported explicitly.

    Replicates BOTH stage 2 (date-subgroup, via the shared
    _compute_date_subgroups helper) and stage 3 (title-similarity
    clustering) on just the rows sharing this content key, and compares
    the resulting EXPECTED partition against the ACTUAL duplicate_group_id
    partition found in `df`. Only genuine inconsistencies (the expected
    and actual partitions disagree) are reported as a warning; otherwise,
    the specific reason for each split (date gap of N days, or a
    genuinely different title) is printed explicitly.
    """
    required_cols = [
        "department",
        "job_grade",
        "profession",
        "business_area",
        "salary_minimum",
        "salary_maximum_optional",
        "job_description",
        vacancy_id_col,
        title_col,
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: df is missing required column(s): {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    was_unique = df.index.is_unique
    df = df.reset_index(drop=True)
    if not was_unique:
        print(
            f"NOTE: input dataframe had a NON-UNIQUE index -- this has been reset internally. "
            f"This is a common symptom of merge()/concat() without reset_index() upstream, "
            f"and could explain unexpected behaviour elsewhere in your pipeline too.\n"
        )

    target_mask = (df[vacancy_id_col].astype(str) == str(vacancy_id)).to_numpy()
    n_target = int(target_mask.sum())
    print(f"Rows matching {vacancy_id_col}={vacancy_id!r}: {n_target}")
    if n_target == 0:
        print("No row found with that vacancy_id.")
        return pd.DataFrame()
    if n_target > 1:
        print(
            f"WARNING: {n_target} rows share this vacancy_id -- expected exactly 1. Using the first."
        )

    target_pos = target_mask.nonzero()[0][0]
    target_row = df.iloc[target_pos]
    target_title = target_row[title_col]

    for col in required_cols:
        if col in (vacancy_id_col, title_col):
            continue
        if pd.isna(target_row[col]):
            print(
                f"  NOTE: column '{col}' is missing/NA for this row (dtype={df[col].dtype}) "
                f"-- treated as blank/'<MISSING>' for matching purposes, not excluded."
            )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    imputed_flag_col = f"{date_col}_imputed"
    has_imputed_flag = imputed_flag_col in df.columns

    keys = _make_content_key(df)
    target_key = keys.iloc[target_pos]
    print(f"\nComputed content key for this row:\n  {target_key!r}\n")

    matching_mask = (keys == target_key).to_numpy()
    matching_idx = df.index[matching_mask].tolist()
    matching = df.loc[matching_idx].copy()

    matching["title_matches_target"] = matching[title_col].apply(
        lambda t: _titles_indicate_same_role(target_title, t)
    )

    order = df.loc[matching_idx, date_col].sort_values(na_position="first").index
    date_subgroup_of = _compute_date_subgroups(df, order, date_col, imputed_flag_col, gap_days)
    titles_series = df[title_col].fillna("")

    expected_cluster_of = {}
    n_date_subgroups = max(date_subgroup_of.values()) + 1 if date_subgroup_of else 0
    for sg in range(n_date_subgroups):
        sg_idxs = [i for i in order if date_subgroup_of[i] == sg]
        title_clusters = (
            _cluster_by_title_similarity(sg_idxs, titles_series) if len(sg_idxs) > 1 else [sg_idxs]
        )
        for cluster_num, cluster in enumerate(title_clusters):
            for i in cluster:
                expected_cluster_of[i] = (sg, cluster_num)

    cols = [
        vacancy_id_col,
        title_col,
        "title_matches_target",
        date_col,
        f"{date_col}_imputed",
        "duplicate_group_id",
        "is_duplicate",
        "is_duplicate_to_drop",
        "duplicate_canonical_method",
        "date_subgroup_uses_imputed_date",
    ]
    cols = [c for c in cols if c in matching.columns]
    if date_col in matching.columns:
        matching = matching[cols].sort_values(date_col, na_position="first")
    else:
        matching = matching[cols]

    print(f"Found {len(matching)} row(s) sharing this content key:\n")
    print(matching.to_string())

    if "duplicate_group_id" not in matching.columns:
        return matching

    target_idx_in_df = target_pos
    target_expected_cluster = expected_cluster_of.get(target_idx_in_df)
    target_actual_group = df.at[target_idx_in_df, "duplicate_group_id"]

    n_distinct_groups = matching["duplicate_group_id"].nunique()
    print(f"\nDistinct duplicate_group_id values among these rows: {n_distinct_groups}")

    genuine_anomalies = []
    explained_splits = []
    for i in matching_idx:
        if i == target_idx_in_df:
            continue
        # NOTE: duplicate_group_id == -1 means "not a real duplicate group"
        # (a singleton) for EACH row independently -- two different
        # singleton rows both showing -1 must NOT be treated as "the same
        # group". Guard explicitly rather than relying on -1 == -1.
        i_group = df.at[i, "duplicate_group_id"]
        actual_same_group = (
            target_actual_group != -1 and i_group != -1 and i_group == target_actual_group
        )
        expected_same_cluster = expected_cluster_of.get(i) == target_expected_cluster

        if expected_same_cluster != actual_same_group:
            genuine_anomalies.append(i)
        elif not expected_same_cluster:
            other_sg = expected_cluster_of.get(i, (None, None))[0]
            target_sg = target_expected_cluster[0] if target_expected_cluster else None
            title_ok = bool(df.at[i, title_col]) and _titles_indicate_same_role(
                target_title, df.at[i, title_col]
            )
            if other_sg != target_sg:
                reason = "date gap exceeds gap_days threshold (different re-advertisement cycle)"
            elif not title_ok:
                reason = "title genuinely differs (different underlying role)"
            else:
                reason = "split for an unspecified combination of the above"
            explained_splits.append((df.at[i, vacancy_id_col], reason))

    if explained_splits:
        print(
            f"\n{len(explained_splits)} row(s) are in a different duplicate_group_id than the "
            f"target for an EXPECTED, EXPLAINED reason (not a bug):"
        )
        for vid, reason in explained_splits:
            print(f"  - {vacancy_id_col}={vid!r}: {reason}")

    if genuine_anomalies:
        print(
            f"\nWARNING: {len(genuine_anomalies)} row(s) do NOT match the expected grouping "
            f"(recomputing stages 2+3 fresh disagrees with what's stored in df's "
            f"duplicate_group_id column). This is NOT expected -- `df`'s duplicate_group_id "
            f"likely came from a stale/different run of assign_duplicate_groups. Try "
            f"restarting your kernel and re-running assign_duplicate_groups fresh on this "
            f"exact dataframe, then check again."
        )
        for i in genuine_anomalies:
            print(f"  - {vacancy_id_col}={df.at[i, vacancy_id_col]!r}")
    elif not explained_splits:
        print(
            "\nNo unexplained grouping anomalies found -- all rows are grouped consistently "
            "with what stages 2 (date-gap) and 3 (title-similarity) would predict."
        )

    return matching


# ===========================================================================
# 5. Final filtering
# ===========================================================================


def apply_final_filters(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_discarded_titles: bool = True,
    drop_repeated_test_titles=True,
    drop_error_titles: bool = True,
    drop_dummy_titles: bool = True,
    drop_test_titles: bool = True,
    fillna_jobs_available: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Applies the data-quality flags computed above to produce the final,
    deduplicated dataset ready for downstream raking/monetisation.

    Drops (by default): is_duplicate_to_drop, title_marked_discard,
    title_marked_error.

    Does NOT drop (even by default): is_test_data, title_marked_copy
    alone, duplicate_njobs_differs, duplicate_date_gap_unresolved.

    Also fills any remaining NaN number_of_jobs_available with 1 at this
    final stage, so the column is safe to use as a headcount MULTIPLIER
    (not a row-count) in downstream aggregation.
    """
    df = df.copy()
    n_before = len(df)

    drop_mask = pd.Series(False, index=df.index)
    if drop_duplicates and "is_duplicate_to_drop" in df.columns:
        drop_mask |= df["is_duplicate_to_drop"].fillna(False)
    if drop_discarded_titles and "title_marked_discard" in df.columns:
        drop_mask |= df["title_marked_discard"].fillna(False)
    if drop_error_titles and "title_marked_error" in df.columns:
        drop_mask |= df["title_marked_error"].fillna(False)
    if drop_repeated_test_titles and "title_marked_repeated_test" in df.columns:
        drop_mask |= df["title_marked_repeated_test"].fillna(False)
    if drop_dummy_titles and "title_marked_dummy" in df.columns:
        drop_mask |= df["title_marked_dummy"].fillna(False)
    if drop_test_titles and "is_test_data" in df.columns:
        drop_mask |= df["is_test_data"].fillna(False)

    out = df.loc[~drop_mask].copy()

    if fillna_jobs_available and "number_of_jobs_available" in out.columns:
        n_filled = int(out["number_of_jobs_available"].isna().sum())
        out["number_of_jobs_available"] = out["number_of_jobs_available"].fillna(1)
    else:
        n_filled = 0

    if verbose:
        n_dup = (
            int((drop_mask & df.get("is_duplicate_to_drop", False)).sum())
            if "is_duplicate_to_drop" in df.columns
            else 0
        )
        n_discard = (
            int((drop_mask & df.get("title_marked_discard", False)).sum())
            if "title_marked_discard" in df.columns
            else 0
        )
        n_error = (
            int((drop_mask & df.get("title_marked_error", False)).sum())
            if "title_marked_error" in df.columns
            else 0
        )
        n_repeated_test = (
            int((drop_mask & df.get("title_marked_repeated_test", False)).sum())
            if "title_marked_repeated_test" in df.columns
            else 0
        )
        n_dummy = (
            int((drop_mask & df.get("title_marked_dummy", False)).sum())
            if "title_marked_dummy" in df.columns
            else 0
        )
        print(
            f"apply_final_filters: {n_before:,} rows -> {len(out):,} rows ({int(drop_mask.sum()):,} dropped)"
        )
        print(f"  - dropped as is_duplicate_to_drop:  {n_dup:,}")
        print(f"  - dropped as title_marked_discard:  {n_discard:,}")
        print(f"  - dropped as title_marked_error:    {n_error:,}")
        print(f" - dropped as title_marked_repeated_test: {n_repeated_test:,}")
        print(f" - dropped as title_marked_dummy: {n_dummy:,}")
        if fillna_jobs_available:
            print(f"  - number_of_jobs_available: {n_filled:,} remaining NaNs filled with 1")
        if "number_of_jobs_available" in out.columns:
            total_roles = out["number_of_jobs_available"].sum()
            print(
                f"  Final row count: {len(out):,}  |  Total implied roles "
                f"(sum of number_of_jobs_available): {total_roles:,.0f}"
            )

    return out


# ===========================================================================
# 6. Diagnostics
# ===========================================================================


def export_final_summary_stats(
    df: pd.DataFrame,
    path: str = "spot_check_summary.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    High-level sanity-check summary of `final_df`: row count, total implied
    roles, breakdowns by department/profession/grade, salary range, and
    counts of every remaining data-quality flag (is_duplicate,
    duplicate_same_timestamp_batch, duplicate_njobs_differs, is_test_data,
    title_marked_copy) so you can see at a glance how much of each category
    survived into the final dataset.

    Returns a small "metric, value" dataframe (also written to `path`) --
    NOT the per-department/profession breakdowns, which are printed
    directly to the console since they're usually short enough to read
    without a separate file.
    """
    n_rows = len(df)
    total_roles = (
        df["number_of_jobs_available"].sum() if "number_of_jobs_available" in df.columns else None
    )

    metrics = {
        "total_rows": n_rows,
        "total_implied_roles (sum number_of_jobs_available)": total_roles,
        "rows_with_number_of_jobs_available_eq_1": int((df["number_of_jobs_available"] == 1).sum())
        if "number_of_jobs_available" in df.columns
        else None,
        "rows_still_flagged_is_duplicate": int(df["is_duplicate"].sum())
        if "is_duplicate" in df.columns
        else None,
        "rows_in_protected_same_timestamp_batch": int(df["duplicate_same_timestamp_batch"].sum())
        if "duplicate_same_timestamp_batch" in df.columns
        else None,
        "rows_in_group_with_njobs_differs": int(df["duplicate_njobs_differs"].sum())
        if "duplicate_njobs_differs" in df.columns
        else None,
        "rows_still_flagged_is_test_data": int(df["is_test_data"].sum())
        if "is_test_data" in df.columns
        else None,
        "rows_still_flagged_title_marked_copy": int(df["title_marked_copy"].sum())
        if "title_marked_copy" in df.columns
        else None,
        "rows_with_missing_salary_minimum": int(df["salary_minimum"].isna().sum())
        if "salary_minimum" in df.columns
        else None,
        "rows_with_missing_salary_maximum_optional": int(df["salary_maximum_optional"].isna().sum())
        if "salary_maximum_optional" in df.columns
        else None,
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metrics_df.to_csv(path, index=False)

    if verbose:
        print(f"=== Final dataset summary (n={n_rows:,} rows) ===\n")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:,.0f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

        if "department" in df.columns and "number_of_jobs_available" in df.columns:
            print("\n--- Top 15 departments by total implied roles ---")
            dept_summary = (
                df.groupby("department")["number_of_jobs_available"]
                .agg(rows="count", total_roles="sum")
                .sort_values("total_roles", ascending=False)
                .head(15)
            )
            print(dept_summary.to_string())

        if "cs_grade" in df.columns and "number_of_jobs_available" in df.columns:
            print("\n--- Total implied roles by grade ---")
            grade_summary = (
                df.groupby("cs_grade", observed=True)["number_of_jobs_available"]
                .agg(rows="count", total_roles="sum")
                .sort_values("total_roles", ascending=False)
            )
            print(grade_summary.to_string())

        if "salary_minimum" in df.columns:
            print(f"\n--- salary_minimum range ---")
            print(df["salary_minimum"].describe().to_string())

        print(f"\nSummary metrics written to {path}")

    return metrics_df


def export_remaining_duplicate_check(
    df: pd.DataFrame,
    path: str = "spot_check_remaining_dupes.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Checks whether any REAL duplicate group (duplicate_group_id != -1,
    excluding protected same-timestamp batches) still has MORE THAN ONE
    row surviving in the final (post-drop) dataset -- which would mean
    the drop logic failed to fully resolve that group and left genuine
    duplicates in place.

    FIXED (v2): an earlier version of this function recomputed the exact
    content key from scratch and applied an ALL-OR-NOTHING test across
    the entire content-key "bucket" (i.e. every row sharing department +
    job_grade + profession + business_area + salary + normalised
    description) -- flagging the WHOLE bucket unless every single row in
    it was either a protected batch or a singleton. This produced a large
    false-positive blowup in practice: buckets containing e.g. 48
    genuinely different location postings (correctly resolved to
    singletons by title-similarity) PLUS one unrelated leftover canonical
    survivor from a completely separate, correctly-resolved duplicate
    pair -- that one non-singleton row caused the entire 48+1-row bucket
    to be flagged as "unexpected", even though 48 of those rows were
    never duplicates of each other at all (they just happen to share
    templated boilerplate text with a THIRD, unrelated pair elsewhere in
    the same bucket).

    This version instead groups directly by `duplicate_group_id` (which
    already encodes the fully-resolved stage-1/2/3 decision from
    assign_duplicate_groups -- content key + date-gap + title-similarity)
    and only flags a group if 2+ of its members BOTH survived into
    final_df. That is the only combination that would actually indicate
    a drop-logic failure; it does not require recomputing the content key
    at all, and does not suffer from the all-or-nothing bucket problem.
    """
    df = df.reset_index(drop=True).copy()

    real_groups = df[
        (df["duplicate_group_id"] != -1) & (~df["duplicate_same_timestamp_batch"].fillna(False))
    ]
    group_sizes = real_groups.groupby("duplicate_group_id").size()
    bad_group_ids = group_sizes[group_sizes > 1].index

    out = real_groups[real_groups["duplicate_group_id"].isin(bad_group_ids)].copy()

    if len(out):
        cols_to_show = [
            c
            for c in [
                "vacancy_id",
                "vacancy_title",
                "department",
                "job_grade",
                "profession",
                "business_area",
                "salary_minimum",
                "salary_maximum_optional",
                "number_of_jobs_available",
                "duplicate_group_id",
                "is_duplicate",
                "duplicate_same_timestamp_batch",
            ]
            if c in out.columns
        ]
        out = out[cols_to_show]
    out.to_csv(path, index=False)

    if verbose:
        print(f"Real (non-batch) duplicate groups in final_df: {len(group_sizes):,}")
        print(f"  - correctly resolved to exactly 1 surviving row: {(group_sizes == 1).sum():,}")
        print(
            f"  - UNEXPECTED (>1 row survived, written to {path}): {len(bad_group_ids):,} groups, "
            f"{len(out):,} rows"
        )
        if len(out) == 0:
            print("  No unexpected remaining duplicates found -- final_df looks clean.")

    return out


def export_manual_review_sample(
    df: pd.DataFrame,
    path: str = "spot_check_manual_review.csv",
    description_chars: int = 200,
    n_sample: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Exports every row still flagged `is_test_data` (ambiguous test/
    placeholder titles) or `title_marked_copy` (GRID's own "(Copy)" marker,
    with no matching duplicate sibling -- otherwise it would have already
    been dropped by is_duplicate_to_drop) -- capped at `n_sample` rows if
    there are more, since these were deliberately left for human judgement
    rather than auto-dropped by apply_final_filters.

    A `flag_reason` column records which flag(s) caused inclusion.
    """
    df = df.copy()
    reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "is_test_data" in df.columns:
        reasons.loc[df["is_test_data"].fillna(False)] += "is_test_data;"
    if "title_marked_copy" in df.columns:
        reasons.loc[df["title_marked_copy"].fillna(False)] += "title_marked_copy;"

    keep_mask = reasons != ""
    out = df.loc[keep_mask].copy()
    out["flag_reason"] = reasons.loc[keep_mask].str.rstrip(";")

    if len(out) > n_sample:
        out = out.sample(n=n_sample, random_state=random_state)

    cols = [
        c
        for c in [
            "vacancy_id",
            "vacancy_title",
            "department",
            "job_grade",
            "profession",
            "business_area",
            "number_of_jobs_available",
            "flag_reason",
            "job_description",
        ]
        if c in out.columns
    ]
    out = out[cols]

    if "job_description" in out.columns:
        out["job_description"] = (
            out["job_description"].astype("string").str.slice(0, description_chars)
        )

    out.to_csv(path, index=False)

    if verbose:
        n_test = int(df["is_test_data"].sum()) if "is_test_data" in df.columns else 0
        n_copy = int(df["title_marked_copy"].sum()) if "title_marked_copy" in df.columns else 0
        print(
            f"Manual-review candidates in final_df: {n_test:,} is_test_data, {n_copy:,} title_marked_copy "
            f"({int((reasons != '').sum()):,} unique rows total)"
        )
        print(f"Exported {len(out):,} rows (sampled if above n_sample={n_sample}) to {path}")

    return out


def export_njobs_distribution_check(
    df: pd.DataFrame,
    path: str = "spot_check_njobs_distribution.csv",
    top_n: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Distribution sanity check for number_of_jobs_available in final_df:
    overall describe() stats, and the top `top_n` rows by headcount (the
    ones with the most influence on any downstream raking/monetisation
    total, so worth a final visual check that they still look sensible
    after all the dedup logic).
    """
    df = df.copy()
    njobs = df["number_of_jobs_available"]

    if verbose:
        print("=== number_of_jobs_available distribution ===")
        print(njobs.describe().to_string())
        print(f"\nValue counts for njobs <= 10:")
        print(njobs[njobs <= 10].value_counts().sort_index().to_string())
        print(f"\nRows with njobs > 100: {int((njobs > 100).sum()):,}")
        print(f"Rows with njobs > 500: {int((njobs > 500).sum()):,}")

    cols = [
        c
        for c in [
            "vacancy_id",
            "vacancy_title",
            "department",
            "job_grade",
            "number_of_jobs_available",
            "duplicate_same_timestamp_batch",
            "duplicate_njobs_differs",
            "is_duplicate",
        ]
        if c in df.columns
    ]
    out = df[cols].sort_values("number_of_jobs_available", ascending=False).head(top_n)
    out.to_csv(path, index=False)

    if verbose:
        print(f"\nTop {top_n} rows by number_of_jobs_available written to {path}")

    return out


# Diagnostic/intermediate columns produced by data_quality.py that are not
# needed downstream once you're satisfied with the spot checks above.
_DIAGNOSTIC_COLUMNS_TO_DROP = [
    "duplicate_group_id",
    "is_duplicate",
    "is_duplicate_to_drop",
    "duplicate_canonical_method",
    "duplicate_date_gap_unresolved",
    "duplicate_njobs_differs",
    "duplicate_same_timestamp_batch",
    "date_subgroup_uses_imputed_date",
    "advertising_window_days",
    "short_advert_window",
    "title_marked_copy",
    "title_marked_discard",
    "title_marked_error",
    "is_test_data",
    "title_marked_repeated_test",
    "title_marked_dummy",
]


# ===========================================================================
# 7. Outputs
# ===========================================================================


def finalize_and_save_grid(
    df: pd.DataFrame,
    path: str,
    drop_diagnostic_columns: bool = True,
    columns_to_drop: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Final clean-up step: optionally strips the diagnostic/intermediate
    columns produced during data-quality processing (duplicate_group_id,
    is_duplicate, is_duplicate_to_drop, etc. -- see
    _DIAGNOSTIC_COLUMNS_TO_DROP for the full default list), then saves the
    result to `path` (parquet if the extension is .parquet, otherwise CSV)
    ready for the next pipeline stage (raking).

    Pass `drop_diagnostic_columns=False` if you'd rather keep the full
    audit trail in the saved file (e.g. to cross-reference later); pass an
    explicit `columns_to_drop` list to override which columns are removed.

    Does NOT modify `number_of_jobs_available` further -- apply_final_filters
    already guaranteed no remaining NaNs there.
    """
    df = df.copy()
    cols_to_drop = columns_to_drop if columns_to_drop is not None else _DIAGNOSTIC_COLUMNS_TO_DROP
    if drop_diagnostic_columns:
        present = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=present)
    else:
        present = []

    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    if verbose:
        print(f"Saved {len(df):,} rows, {len(df.columns)} columns to {path}")
        if drop_diagnostic_columns:
            print(f"  Dropped {len(present)} diagnostic column(s): {present}")
        total_roles = (
            df["number_of_jobs_available"].sum()
            if "number_of_jobs_available" in df.columns
            else None
        )
        if total_roles is not None:
            print(f"  Total implied roles (sum of number_of_jobs_available): {total_roles:,.0f}")

    return df
