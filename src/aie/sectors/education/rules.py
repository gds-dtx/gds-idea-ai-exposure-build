"""Constants for the schools cleaning (thresholds, field vocabularies, salary parsing).

Copied from gds-idea-education-ai-exposure-cleaning (src/education_ai_exposure/config.py,
commit 6d49b8f), without its local paths and S3 locations.
"""

from __future__ import annotations

RAW_COLS = [
    "job_title",
    "job_advert",
    "first_published_on",
    "publish_on",
    "expires_at",
    "salary",
    "working_patterns",
    "education_phase",
    "Job_role",
    "Job_Type",
]

# Cleaning thresholds — same light-touch philosophy as the NHS repo: don't over-clean, only
# drop rows that carry no usable signal.
MIN_WORDS = 20
MIN_UNIQ_RATIO = 0.05

# There's no employer_name column here (unlike NHS Jobs), so junk/test detection runs on
# title + advert text only. Kept specific to avoid catching genuine "assessment"/"testing"
# language in SENCO or exam-related adverts.
JUNK_TEXT_PATTERNS = [
    r"smoke test",
    r"this is a test",
    r"for testing purposes",
    r"\btest vacancy\b",
    r"\bdummy (?:vacancy|advert|job)\b",
    r"\bdo not apply\b",
    r"\bplease ignore this\b",
]

# Job_Type is observed as a clean binary across the sampled shards; anything else is flagged
# (not silently coerced) in case later shards contain a third value.
VALID_JOB_TYPES = {"Teaching Role", "Non Teaching Role"}

# education_phase and Job_role both arrive as single values, comma-separated multi-values, or
# (a handful of rows) " and "-joined instead of comma-joined. Split on either.
MULTI_LABEL_SPLIT_RE = r"\s*,\s*|\s+and\s+"

VALID_PHASES = {
    "nursery",
    "primary",
    "middle_deemed_primary",
    "middle_deemed_secondary",
    "secondary",
    "sixth_form_or_college",
    "through",
    "not_applicable",
}

# Observed Job_role tokens (snake_case canonical ones + several free-text group labels that
# appear verbatim in the raw data, e.g. "Administration, HR, Data, and Finance" — note this
# one group label itself contains commas, so it is excluded from the generic multi-label
# splitter and matched whole first; see categories.py).
FREE_TEXT_ROLE_GROUPS = {
    "Administration, HR, Data, and Finance",
    "Catering, Cleaning and Site Management",
    "Other Support",
    "IT Support",
    "Patoral Health and Welfare",  # verbatim typo in the source data ("Pastoral"), kept as-is
}

# Salary free-text parsing.
SALARY_AMOUNT_RE = r"£\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)"
SALARY_BASIS_KEYWORDS = {
    "hourly": (r"\bhourly\b", r"per hour", r"\bph\b"),
    "daily": (r"\bdaily\b", r"per day"),
    "weekly": (r"\bweekly\b", r"per week"),
    "annual": (r"\bannually\b", r"\bannual\b", r"per annum", r"\bpa\b"),
}
# Non-numeric salary descriptions worth keeping as a category rather than treating as garbage.
SALARY_TEXT_CATEGORIES = {
    "competitive": r"competitive",
    "negotiable": r"negotiable",
    "unpaid_voluntary": r"\bunpaid\b|\bvoluntary\b",
}
