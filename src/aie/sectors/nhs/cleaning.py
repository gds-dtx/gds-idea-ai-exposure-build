"""NHS cleaning: NHS Jobs yearly exports -> one row per NHS-scheme advert, with a reason if it
is not used.

Ported from gds-idea-nhs-ai-exposure-cleaning (config.py, loading.py, cleaning.py and the
pooling in batch.py / aggregate_vacancies.py) so the build no longer reads that repo's output.

Per year file (2020-2026; 2019 is empty):
  0. keep only the NHS pay schemes (Agenda for Change, hospital medical and dental, very senior
     manager). Adverts on OTHER (care homes, GP practices, private providers) are outside HCHS
     and are counted in the funnel but not kept.
Then, in order, the first rule that applies sets `exclusion_reason`:
  1. a test-account employer
  2. an employer outside HCHS that uses NHS pay (hospices, primary care networks)
  3. test content in the title or text
  4. no title or no advert text
  5. fewer than MIN_WORDS words
  6. fewer than MIN_UNIQ_RATIO distinct words per word
  7. the same vacancy_id again in the same year (the latest created_date is kept)
  8. identical normalised text in the same year (the latest created_date is kept)
Across years: 9. a vacancy_id already kept in an earlier year.

The original ran steps 7 and 8 with pandas' default sort; the same calls are made here on the
same rows in the same order, so the same advert is kept.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import pandas as pd
from selectolax.lexbor import LexborHTMLParser

from ... import storage

YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
NHS_SCHEMES = {"AGENDA_FOR_CHANGE", "HOSPITAL_MEDICAL_DENTAL", "VERY_SENIOR_MANAGER"}

# The four advert text fields, joined into job_text for the cleaning rules.
TEXT_COLS = [
    "job_advert_job_overview",
    "job_advert_job_description",
    "job_advert_about_organisation",
    "job_description",
]
# What the extraction prompt was given (about_organisation left out: no tasks, many tokens).
INPUT_FIELDS = ["job_advert_job_overview", "job_advert_job_description", "job_description"]

KEEP_COLS = [
    "id", "vacancy_id", "closing_date", "employer_address_postcode", "employer_name",
    "contract_duration", "contract_duration_unit", "contract_type", "expected_salary_type",
    "fixed_salary", "full_time", "job_title", "part_time", "per_week_count", "per_week_unit",
    "salary_range_max", "salary_range_min", "salary_time_unit", "created_date",
    "pay_scheme_type", "pay_scheme_band", "advertise_type", "job_share", "flexible_working",
    "remote_working", "compressed_hours", "term_time_hours", "annualised_hours", "organisation",
    "salary_additional_details", *TEXT_COLS,
]  # fmt: skip

MIN_WORDS = 20
MIN_UNIQ_RATIO = 0.05
# Test-environment accounts only: generic "training"/"demo"/"test" caught real employers.
JUNK_EMPLOYER_PATTERNS = [
    r"\bUAT\d*\b",
    r"smoke test",
    r"do not use",
    r"\btest account\b",
    r"implementation team test",
]
JUNK_TEXT_PATTERNS = [
    r"smoke test",
    r"this is a test",
    r"for testing purposes",
    r"\btest vacancy\b",
]
# Employers that use NHS pay but are not in the HCHS workforce totals.
NON_HCHS_EMPLOYER_PATTERNS = [
    r"\bhospice\b",
    r"marie curie",
    r"sue ryder",
    r"\bpcn\b",
    r"primary care network",
]
_JUNK_RE = re.compile("|".join(JUNK_EMPLOYER_PATTERNS), re.IGNORECASE)
_JUNK_TEXT_RE = re.compile("|".join(JUNK_TEXT_PATTERNS), re.IGNORECASE)
_NON_HCHS_RE = re.compile("|".join(NON_HCHS_EMPLOYER_PATTERNS), re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# Leftover Word/Office noise that survives tag stripping.
_NOISE_RE = re.compile(r"(&nbsp;|\u00a0|\u200b|\ufeff|\bmso-[^;\"']+;?)", re.IGNORECASE)

EARLIER_YEAR = "same vacancy id kept in an earlier year"


def strip_nhs_html(value: object) -> str:
    """HTML/entities -> plain text. Unlike the job-board stripper, every value is parsed (so
    entities in plain text are decoded) and Word `mso-` style debris is removed too."""
    if not isinstance(value, str) or not value:
        return ""
    text = LexborHTMLParser(value).text(separator=" ")
    return _WS_RE.sub(" ", _NOISE_RE.sub(" ", text)).strip()


def _norm_key(text: str) -> str:
    key = _WS_RE.sub(" ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()
    return hashlib.md5(key.encode()).hexdigest()


def read_year(src: str, year: int) -> tuple[pd.DataFrame, int]:
    """One year's NHS-scheme adverts (needed columns only), and the file's total row count."""
    with storage.open_input(src) as fh:
        df = pd.read_csv(fh, usecols=lambda c: c in KEEP_COLS, dtype=str, low_memory=False)
    n_all = len(df)
    df["year"] = year
    return df[df["pay_scheme_type"].isin(NHS_SCHEMES)].copy(), n_all


def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    """Rules 1-8 on one year's NHS-scheme adverts. Adds job_text, input_text and
    exclusion_reason; rows keep their order. Pure function (tested)."""
    d = df.copy()
    employer = d["employer_name"].fillna("")
    reason = pd.Series(pd.NA, index=d.index, dtype="string")

    def first(mask, why: str) -> None:
        reason.loc[reason.isna() & mask] = why

    first(employer.str.contains(_JUNK_RE), "test-account employer")
    first(employer.str.contains(_NON_HCHS_RE), "employer outside HCHS (hospice, PCN)")
    d["job_text"] = d[TEXT_COLS].fillna("").agg(" ".join, axis=1).map(strip_nhs_html)
    title = d["job_title"].fillna("")
    first(d["job_text"].str.contains(_JUNK_TEXT_RE) | title.str.contains(_JUNK_TEXT_RE),
          "test or dummy advert")  # fmt: skip
    first(title.str.strip().eq("") | d["job_text"].eq(""), "no title or advert text")
    words = d["job_text"].str.split()
    n_words = words.map(len)
    first(n_words < MIN_WORDS, f"under {MIN_WORDS} words")
    uniq = words.map(lambda w: len(set(w))) / n_words.clip(lower=1)
    first(uniq < MIN_UNIQ_RATIO, "repeated filler text")

    alive = d[reason.isna()]
    s7 = alive.sort_values("created_date").drop_duplicates("vacancy_id", keep="last")
    first(d.index.isin(alive.index) & ~d.index.isin(s7.index), "repeat of the same vacancy id")
    s8 = s7.assign(_k=s7["job_text"].map(_norm_key)).sort_values("created_date")
    s8 = s8.drop_duplicates("_k", keep="last")
    first(d.index.isin(s7.index) & ~d.index.isin(s8.index), "repeat of another advert's text")

    sent = reason.isna()
    d["input_text"] = pd.NA
    if sent.any():
        parts = [d.loc[sent, f].map(strip_nhs_html) for f in INPUT_FIELDS]
        body = pd.concat(parts, axis=1).apply(lambda r: "\n\n".join(p for p in r if p), axis=1)
        d.loc[sent, "input_text"] = (
            "Job title: " + d.loc[sent, "job_title"].map(strip_nhs_html) + "\n\n" + body
        )
    d["exclusion_reason"] = reason
    return d


def clean_years(files: Iterable[tuple[int, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All years: the kept columns of every NHS-scheme advert with its exclusion_reason
    (raw text fields dropped), and the funnel."""
    frames, funnel = [], []
    for year, src in files:
        df, n_all = read_year(src, year)
        d = clean_year(df).drop(columns=[*TEXT_COLS, "job_text"])
        funnel += [
            {"step": "raw adverts", "year": year, "rows": n_all},
            {"step": "on an NHS pay scheme", "year": year, "rows": len(d)},
        ]
        for why, n in d["exclusion_reason"].value_counts().items():
            funnel.append({"step": f"excluded: {why}", "year": year, "rows": int(n)})
        frames.append(d)
    v = pd.concat(frames, ignore_index=True)
    # Rule 9: a vacancy_id kept in more than one year keeps its earliest year's row.
    kept = v["exclusion_reason"].isna()
    dup = kept & v.loc[kept, "vacancy_id"].duplicated(keep="first").reindex(
        v.index, fill_value=False
    )
    v.loc[dup, "exclusion_reason"] = EARLIER_YEAR
    funnel.append({"step": f"excluded: {EARLIER_YEAR}", "year": pd.NA, "rows": int(dup.sum())})
    funnel.append(
        {"step": "sent for scoring", "year": pd.NA, "rows": int(v["exclusion_reason"].isna().sum())}
    )
    return v, pd.DataFrame(funnel)
