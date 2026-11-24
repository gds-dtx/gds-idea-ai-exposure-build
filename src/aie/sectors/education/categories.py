"""Normalise the Education dataset's multi-label categorical fields.

``Job_role`` and ``education_phase`` arrive as a single value, a comma-separated list of
values, or (a handful of rows) an " and "-joined list instead of comma-joined — a data-entry
inconsistency, not a different schema. We split on either, keep every row's categories as a
list, and also expose a ``primary_*`` column (first listed value) for single-value cuts —
mirroring how NHS's ``staff_group`` gives one clean category per row, except here the raw data
genuinely allows more than one role/phase per advert (e.g. a part-time TA/MDSA split post), so
we don't collapse that information away.

One group label in ``Job_role``, "Administration, HR, Data, and Finance", itself contains
commas and the word "and" — it's matched and protected whole BEFORE the generic splitter runs,
so it isn't shredded into four fake categories.

Copied from gds-idea-education-ai-exposure-cleaning (src/education_ai_exposure/categories.py,
commit 6d49b8f) so the build no longer depends on that repo.
The rules are unchanged; comments quoting sample figures were reworded (public repo).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .rules import FREE_TEXT_ROLE_GROUPS, MULTI_LABEL_SPLIT_RE, VALID_JOB_TYPES, VALID_PHASES

_SPLIT_RE = re.compile(MULTI_LABEL_SPLIT_RE)
# Sort longest-first so a group that is a substring of another can't shadow it.
_PROTECTED_ROLE_GROUPS = sorted(FREE_TEXT_ROLE_GROUPS, key=len, reverse=True)
_PLACEHOLDER = "\x00{}\x00"


def _split_multi_label(value: str, protect: list[str] = ()) -> list[str]:
    """Split a raw multi-label string on comma / " and ", protecting any verbatim phrases in
    `protect` from being split internally."""
    if not isinstance(value, str) or not value.strip():
        return []
    text = value
    placeholders: dict[str, str] = {}
    for i, phrase in enumerate(protect):
        if phrase in text:
            key = _PLACEHOLDER.format(i)
            placeholders[key] = phrase
            text = text.replace(phrase, key)
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    return [placeholders.get(p, p) for p in parts]


def normalize_job_role(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``job_role_list`` (list[str]) and ``primary_job_role`` (first entry, or None)."""
    df = df.copy()
    df["job_role_list"] = df["Job_role"].map(
        lambda v: _split_multi_label(v, protect=_PROTECTED_ROLE_GROUPS)
    )
    df["primary_job_role"] = df["job_role_list"].map(lambda xs: xs[0] if xs else None)
    return df


def normalize_education_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``education_phase_list`` and ``primary_education_phase``; flag anything outside
    ``VALID_PHASES`` so new/unexpected tokens surface rather than silently passing through."""
    df = df.copy()
    df["education_phase_list"] = df["education_phase"].map(_split_multi_label)
    df["primary_education_phase"] = df["education_phase_list"].map(lambda xs: xs[0] if xs else None)
    all_tokens = {t for xs in df["education_phase_list"] for t in xs}
    unexpected = all_tokens - VALID_PHASES
    if unexpected:
        import warnings

        warnings.warn(
            f"Unexpected education_phase tokens (not in VALID_PHASES): {unexpected}",
            stacklevel=2,
        )
    return df


def normalize_job_type(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``Job_Type`` against the expected binary; anything else becomes 'Unknown' and
    is counted, not silently coerced."""
    df = df.copy()
    valid = df["Job_Type"].isin(VALID_JOB_TYPES)
    n_unexpected = (~valid & df["Job_Type"].notna()).sum()
    if n_unexpected:
        import warnings

        unexpected_vals = df.loc[~valid & df["Job_Type"].notna(), "Job_Type"].unique().tolist()
        warnings.warn(
            f"{n_unexpected} rows with unexpected Job_Type values: {unexpected_vals}",
            stacklevel=2,
        )
    df["job_type_clean"] = np.where(valid, df["Job_Type"], "Unknown")
    return df


def normalize_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three normalisers in sequence."""
    df = normalize_job_role(df)
    df = normalize_education_phase(df)
    df = normalize_job_type(df)
    return df
