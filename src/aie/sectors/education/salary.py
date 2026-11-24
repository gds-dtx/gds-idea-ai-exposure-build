"""Parse the free-text ``salary`` field into structured min/max/basis.

Unlike the NHS/Civil Service sources (which already carry numeric salary-range columns), this
dataset has a single free-text field that mixes pay ranges, single figures, hourly/daily/annual
bases, "Actual" vs "FTE" qualifiers, pay-scale codes, and outright garbage (data-entry noise
like "grgetgerg"). Parsing is deliberately light-touch: pull out the £ amount(s) and a basis
keyword, flag "actual"/"fte" qualifiers, and leave anything that doesn't contain a plausible
£ figure as unparsed rather than guessing. A failed parse does not drop the row — salary isn't
needed for task extraction, only for optional downstream pay analysis.

Copied from gds-idea-education-ai-exposure-cleaning (src/education_ai_exposure/salary.py,
commit 6d49b8f) so the build no longer depends on that repo.
The rules are unchanged; comments quoting sample figures were reworded (public repo).
"""

from __future__ import annotations

import re

import pandas as pd

from .rules import SALARY_AMOUNT_RE, SALARY_BASIS_KEYWORDS, SALARY_TEXT_CATEGORIES

_AMOUNT_RE = re.compile(SALARY_AMOUNT_RE)
_BASIS_RE = {
    basis: re.compile("|".join(pats), re.IGNORECASE)
    for basis, pats in SALARY_BASIS_KEYWORDS.items()
}
_TEXT_CAT_RE = {cat: re.compile(pat, re.IGNORECASE) for cat, pat in SALARY_TEXT_CATEGORIES.items()}
_ACTUAL_RE = re.compile(r"\bactual\b", re.IGNORECASE)
_FTE_RE = re.compile(r"\bfte\b", re.IGNORECASE)


def _basis(text: str) -> str | None:
    for basis, rx in _BASIS_RE.items():
        if rx.search(text):
            return basis
    return None


def _text_category(text: str) -> str | None:
    for cat, rx in _TEXT_CAT_RE.items():
        if rx.search(text):
            return cat
    return None


def parse_salary_text(text: str | float) -> dict:
    """Parse one salary free-text value. Always returns the same dict shape."""
    out = {
        "salary_min": None,
        "salary_max": None,
        "salary_basis": None,
        "salary_is_actual": False,
        "salary_is_fte": False,
        "salary_text_category": None,
        "salary_parsed": False,
    }
    if not isinstance(text, str) or not text.strip():
        return out

    amounts = [float(m.replace(",", "")) for m in _AMOUNT_RE.findall(text)]
    if amounts:
        # Take the FIRST quoted range/figure as primary. Some rows quote a second "Actual"
        # or "FTE" range on a later line (e.g. pro-rata vs full-time-equivalent) — we don't
        # try to disambiguate which is "the" salary, we just keep the first one, flagged.
        out["salary_min"] = min(amounts[:2])
        out["salary_max"] = max(amounts[:2])
        out["salary_basis"] = _basis(text) or "annual"  # default: most adverts are annual
        out["salary_is_actual"] = bool(_ACTUAL_RE.search(text))
        out["salary_is_fte"] = bool(_FTE_RE.search(text))
        out["salary_parsed"] = True
    else:
        out["salary_text_category"] = _text_category(text)

    return out


def parse_salary_column(series: pd.Series) -> pd.DataFrame:
    """Vectorised (well, mapped) parse of a whole ``salary`` column into new columns."""
    parsed = series.map(parse_salary_text)
    df = pd.DataFrame(parsed.tolist(), index=series.index)
    return df


def salary_funnel_summary(df: pd.DataFrame) -> pd.Series:
    """Quick coverage breakdown for sanity-checking the parse (fraction parsed / text-category /
    genuinely unparseable), for use in run_cleaning.py's report."""
    n = len(df)
    parsed = df["salary_parsed"].sum()
    text_cat = df["salary_text_category"].notna().sum()
    unparsed_garbage = n - parsed - text_cat - df["salary"].isna().sum()
    return pd.Series(
        {
            "n": n,
            "missing": df["salary"].isna().sum(),
            "numeric_parsed": parsed,
            "text_category": text_cat,
            "unparseable_garbage": max(unparsed_garbage, 0),
        }
    )
