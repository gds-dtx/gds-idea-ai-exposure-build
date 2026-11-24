"""Light-touch advert cleaning shared by the scraped job-board sectors (police, local government).

Ported from the original cleaning repos (gds-idea-police-ai-exposure-cleaning,
gds-idea-local-govt-ai-exposure-cleaning), whose cleaning.py files were identical apart from the
repeat key. The rules only remove adverts with no usable signal. Every raw row is kept; the
first rule that applies sets `exclusion_reason` (NA = sent for scoring):

  1. test or dummy content (title or text)
  2. no title or no advert text
  3. fewer than MIN_WORDS words
  4. fewer than MIN_UNIQ_RATIO distinct words per word (repeated filler)
  5. a repeat of another advert on the sector's key (normalised title + text, plus e.g. the
     force for police); the latest date is kept, ties keep the later row in the file
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd
from selectolax.lexbor import LexborHTMLParser

MIN_WORDS = 20
MIN_UNIQ_RATIO = 0.05
JUNK_TEXT_PATTERNS = [
    r"smoke test",
    r"this is a test",
    r"for testing purposes",
    r"\btest vacancy\b",
    r"\bdummy (?:vacancy|advert|job)\b",
    r"\bplease ignore this\b",
]
_JUNK_TEXT_RE = re.compile("|".join(JUNK_TEXT_PATTERNS), re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"(&nbsp;| |​|﻿)", re.IGNORECASE)

REPEAT = "repeat of another advert"


def strip_html(value: object) -> str:
    """HTML and entities -> plain text; plain strings are only whitespace-normalised."""
    if not isinstance(value, str) or not value:
        return ""
    if "<" in value:
        value = LexborHTMLParser(value).text(separator=" ")
    return _WS_RE.sub(" ", _NOISE_RE.sub(" ", value)).strip()


def _norm_key(text: str) -> str:
    key = _WS_RE.sub(" ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()
    return hashlib.md5(key.encode()).hexdigest()


def exclusion_reasons(d: pd.DataFrame, key_prefix: pd.Series | None = None) -> pd.Series:
    """The first cleaning rule each row fails (NA if none). `d` needs job_title, job_text and
    date; `key_prefix` (e.g. the force) is put in front of title + text in the repeat key."""
    title = d["job_title"].fillna("")
    words = d["job_text"].str.split()
    n_words = words.map(len)
    uniq = words.map(lambda w: len(set(w))) / n_words.clip(lower=1)
    reason = pd.Series(pd.NA, index=d.index, dtype="string")

    def first(mask, why: str) -> None:
        reason.loc[reason.isna() & mask] = why

    first(d["job_text"].str.contains(_JUNK_TEXT_RE) | title.str.contains(_JUNK_TEXT_RE),
          "test or dummy advert")  # fmt: skip
    first(title.str.strip().eq("") | d["job_text"].eq(""), "no title or advert text")
    first(n_words < MIN_WORDS, f"under {MIN_WORDS} words")
    first(uniq < MIN_UNIQ_RATIO, "repeated filler text")

    alive = d[reason.isna()]
    text = alive["job_title"].fillna("") + " " + alive["job_text"]
    if key_prefix is not None:
        text = key_prefix.loc[alive.index].fillna("") + " " + text
    order = alive.assign(_key=text.map(_norm_key)).sort_values("date", kind="stable")
    keep = order.drop_duplicates("_key", keep="last").index
    first(d.index.isin(alive.index) & ~d.index.isin(keep), REPEAT)
    return reason


def funnel(d: pd.DataFrame) -> pd.DataFrame:
    rows = [{"step": "raw adverts", "rows": len(d)}]
    for why, n in d["exclusion_reason"].value_counts().items():
        rows.append({"step": f"excluded: {why}", "rows": int(n)})
    rows.append({"step": "sent for scoring", "rows": int(d["exclusion_reason"].isna().sum())})
    return pd.DataFrame(rows)
