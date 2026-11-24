"""Schools cleaning: Teaching Vacancies CSV shards -> one row per raw advert, with a reason if it
is not used, plus the parsed salary, normalised role and phase, DfE category and school match.

Ported from gds-idea-education-ai-exposure-cleaning (loading.py, cleaning.py,
scripts/run_cleaning.py) so the build no longer reads that repo's output. The order of steps
is the original's:

  1. clean (rules below), on the pooled shards
  2. parse the free-text salary (salary.py)
  3. normalise job role, phase and job type (categories.py)
  4. DfE category from role and title (role_groups.py)
  5. match the school name to GIAS state-funded schools (schools.py)

Steps 4 and 5 were applied to the original's cleaned pool outside its scripts; here they are
part of the build. Cleaning rules, in order; the first that applies sets `exclusion_reason`:
  1. test or dummy content (title or text)
  2. no title or no advert text
  3. fewer than MIN_WORDS words
  4. fewer than MIN_UNIQ_RATIO distinct words per word
  5. identical normalised text (the latest publish_on is kept)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import pandas as pd
from selectolax.lexbor import LexborHTMLParser

from ... import storage
from .categories import normalize_all
from .role_groups import assign_dfe_category
from .rules import JUNK_TEXT_PATTERNS, MIN_UNIQ_RATIO, MIN_WORDS, RAW_COLS
from .salary import parse_salary_column
from .schools import match_vacancies

_JUNK_TEXT_RE = re.compile("|".join(JUNK_TEXT_PATTERNS), re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"(&nbsp;| |​|﻿|\bmso-[^;\"']+;?)", re.IGNORECASE)


def strip_html(value: object) -> str:
    """HTML/entities -> plain text; every value is parsed, and Word `mso-` debris removed."""
    if not isinstance(value, str) or not value:
        return ""
    text = LexborHTMLParser(value).text(separator=" ")
    return _WS_RE.sub(" ", _NOISE_RE.sub(" ", text)).strip()


def _norm_key(text: str) -> str:
    key = _WS_RE.sub(" ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()
    return hashlib.md5(key.encode()).hexdigest()


def read_shards(paths: Iterable[str]) -> pd.DataFrame:
    """The shards in file-name order, with vacancy_id "{shard:02d}-{row:07d}" and year."""
    frames = []
    for i, src in enumerate(sorted(paths)):
        with storage.open_input(src) as fh:
            df = pd.read_csv(fh, usecols=lambda c: c in RAW_COLS, dtype=str, low_memory=False)
        df["vacancy_id"] = [f"{i:02d}-{j:07d}" for j in range(len(df))]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["publish_on"], errors="coerce").dt.year
    return df


def exclusion_reasons(d: pd.DataFrame) -> pd.Series:
    """Cleaning rules 1-5 (d needs job_title, job_text, vacancy_id, publish_on)."""
    title = d["job_title"].fillna("")
    reason = pd.Series(pd.NA, index=d.index, dtype="string")

    def first(mask, why: str) -> None:
        reason.loc[reason.isna() & mask] = why

    first(d["job_text"].str.contains(_JUNK_TEXT_RE) | title.str.contains(_JUNK_TEXT_RE),
          "test or dummy advert")  # fmt: skip
    first(title.str.strip().eq("") | d["job_text"].eq(""), "no title or advert text")
    words = d["job_text"].str.split()
    n_words = words.map(len)
    first(n_words < MIN_WORDS, f"under {MIN_WORDS} words")
    uniq = words.map(lambda w: len(set(w))) / n_words.clip(lower=1)
    first(uniq < MIN_UNIQ_RATIO, "repeated filler text")
    # The original dedups vacancy_id first (a no-op: ids are made unique) and then text, each
    # after a sort on publish_on; the same calls are made here so the same advert is kept.
    alive = d[reason.isna()]
    s6 = alive.sort_values("publish_on").drop_duplicates("vacancy_id", keep="last")
    s7 = s6.assign(_k=s6["job_text"].map(_norm_key)).sort_values("publish_on")
    s7 = s7.drop_duplicates("_k", keep="last")
    first(d.index.isin(s6.index) & ~d.index.isin(s7.index), "repeat of another advert's text")
    return reason


def clean_adverts(raw: pd.DataFrame, gias: pd.DataFrame) -> pd.DataFrame:
    """Every raw row with exclusion_reason; adverts sent for scoring also get the salary
    parse, normalised categories, DfE category and school match."""
    d = raw.copy()
    d["job_text"] = d["job_advert"].fillna("").map(strip_html)
    d["exclusion_reason"] = exclusion_reasons(d)
    sent = d[d["exclusion_reason"].isna()].copy()
    sal = parse_salary_column(sent["salary"])
    sent = pd.concat([sent, sal], axis=1)
    sent = normalize_all(sent)
    sent = assign_dfe_category(sent)
    sent = match_vacancies(sent, gias)
    new_cols = [c for c in sent.columns if c not in d.columns]
    return d.join(sent[new_cols])


def funnel(d: pd.DataFrame) -> pd.DataFrame:
    rows = [{"step": "raw adverts", "rows": len(d)}]
    for why, n in d["exclusion_reason"].value_counts().items():
        rows.append({"step": f"excluded: {why}", "rows": int(n)})
    rows.append({"step": "sent for scoring", "rows": int(d["exclusion_reason"].isna().sum())})
    return pd.DataFrame(rows)
