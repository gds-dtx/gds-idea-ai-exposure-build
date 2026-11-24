"""Police cleaning: raw force job-board adverts -> one row per raw advert with a reason if it
is not used.

Ported from gds-idea-police-ai-exposure-cleaning (cleaning.py, forces.py, run_cleaning.py) so
the build no longer reads that repo's output. Applied to the same raw file it gives exactly the
same advert set, force labels and flags (checked 2026-11-24).

Rules, in order; the first that applies sets `exclusion_reason`:
  1. test or dummy content (title or text)
  2. no title or no advert text
  3. fewer than MIN_WORDS words
  4. fewer than MIN_UNIQ_RATIO distinct words per word (repeated filler)
  5. a repeat: same force, title and text as another advert; the latest date is kept (ties: the
     later row in the file)
  6. a recruitment-route page (transferee schemes, talent pools, entry routes, promotion
     processes), which describes how to join rather than a job
Rows 1-6 are not sent for scoring. Adverts from bodies that are not UK police forces are
scored but carry `in_scope = False` (the sector sets that reason).
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd
from selectolax.lexbor import LexborHTMLParser

RAW_COLS = [
    "source", "id", "url", "job_title", "force", "salary_min", "salary_max",
    "hours", "date", "jd_text",
]  # fmt: skip

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

RECRUITMENT_PAGE = "recruitment-route page, not a role"

# --- force labels ---------------------------------------------------------------------------

_TERRITORIAL = [
    "Avon & Somerset", "Bedfordshire", "Cambridgeshire", "Cheshire", "City of London", "Cleveland",
    "Cumbria", "Derbyshire", "Devon & Cornwall", "Dorset", "Durham", "Dyfed Powys", "Essex",
    "Gloucestershire", "Greater Manchester", "Gwent", "Hampshire", "Hertfordshire", "Humberside",
    "Kent", "Lancashire", "Leicestershire", "Lincolnshire", "Merseyside", "Metropolitan",
    "Norfolk", "North Wales", "North Yorkshire", "Northamptonshire", "Northumbria",
    "Nottinghamshire", "South Wales", "South Yorkshire", "Staffordshire", "Suffolk", "Surrey",
    "Sussex", "Thames Valley", "Warwickshire", "West Mercia", "West Midlands", "West Yorkshire",
    "Wiltshire", "Scotland", "Northern Ireland",
]  # fmt: skip
_NATIONAL = [
    "British Transport Police", "Ministry of Defence Police", "Civil Nuclear Constabulary",
    "Surrey & Sussex Police Collaboration", "Norfolk and Suffolk Constabularies",
]  # fmt: skip
_ALIASES = {  # lower-cased cleaned raw label -> canonical
    "metropolitan police service": "Metropolitan",
    "psni": "Northern Ireland",
    "police service of northern ireland": "Northern Ireland",
    "british transport police authority": "British Transport Police",
    "civil nuclear constabulary (cnc)": "Civil Nuclear Constabulary",
    "civil nuclear (formerly ukaea)": "Civil Nuclear Constabulary",
    "ministry of defence police": "Ministry of Defence Police",
    "city of london": "City of London",
}
_CANON = {c.lower(): c for c in _TERRITORIAL + _NATIONAL}
_SUFFIX_RE = re.compile(r"\s*-\s*heddlu.*$|\s+(police|constabulary)$", re.IGNORECASE)
_THE_RE = re.compile(r"^the\s+", re.IGNORECASE)

_RECRUIT_RE = re.compile(
    r"transferee|re-?joiner|expressions? of interest|register (?:your )?interest"
    r"|talent (?:bank|pool)"
    r"|new recruits?\b|degree apprenticeship|\bPCDA\b|\bDHEP\b|\bPPD\b|entry programme|promotion"
    r"|police officer \(non degree",
    re.IGNORECASE,
)


def normalise_force(raw: object) -> tuple[str, bool]:
    """(canonical force label, is a UK police force) for a raw `force` value.

    UK police forces are the territorial forces, Police Scotland, PSNI, BTP, MDP, CNC and the
    two joint collaborations; anything else (Garda, overseas forces, contractors, HMRC,
    universities) keeps a cleaned label and is not in scope.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown", False
    cleaned = _THE_RE.sub("", raw.strip())
    # e.g. "Directorate of Standards and Ethics (Ref 24-024) - Cleveland Police"
    if " - " in cleaned and cleaned.lower().endswith(" police") and "heddlu" not in cleaned.lower():
        cleaned = cleaned.rsplit(" - ", 1)[1]
    stripped = _SUFFIX_RE.sub("", _SUFFIX_RE.sub("", cleaned)).strip().lower()
    for key in (cleaned.lower(), stripped):
        key = _ALIASES.get(key, key).lower()
        if key in _CANON:
            return _CANON[key], True
    return cleaned, False


def is_recruitment_process(title: object) -> bool:
    """True for recruitment-route and talent-pool pages rather than a role."""
    return bool(isinstance(title, str) and _RECRUIT_RE.search(title))


# --- text -----------------------------------------------------------------------------------


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


def clean_adverts(raw: pd.DataFrame) -> pd.DataFrame:
    """Every raw row, in file order, with job_text, force, is_uk_force, is_recruitment_process
    and exclusion_reason (NA when the advert is sent for scoring). Pure function (tested).

    `raw` must already carry `vacancy_id`.
    """
    d = raw.copy().reset_index(drop=True)
    d["job_text"] = d["jd_text"].map(strip_html)
    d["force_raw"] = d["force"]
    norm = d["force_raw"].map(normalise_force)
    d["force"] = norm.map(lambda t: t[0])
    d["is_uk_force"] = norm.map(lambda t: t[1])
    d["is_recruitment_process"] = d["job_title"].map(is_recruitment_process)

    title = d["job_title"].fillna("")
    words = d["job_text"].str.split()
    n_words = words.map(len)
    uniq = words.map(lambda w: len(set(w))) / n_words.clip(lower=1)
    reason = pd.Series(pd.NA, index=d.index, dtype="string")

    def first(mask: pd.Series, why: str) -> None:
        reason.loc[reason.isna() & mask] = why

    first(d["job_text"].str.contains(_JUNK_TEXT_RE) | title.str.contains(_JUNK_TEXT_RE),
          "test or dummy advert")  # fmt: skip
    first(title.str.strip().eq("") | d["job_text"].eq(""), "no title or advert text")
    first(n_words < MIN_WORDS, f"under {MIN_WORDS} words")
    first(uniq < MIN_UNIQ_RATIO, "repeated filler text")

    # Repeats across boards: same force + normalised title + text. The force is in the key
    # because national campaign adverts share identical text across forces. Keep the latest
    # date; ties keep the later row (stable sort).
    alive = d[reason.isna()]
    key = (alive["force"].fillna("") + " " + alive["job_title"].fillna("") + " "
           + alive["job_text"]).map(_norm_key)  # fmt: skip
    order = alive.assign(_key=key).sort_values("date", kind="stable")
    keep = order.drop_duplicates("_key", keep="last").index
    first(d.index.isin(alive.index) & ~d.index.isin(keep), "repeat of another advert")
    first(d["is_recruitment_process"], RECRUITMENT_PAGE)

    d["exclusion_reason"] = reason
    return d


def funnel(d: pd.DataFrame) -> pd.DataFrame:
    rows = [{"step": "raw adverts", "rows": len(d)}]
    for why, n in d["exclusion_reason"].value_counts().items():
        rows.append({"step": f"excluded: {why}", "rows": int(n)})
    rows.append({"step": "sent for scoring", "rows": int(d["exclusion_reason"].isna().sum())})
    return pd.DataFrame(rows)
