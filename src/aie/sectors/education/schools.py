"""Match adverts to DfE's GIAS state-funded-schools register, to (a) confirm population
scope (our source data has no sector/employer-type column, unlike NHS's `pay_scheme_type`, so
this is the only lever we have to screen out independent-school adverts) and (b) recover a
region dimension we otherwise don't have at all.

Light-touch and heuristic by design: extract a candidate school/trust name from the advert
text, normalise, and exact-match against GIAS's "open state-funded schools" extract (DfE's own
canonical allowlist — a match is state-funded BY CONSTRUCTION, no independent schools in that
file). Unmatched rows are NOT assumed independent — failure to extract/match a name is at least
as likely as the school genuinely being independent, so unmatched stays "unknown", not excluded.
See README "Population scope" for the honest coverage numbers and what's still a known gap.

Copied from gds-idea-education-ai-exposure-cleaning (src/education_ai_exposure/schools.py,
commit 6d49b8f) so the build no longer depends on that repo.
The rules are unchanged; comments quoting sample figures were reworded (public repo).
"""

from __future__ import annotations

import re

import pandas as pd

# School/trust-name suffix words the regex anchors on. Order doesn't matter (used in a
# non-capturing alternation); kept to common identifiers that terminate a UK school/trust name.
_SUFFIX = (
    r"(?:School|Schools|Academy|Academies|College|Colleges|Trust|Centre|Nursery|"
    r"PRU|UTC|University Technical College)"
)
# A run of Title-Case words (allowing short lowercase connectors: of/and/the/&/for/for) ending
# in one of the suffix words. Deliberately permissive — this is a first-pass heuristic, not NER.
_NAME_RE = re.compile(
    r"\b((?:[A-Z][\w'.-]*|of|and|the|The|&|for)(?:\s+(?:[A-Z][\w'.-]*|of|and|the|The|&|for)){0,7}"
    rf"\s+{_SUFFIX})\b"
)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
# Generic words stripped before matching so e.g. "Primary School" / "C of E School" variants
# collapse onto the same normalised key as GIAS's own naming.
_STOPWORDS = {"the", "of", "and", "for"}
# GIAS always spells this out in full; adverts commonly abbreviate it.
_COFE_RE = re.compile(r"\bc\s*\.?\s*of\s*\.?\s*e\.?\b", re.IGNORECASE)
# Leading connector words ("At The Regis School, we are...") that the capitalised-run regex
# picks up because they're sentence-initial capitals, not part of the school's actual name.
_LEADING_RE = re.compile(r"^(?:at|in|to|from|join|visit)\s+", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Lowercase, expand common abbreviations, strip punctuation, drop stopwords, collapse ws."""
    if not isinstance(name, str):
        return ""
    text = _COFE_RE.sub("church of england", name.lower())
    text = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return _WS_RE.sub(" ", " ".join(tokens)).strip()


def extract_candidate_name(job_title: str, job_text: str, search_chars: int = 600) -> str | None:
    """Heuristic extraction of a candidate school/trust name from the title or the advert's
    opening text (school identity is almost always stated up front; searching the whole advert
    would pick up unrelated capitalised phrases further down, e.g. partner-organisation names)."""
    for source in (job_title or "", (job_text or "")[:search_chars]):
        m = _NAME_RE.search(source)
        if m:
            return _LEADING_RE.sub("", m.group(1).strip())
    return None


def load_gias(path: str) -> pd.DataFrame:
    """Load the GIAS 'open state-funded schools' extract, keep Open establishments, and attach
    a normalised-name key for matching."""
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df = df[df["EstablishmentStatus (name)"].isin(["Open", "Open, but proposed to close"])].copy()
    df["_name_key"] = df["EstablishmentName"].map(normalize_name)
    # A handful of names collide (e.g. multiple "St Mary's Primary School" nationally) — keep
    # first for the lookup but note collision rate separately for transparency.
    return df


def build_lookup(gias: pd.DataFrame) -> tuple[dict[str, dict], pd.Series]:
    """(name_key -> first-matching GIAS row's fields, collision counts per name_key)."""
    counts = gias["_name_key"].value_counts()
    cols = ["URN", "LA (name)", "GOR (name)", "PhaseOfEducation (name)", "EstablishmentName"]
    first = gias.drop_duplicates("_name_key", keep="first").set_index("_name_key")[cols]
    lookup = first.to_dict(orient="index")
    return lookup, counts


def match_vacancies(df: pd.DataFrame, gias: pd.DataFrame) -> pd.DataFrame:
    """Add candidate_school_name, school_match_* columns, and state_funded_confirmed (bool)."""
    lookup, collisions = build_lookup(gias)
    out = df.copy()
    candidates = [
        extract_candidate_name(t, x) for t, x in zip(out["job_title"], out["job_text"], strict=True)
    ]
    out["candidate_school_name"] = candidates
    keys = [normalize_name(c) if c else None for c in candidates]
    matches = [lookup.get(k) if k else None for k in keys]
    out["school_match_urn"] = [m["URN"] if m else None for m in matches]
    out["school_la"] = [m["LA (name)"] if m else None for m in matches]
    out["school_region"] = [m["GOR (name)"] if m else None for m in matches]
    out["school_phase_gias"] = [m["PhaseOfEducation (name)"] if m else None for m in matches]
    out["school_name_ambiguous"] = [
        bool(k and collisions.get(k, 0) > 1) for k in keys
    ]  # matched, but the name isn't unique nationally — URN picked is only a best guess
    out["state_funded_confirmed"] = out["school_match_urn"].notna()
    return out
