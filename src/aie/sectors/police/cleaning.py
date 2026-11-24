"""Police cleaning: raw force job-board adverts -> one row per raw advert with a reason if it
is not used.

Ported from gds-idea-police-ai-exposure-cleaning (cleaning.py, forces.py, run_cleaning.py) so
the build no longer reads that repo's output. Applied to the same raw file it gives exactly the
same advert set, force labels and flags (checked 2026-11-24).

Rules, in order; the first that applies sets `exclusion_reason`:
  1-5. the shared job-board rules (sectors/advert_cleaning.py), with the force in the repeat key
       because national campaign adverts share identical text across forces
  6. a recruitment-route page (transferee schemes, talent pools, entry routes, promotion
     processes), which describes how to join rather than a job
Rows 1-6 are not sent for scoring. Adverts from bodies that are not UK police forces are
scored but carry `in_scope = False` (the sector sets that reason).
"""

from __future__ import annotations

import re

import pandas as pd

from ..advert_cleaning import exclusion_reasons, funnel, strip_html

__all__ = ["clean_adverts", "funnel", "is_recruitment_process", "normalise_force", "strip_html"]

RAW_COLS = [
    "source", "id", "url", "job_title", "force", "salary_min", "salary_max",
    "hours", "date", "jd_text",
]  # fmt: skip

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

    reason = exclusion_reasons(d, key_prefix=d["force"])
    reason.loc[reason.isna() & d["is_recruitment_process"]] = RECRUITMENT_PAGE
    d["exclusion_reason"] = reason
    return d
