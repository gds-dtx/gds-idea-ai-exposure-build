"""Public corporations: cleaning rules and the per-body rake, on toy adverts."""

from __future__ import annotations

import pandas as pd
import pytest

from aie.sectors import get
from aie.sectors.public_corporations import targets
from aie.sectors.public_corporations.sector import clean_adverts

BODY = "Responsibilities include managing budgets and writing reports. " * 30  # > 1,000 chars


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {
        "source": "x", "url": "https://example.org/job/1", "salary_min": None,
        "salary_max": None, "hours": "Full Time", "date": "2025-01-01", "document_text": None,
    }  # fmt: skip
    return pd.DataFrame([{**base, "id": str(i), **r} for i, r in enumerate(rows)])


def test_cleaning_rules():
    raw = _raw(
        [
            {"employer": "BBC", "job_title": "Closed", "jd_text": "This advert has now closed."},
            {"employer": "Network Rail", "job_title": "Shell",
             "jd_text": "Location: York | Hours: Full Time\nApply Now Key Information"},
            {"employer": "BBC", "job_title": "Journalist",
             "jd_text": "Journalist Journalism Permanent - Full Time Nairobi, Nairobi City, KE, 0 "
                        "0100 BBC PS - Kenya Job Closing Date: 1/1/25 " + BODY},
            {"employer": "BBC", "job_title": "Researcher",
             "jd_text": "Job Requisition ID: 1 Job Closing Date: 1/1/25 Salford, MAN, GBR, M50 2QH "
                        + BODY},
            {"employer": "BBC", "job_title": "Producer",
             "jd_text": "Producer Production Fixed Term - Full Time Bristol, GB, BS1 6BX "
                        "BBC Studios Production Job Closing Date: 1/1/25 " + BODY},
            {"employer": "Channel 4", "job_title": "Analyst", "date": "2024-01-01",
             "jd_text": "Location: Leeds | Category: Data\n" + BODY},
            {"employer": "Channel 4", "job_title": "Analyst", "date": "2025-06-01",
             "jd_text": "Location: Leeds | Category: Data\n" + BODY},
            {"employer": "Ordnance Survey", "job_title": "Volunteer", "hours": "Volunteering",
             "jd_text": BODY},
            {"employer": "Bank of England", "job_title": "Economist", "date": "2999-12-31",
             "jd_text": "Location: London, United Kingdom | Category: Economics\n" + BODY},
            {"employer": "Network Rail", "job_title": "Template",
             "jd_text": "Location: York | Hours: Full Time\nEnter Department description here. "
                        + BODY},
        ]
    )  # fmt: skip
    d = clean_adverts(raw, today=pd.Timestamp("2026-11-06"))
    r = d["exclusion_reason"].tolist()
    assert r[0] == "advert closed: no job description left"
    assert r[1] == "no job description (listing page only)"
    assert r[2] == "based outside the UK"
    assert pd.isna(r[3])  # Salford, MAN (county), GBR: in the UK
    assert pd.isna(r[4])
    assert r[5].startswith("repeat of a later advert") and pd.isna(r[6])  # latest kept
    assert r[7] == "volunteering or work experience"
    assert pd.isna(r[8]) and pd.isna(d.loc[8, "posted_date"])  # placeholder date dropped
    assert r[9] == "unfilled advert template (placeholder text, no duties)"
    assert d.loc[3, "org_unit"] == "BBC public service"
    assert d.loc[4, "org_unit"] == "BBC commercial"
    assert d.loc[6, "location"] == "Leeds" and d.loc[6, "occupation_group"] == "Data"
    assert not d.loc[6, "body"].startswith("Location")


def test_rake_to_employers():
    s = get("public-corporations")
    adverts = pd.DataFrame(
        {
            "vacancy_id": ["pc-c4-1", "pc-c4-2", "pc-os-1", "pc-os-2"],
            "org_unit": ["Channel 4", "Channel 4", "Ordnance Survey", "Ordnance Survey"],
            "in_scope": [True, True, True, False],
        }
    )
    n_tasks = pd.Series({"pc-c4-1": 5, "pc-c4-2": 2, "pc-os-1": 7, "pc-os-2": 9})
    out = targets.rake_to_employers(s, adverts, n_tasks)
    w = out.weights.set_index("vacancy_id")["weight"]
    c4 = s.config["targets"]["Channel 4"]["value"]
    os_ = s.config["targets"]["Ordnance Survey"]["value"]
    assert w["pc-c4-1"] == pytest.approx(c4)  # pc-c4-2 has < MIN_TASKS
    assert pd.isna(w["pc-c4-2"]) and pd.isna(w["pc-os-2"])
    assert w["pc-os-1"] == pytest.approx(os_)
    assert "Ordnance Survey" in out.diagnostics["small_cells"]
    assert "Network Rail" in out.diagnostics["empty_cells"]


def test_every_target_has_a_reference_file():
    s = get("public-corporations")
    files = set(s.config["reference_sources"])
    assert {t["file"] for t in s.config["targets"].values()} <= files
