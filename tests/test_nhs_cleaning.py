import pandas as pd

from aie.sectors.nhs import cleaning
from aie.sectors.nhs.staff_groups import assign_staff_group

BODY = " ".join(f"word{i}" for i in range(30))


def _year(rows):
    d = pd.DataFrame(rows)
    for c in cleaning.KEEP_COLS:
        if c not in d:
            d[c] = pd.NA
    d["pay_scheme_type"] = d["pay_scheme_type"].fillna("AGENDA_FOR_CHANGE")
    return d


def _ad(vid, created, title="Staff Nurse", employer="Leeds Teaching Hospitals", text=BODY):
    return {"vacancy_id": vid, "created_date": created, "job_title": title,
            "employer_name": employer, "job_description": text}  # fmt: skip


def test_strip_nhs_html():
    assert cleaning.strip_nhs_html("<p>Band&nbsp;5</p>") == "Band 5"
    assert cleaning.strip_nhs_html("Fish &amp; chips") == "Fish & chips"  # always parsed
    assert cleaning.strip_nhs_html('<p style="x">a mso-font:Arial; b</p>') == "a b"


def test_clean_year_rules_in_order():
    d = cleaning.clean_year(
        _year(
            [
                _ad("1", "2024-01-01"),
                _ad("1", "2024-02-01"),  # same vacancy id: the later one is kept
                _ad("2", "2024-03-01"),  # same text as id 1: the later one is kept
                _ad("3", "2024-01-01", employer="NHS BSA-UAT2"),
                _ad("4", "2024-01-01", employer="St Gemma's Hospice"),
                _ad("5", "2024-01-01", text="short text"),
                _ad("6", "2024-01-01", text="This is a test " + BODY),
            ]
        )
    )
    assert d["exclusion_reason"].tolist() == [
        "repeat of the same vacancy id",
        "repeat of another advert's text",
        pd.NA,
        "test-account employer",
        "employer outside HCHS (hospice, PCN)",
        "under 20 words",
        "test or dummy advert",
    ]
    assert d["input_text"].iloc[2] == "Job title: Staff Nurse\n\n" + BODY


def test_staff_groups_from_pay_scheme_and_title():
    df = pd.DataFrame(
        {
            "job_title": ["Staff Nurse", "Consultant Cardiologist", "Chief Executive", "Midwife"],
            "pay_scheme_type": ["AGENDA_FOR_CHANGE", "HOSPITAL_MEDICAL_DENTAL",
                                "VERY_SENIOR_MANAGER", "AGENDA_FOR_CHANGE"],
        }
    )  # fmt: skip
    assert assign_staff_group(df).tolist() == [
        "Nurses & health visitors",
        "HCHS doctors",
        "NHS infrastructure support",
        "Midwives",
    ]


def test_postcode_area_regions():
    from aie.sectors.nhs import regions

    lk = regions.area_to_nhse_region()
    assert len(lk) == 99
    assert lk["LS"] == "North East and Yorkshire" and lk["B"] == "Midlands"
    assert lk["SY"] == "Midlands" and "CF" not in lk  # Shropshire in, Wales out
