import pandas as pd

from aie.sectors.police import cleaning

BODY = " ".join(f"word{i}" for i in range(30))


def _raw(rows):
    d = pd.DataFrame(rows)
    for c in cleaning.RAW_COLS:
        if c not in d:
            d[c] = pd.NA
    d["vacancy_id"] = [f"pol-{i:05d}" for i in range(len(d))]
    return d


def test_normalise_force():
    assert cleaning.normalise_force("Leicestershire Police") == ("Leicestershire", True)
    assert cleaning.normalise_force("Gwent Police - Heddlu Gwent") == ("Gwent", True)
    assert cleaning.normalise_force("The Metropolitan Police Service") == ("Metropolitan", True)
    assert cleaning.normalise_force("PSNI") == ("Northern Ireland", True)
    assert cleaning.normalise_force(
        "Directorate of Standards and Ethics (Ref 24-024) - Cleveland Police"
    ) == ("Cleveland", True)
    assert cleaning.normalise_force("An Garda Siochana") == ("An Garda Siochana", False)
    assert cleaning.normalise_force(None) == ("Unknown", False)


def test_recruitment_pages():
    assert cleaning.is_recruitment_process("Police Constable Transferee")
    assert cleaning.is_recruitment_process("PCDA intake 2025")
    assert cleaning.is_recruitment_process("Register your interest - Switchboard Operator")
    assert not cleaning.is_recruitment_process("Control Room Operator")


def test_strip_html():
    assert cleaning.strip_html("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"
    assert cleaning.strip_html(" plain   text ") == "plain text"
    assert cleaning.strip_html(None) == ""


def _ad(title, force, date):
    return {"job_title": title, "force": force, "jd_text": BODY, "date": date}


def test_clean_adverts_rules_in_order():
    d = cleaning.clean_adverts(
        _raw(
            [
                _ad("Analyst", "Kent Police", "2024-01-01"),
                _ad("Analyst", "Kent Police", "2025-01-01"),
                _ad("Analyst", "Essex Police", "2024-01-01"),
                {"job_title": "Test", "force": "Kent", "jd_text": "This is a test " + BODY},
                {"job_title": "Clerk", "force": "Kent", "jd_text": "too short"},
                {"job_title": "", "force": "Kent", "jd_text": BODY},
                {"job_title": "Filler", "force": "Kent", "jd_text": "x " * 40},
                {"job_title": "Transferee scheme", "force": "Kent", "jd_text": BODY + " other"},
                {"job_title": "Officer", "force": "Garda", "jd_text": BODY + " garda"},
            ]
        )
    )  # fmt: skip
    assert d["exclusion_reason"].tolist() == [
        "repeat of another advert",  # the later Kent copy is kept
        pd.NA,
        pd.NA,  # same text, different force: not a repeat
        "test or dummy advert",
        "under 20 words",
        "no title or advert text",
        "repeated filler text",
        cleaning.RECRUITMENT_PAGE,
        pd.NA,  # not a UK force: scored, the sector sets in_scope = False
    ]
    assert d["is_uk_force"].tolist()[-1] is False
    assert d["force"].tolist()[:3] == ["Kent", "Kent", "Essex"]
    f = cleaning.funnel(d)
    assert f.iloc[0]["rows"] == 9 and f.iloc[-1]["rows"] == 3


def test_raw_readme_flags_processed_inputs():
    from aie import build
    from aie.sectors import get

    police = get("police")
    notes = build.raw_notes(police)
    assert notes["labelled_tasks_full.parquet"]["kind"] == "processed"
    copied = {name: {"source": "s3://x", **n} for name, n in notes.items()}
    readme = build.raw_readme(police, "v1", copied)
    assert "Not all of these files are raw" in readme
    assert "`labelled_tasks_full.parquet` is a PROCESSED output" in readme
