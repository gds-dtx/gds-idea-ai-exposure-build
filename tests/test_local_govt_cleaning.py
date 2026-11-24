import pandas as pd

from aie.sectors.local_govt.sector import RAW_COLS, clean_adverts

BODY = " ".join(f"word{i}" for i in range(30))


def test_clean_adverts_text_and_repeats():
    rows = [
        {"job_title": "Planner", "council": "Leeds", "jd_text": BODY, "document_text": ""},
        # same title and text at another council: a repeat (council is not in the key)
        {"job_title": "Planner", "council": "York", "jd_text": BODY, "document_text": ""},
        # short advert rescued by its attached job description
        {"job_title": "Clerk", "council": "Leeds", "jd_text": "short", "document_text": BODY},
        {"job_title": "Clerk", "council": "Leeds", "jd_text": "<p>too short</p>"},
    ]
    d = pd.DataFrame(rows)
    for c in RAW_COLS:
        if c not in d:
            d[c] = pd.NA
    d["vacancy_id"] = [f"lg-{i:07d}" for i in range(len(d))]
    out = clean_adverts(d)
    assert out["exclusion_reason"].tolist() == [
        "repeat of another advert",
        pd.NA,
        pd.NA,
        "under 20 words",
    ]
    assert out["job_text"].iloc[2] == "short " + BODY
    assert out["job_text"].iloc[3] == "too short"
