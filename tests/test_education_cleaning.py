import pandas as pd

from aie.sectors.education import cleaning

BODY = " ".join(f"word{i}" for i in range(30))


def test_strip_html_parses_everything():
    assert cleaning.strip_html("<ul><li>Teach&nbsp;maths</li></ul>") == "Teach maths"
    assert cleaning.strip_html("Pupils &amp; staff") == "Pupils & staff"


def test_exclusion_reasons_keep_latest_repost():
    d = pd.DataFrame(
        {
            "vacancy_id": ["00-0000000", "00-0000001", "00-0000002", "00-0000003"],
            "job_title": ["Teacher", "Teacher", "", "TA"],
            "job_text": [BODY, BODY, BODY, "short"],
            "publish_on": ["2024-02-01", "2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )
    assert cleaning.exclusion_reasons(d).tolist() == [
        pd.NA,  # the later of two identical adverts is kept
        "repeat of another advert's text",
        "no title or advert text",
        "under 20 words",
    ]
