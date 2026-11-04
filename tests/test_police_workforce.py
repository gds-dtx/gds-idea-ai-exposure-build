import pandas as pd

from aie.sectors.police import workforce


def test_unit_of_aliases_and_joint_units():
    f = pd.Series(["Avon & Somerset", "Surrey & Sussex Police Collaboration", "Sussex",
                   "Metropolitan", "Kent"])  # fmt: skip
    assert workforce.unit_of(f).tolist() == [
        "Avon and Somerset", "Surrey and Sussex", "Surrey and Sussex", "Metropolitan Police",
        "Kent",
    ]  # fmt: skip


def test_strata_own_region_national_and_pcso():
    ut = pd.DataFrame(
        {
            "unit": ["A", "B", "C", "D", "A"],
            "occupation_group": ["Police staff"] * 4 + ["PCSO"],
            "region": ["R1", "R1", "R1", "R2", "R1"],
            "fte": [100.0, 50.0, 40.0, 30.0, 10.0],
            "n_adverts": [40, 20, 15, 2, 5],
        }
    )
    cells = workforce.strata(ut, min_adverts=30).tolist()
    assert cells == [
        "Police staff | A",
        "Police staff | R1 (pooled)",  # B + C = 35 adverts, enough for a regional pool
        "Police staff | R1 (pooled)",
        "Police staff | national (pooled)",  # R2 has only 2 adverts
        "PCSO | national (pooled)",
    ]


def test_salary_from_text_reads_only_stated_annual_salaries():
    from aie.sectors.police.salary_text import salary_from_text

    assert salary_from_text("Salary Grade Range - £33,618 - £37,281 Working Hours") == 35449.5
    assert salary_from_text("Salary: £46,140 Plus a location allowance of £2,623") == 46140
    # an allowance first, then the salary
    assert salary_from_text("Salary: location allowance £2,623 and £30,000 per annum") == 30000
    # hourly rates, no keyword, or nothing stated: no salary (never estimated)
    assert salary_from_text("Hourly rate for locum £24.00 - £35.00 per hour") is None
    assert salary_from_text("We pay £40,000 to the right person") is None
    assert salary_from_text("Competitive salary and great benefits") is None
