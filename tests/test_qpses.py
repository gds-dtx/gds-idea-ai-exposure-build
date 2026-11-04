import pandas as pd

from aie.sectors.local_govt import qpses


def _q():
    rows = [
        ("Newcastle upon Tyne", "Met District", "North East", 9000),
        ("Gateshead", "Met District", "North East", 5000),
        ("Bromsgrove", "Shire District", "West Midlands", 400),
        ("Redditch", "Shire District", "West Midlands", 500),
        ("Leeds", "Met District", "Yorkshire and the Humber", 25000),
        ("Bradford", "Met District", "Yorkshire and the Humber", 20000),
        ("Babergh", "Shire District", "East of England", 0),
        ("Mid Suffolk", "Shire District", "East of England", 674),
        *[(n, "Shire District", "North West", 100) for n in qpses.CUMBRIA[1:]],
        ("Cumbria", "Shire County", "North West", 7000),
    ]
    q = pd.DataFrame(rows, columns=["qpses_name", "authority_type", "region", "headcount"])
    q["fte"] = q["headcount"]
    q["key"] = q["qpses_name"].map(qpses.normalise)
    return q


def test_normalise_drops_team_prefix_and_council_words():
    assert qpses.normalise("Tenancy Services - Newcastle City Council") == "newcastle"
    assert qpses.normalise("Metropolitan Borough of Gateshead") == "gateshead"
    assert qpses.normalise("Bromsgrove & Redditch Councils") == "bromsgrove and redditch"


def test_match_aliases_and_joint_employers():
    m = qpses.match(
        pd.Series(["Newcastle City Council", "Cumberland Council", "Bromsgrove and Redditch",
                   "Highland Council", None]),
        _q(),
    )  # fmt: skip
    assert m[0] == ["Newcastle upon Tyne"]
    assert m[1] == qpses.CUMBRIA
    assert m[2] == ["Bromsgrove", "Redditch"]
    assert m[3] == [] and m[4] == []


def test_pooling_rules():
    q = _q()
    councils = pd.Series(
        ["Bromsgrove and Redditch"] * 5 + ["Redditch Borough Council"] * 5  # real shared staff
        + ["Leeds and Bradford"] + ["Leeds City Council"] * 50              # one stray advert
        + ["Babergh and Mid Suffolk"]                                       # a member reports 0
        + ["Westmorland and Furness Council"]                               # old names, always
    )  # fmt: skip
    per, ut = qpses.assign(councils, q)
    units = set(ut["unit"])
    assert "Bromsgrove + Redditch" in units
    assert "Babergh + Mid Suffolk" in units
    assert "Bradford + Leeds" not in units and "Leeds" in units
    assert pd.isna(per["qpses_unit"].iloc[10])  # the stray joint advert gets no unit
    cumbria = ut.set_index("unit").loc[" + ".join(sorted(qpses.CUMBRIA))]
    assert cumbria["headcount"] == 7600 and cumbria["authority_type"] == "Shire County"
    assert per["region"].iloc[-1] == "North West"


def test_strata_pool_small_and_donorless_councils():
    ut = pd.DataFrame(
        {
            "unit": ["Big", "Small", "None1", "MetA", "MetB", "Bradford"],
            "region": ["R1", "R1", "R1", "R2", "R2", "R2"],
            "authority_type": ["Met", "District", "District", "Met", "Met", "Met"],
            "fte": [1000.0, 100, 100, 500, 500, 800],
        }
    )
    n = pd.Series({"Big": 50, "Small": 40, "None1": 0, "MetA": 60, "MetB": 60, "Bradford": 0})
    cell = qpses.strata(ut, n, min_adverts=30).set_axis(ut["unit"])
    assert cell["Big"] == "Big"
    # None1 has no adverts and Small (same region and type) has its own cell: None1 is shared
    # out over Small rather than Small being pooled
    assert cell["Small"] == "Small" and cell["None1"] == "R1 | District (shared out)"
    # Bradford has no adverts and no small neighbours of its type: its FTE is shared out over
    # MetA and MetB, which keep their own cells
    assert cell["MetA"] == "MetA" and cell["MetB"] == "MetB"
    assert cell["Bradford"] == "R2 | Met (shared out)"
    targets, shared = qpses.share_out(ut.assign(cell=cell.to_numpy()), "fte")
    assert targets["MetA"] == targets["MetB"] == 900  # 500 each + half of Bradford's 800
    assert targets["Small"] == 200  # its own 100 + None1's 100
    assert targets.sum() == ut["fte"].sum() and "R2 | Met (shared out)" not in targets
    assert shared["R2 | Met (shared out)"]["shared_over"] == ["MetA", "MetB"]
