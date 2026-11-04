"""A tiny synthetic sector built and published with the build package on a local AIE_ROOT,
then cut by this engine."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aie import build, catalog, publish, read
from aie.cuts import CustomThresholds, CutSpec, Filter, engine, options, run_cut, values
from aie.cuts.engine import score_model
from aie.sectors.base import CleanOutput, ScoreOutput, Sector, WeightOutput

CONFIG = """
name = "Toy"
source = "synthetic"
[scoring]
primary_model = "gpt-oss-120b"
[weighting]
primary_spec = "unweighted"
placeholder_reason = "test"
"""
N = 60


class Toy(Sector):
    slug = "police"

    def __init__(self, tmp: Path):
        self.package_dir = tmp
        (tmp / "config.toml").write_text(CONFIG)
        super().__init__()

    def clean(self, release):
        ids = [f"pol-{i}" for i in range(N)]
        adverts = pd.DataFrame(
            {
                "vacancy_id": ids,
                "job_title": "Analyst",
                "occupation_group": ["Police staff", "Police officer"] * (N // 2),
                "region": "London",
                "salary_annual_gbp": 30_000.0,
                "headcount": 1,
                "in_scope": [True] * (N - 1) + [False],
                "exclusion_reason": [pd.NA] * (N - 1) + ["not a UK police force"],
            }
        )
        ext = pd.DataFrame({"vacancy_id": ids, "is_flagged": [i % 3 == 0 for i in range(N)]})
        return CleanOutput(adverts, ext)

    def score(self, release):
        rng = np.random.default_rng(1)
        rows = [
            (f"pol-{i}", k, f"task {k}", float(rng.choice(np.arange(11) / 10)))
            for i in range(N)
            for k in range(1, 2 + i % 6)
        ]
        t = pd.DataFrame(rows, columns=["vacancy_id", "task_index", "task_text", "exposure_score"])
        t["score_model"] = "gpt-oss-120b"
        ms = t[["vacancy_id", "task_index", "exposure_score"]].assign(model="gpt-oss-120b")
        return ScoreOutput(t, ms)

    def weight(self, release, adverts, ext, n_tasks) -> WeightOutput:
        return self.unweighted(adverts, n_tasks, "test")


@pytest.fixture
def published(tmp_path, monkeypatch):
    monkeypatch.setenv("AIE_ROOT", str(tmp_path / "root"))
    read.clear_cache()
    toy = Toy(tmp_path)
    for stage in ("clean", "score", "weight"):
        build.run(toy, stage, "v1")
    publish.publish(toy, "v1")
    catalog.write_sectors_table([toy])
    return toy


def test_cut_end_to_end(published):
    res = run_cut(CutSpec(sectors=["police"], group_by=["occupation_group"], min_cell_adverts=1))
    n_eligible = sum(1 for i in range(N - 1) if 1 + i % 6 >= 3)
    assert res.data["n_adverts"].sum() == n_eligible
    assert res.data["fte"].sum() == n_eligible
    assert any("advert counts" in w for w in res.meta["warnings"])
    assert res.meta["releases"] == {"police": "v1"}

    flagged = run_cut(
        CutSpec(
            sectors=["police"],
            filters=[Filter(column="is_flagged", op="eq", values=[True])],
            min_cell_adverts=1,
        )
    )
    assert 0 < flagged.data["n_adverts"].iloc[0] < n_eligible

    blanked = run_cut(
        CutSpec(sectors=["police"], group_by=["occupation_group"], min_cell_adverts=100)
    )
    assert blanked.data["fte"].isna().all() and blanked.data["n_adverts"].notna().all()

    assert res.save("toy").endswith("cuts/police/release=v1/toy.csv")
    opts = options()
    assert opts["sectors"]["police"]["weight_specs"] == ["unweighted"]
    assert "is_flagged" in opts["sectors"]["police"]["sector_dimensions"]
    assert values("is_flagged", ["police"]) == [False, True]


def test_vacancy_level_cut(published):
    spec = CutSpec(
        sectors=["police"],
        level="vacancy",
        columns=[
            "vacancy_id",
            "occupation_group",
            "fte",
            "mean_exposure",
            "quadrant",
            "is_flagged",
        ],
        filters=[Filter(column="occupation_group", values=["Police staff"])],
    )
    res = run_cut(spec)
    groups = run_cut(
        CutSpec(
            sectors=["police"],
            group_by=["occupation_group"],
            filters=spec.filters,
            min_cell_adverts=1,
        )
    ).data
    assert list(res.data.columns) == spec.columns
    assert res.data["vacancy_id"].is_unique
    assert len(res.data) == groups["n_adverts"].iloc[0]
    assert (res.data["occupation_group"] == "Police staff").all()
    # the advert rows add back up to the grouped figure
    weighted = (res.data["fte"] * res.data["mean_exposure"]).sum() / res.data["fte"].sum()
    assert weighted == pytest.approx(groups["mean_exposure"].iloc[0])


def test_cross_sector_cut_must_use_common_model():
    m = {"comparability": {"cross_sector_score_model": "gpt-oss-120b"}}
    with pytest.raises(ValueError, match="cross-sector"):
        score_model(CutSpec(sectors=["nhs", "civil-service"], score_model="ensemble-8"), m)
    assert score_model(CutSpec(sectors=["nhs", "civil-service"]), m) == "gpt-oss-120b"


def _by_quadrant(**kw):
    spec = CutSpec(sectors=["police"], group_by=["quadrant"], min_cell_adverts=1, **kw)
    res = run_cut(spec)
    return res, res.data.set_index("quadrant")


def test_custom_limits_at_fixed_values_match_fixed(published):
    from aie import settings

    q = settings.methodology()["quadrants"]
    _, fixed = _by_quadrant()
    res, custom = _by_quadrant(
        threshold_basis="custom",
        custom_thresholds=CustomThresholds(
            mean_hi=q["fixed_mean_hi"],
            high_var_lo=q["fixed_high_var_lo"],
            low_var_hi=q["fixed_low_var_hi"],
        ),
    )
    pd.testing.assert_frame_equal(fixed, custom)
    assert any("Custom quadrant limits" in w for w in res.meta["warnings"])
    assert res.meta["custom_thresholds"]["mean_hi"] == q["fixed_mean_hi"]


def test_custom_percentiles_read_off_the_reference_sector(published, monkeypatch):
    # With the toy as the reference sector, 25 / 25 is its own quartiles.
    monkeypatch.setattr(engine, "REFERENCE_SECTOR", "police")
    engine._REFERENCE.clear()
    _, sq = _by_quadrant(threshold_basis="sector_quartiles")
    res, _ = _by_quadrant(
        threshold_basis="custom", custom_thresholds=CustomThresholds(mean_top_pct=25, var_pct=25)
    )
    own = read.thresholds("police", "v1").set_index("basis").loc["sector_quartiles"]
    for k in ("mean_hi", "high_var_lo", "low_var_hi"):
        assert res.meta["custom_thresholds"][k] == round(own[k], 4)
    _, wide = _by_quadrant(
        threshold_basis="custom", custom_thresholds=CustomThresholds(mean_top_pct=50, var_pct=50)
    )
    assert wide["cashable_gbp"].sum() > sq["cashable_gbp"].sum()


def test_custom_limits_are_validated():
    with pytest.raises(ValueError):
        CustomThresholds(mean_top_pct=25)  # needs var_pct too
    with pytest.raises(ValueError):
        CustomThresholds(mean_top_pct=80, var_pct=25)  # out of range
    with pytest.raises(ValueError):
        CutSpec(sectors=["nhs"], threshold_basis="custom")  # no limits given
