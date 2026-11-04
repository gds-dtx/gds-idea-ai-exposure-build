import pandas as pd
import pytest

from aie import metrics, settings
from aie.methods import quadrants


def test_current_version_and_unknown_version():
    assert settings.methodology("m1") is settings.methodology()
    with pytest.raises(ValueError, match="unknown methodology version"):
        settings.methodology("m0")


def test_scenario_is_labelled_and_leaves_primary_alone():
    s = settings.with_changes({"decay.primary": 0.9}, "decay_090")
    assert s["version"] == "m1+decay_090"
    assert s["decay"]["primary"] == 0.9
    assert settings.methodology()["decay"]["primary"] == 0.75


def test_option_b_rederives_fixed_thresholds_unless_given():
    assert settings.with_changes({"decay.primary": 0.6}, "x")["quadrants"]["fixed_source"] == (
        "rederive"
    )
    q = settings.with_changes({"quadrants.high_mean_top": 0.3}, "x")["quadrants"]
    assert q["fixed_source"] == "rederive"
    given = settings.with_changes({"decay.primary": 0.6, "quadrants.fixed_mean_hi": 0.4}, "x")[
        "quadrants"
    ]
    assert given["fixed_source"] == "values"
    val = settings.with_changes({"valuation.overhead_multiplier": 1.3}, "x")["quadrants"]
    assert val["fixed_source"] == "values"


def test_rejects_unknown_and_pre_raking_settings():
    with pytest.raises(ValueError, match="unknown methodology setting"):
        settings.with_changes({"decay.primry": 0.9}, "x")
    with pytest.raises(ValueError, match="new release"):
        settings.with_changes({"population.min_tasks": 4}, "x")


def test_compute_metrics_uses_given_fixed_thresholds():
    tasks = pd.DataFrame(
        {
            "vacancy_id": ["a"] * 3 + ["b"] * 3,
            "task_index": [1, 2, 3] * 2,
            "score": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
        }
    )
    roles = pd.DataFrame({"vacancy_id": ["a", "b"], "weight": [1.0, 1.0], "salary": [100.0] * 2})
    loose = quadrants.Thresholds("fixed", 0.05, 0.01, 0.02, float("nan"))
    rows, thr = metrics.compute_metrics(tasks, roles, "m", fixed=loose)
    assert rows["quadrant"].tolist() == ["Displacement-type", "Displacement-type"]
    assert thr.iloc[0]["mean_hi"] == 0.05
