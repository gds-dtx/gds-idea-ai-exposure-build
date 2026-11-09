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


def test_weight_on_salaried_runs_the_rake_twice():
    import pandas as pd

    from aie import build
    from aie.sectors.base import WeightOutput

    class Toy:
        slug = "toy"

        def min_tasks_sample(self, adverts, n_tasks):
            return adverts[adverts["in_scope"]]

        def weight(self, release, adverts, ext, n_tasks):
            pool = adverts[adverts["in_scope"]]
            w = adverts["vacancy_id"].map(pd.Series(100.0 / len(pool), index=pool["vacancy_id"]))
            weights = pd.DataFrame(
                {"vacancy_id": adverts["vacancy_id"], "weight_spec": "rake_x", "weight": w,
                 "is_primary": True, "target_source": "t", "target_date": "d"}
            )  # fmt: skip
            return WeightOutput(
                weights, pd.DataFrame({"weight_spec": ["rake_x"]}), {"n": len(pool)}
            )

    adverts = pd.DataFrame(
        {"vacancy_id": list("abcde"), "in_scope": [True] * 4 + [False],
         "salary_annual_gbp": [30_000.0, None, 40_000.0, None, 50_000.0]}
    )  # fmt: skip
    out = build.weight_on_salaried(Toy(), "v2", adverts, None, None)
    w = out.weights.pivot(index="vacancy_id", columns="weight_spec", values="weight")
    assert w.loc["a", "rake_x"] == 50.0 and pd.isna(w.loc["b", "rake_x"])  # 100 over 2 salaried
    assert w.loc["b", "rake_x_all_adverts"] == 25.0  # 100 over 4 usable
    assert pd.isna(w.loc["e", "rake_x"])  # out of scope stays out
    prim = out.weights.groupby("weight_spec")["is_primary"].first().to_dict()
    assert prim == {"rake_x": True, "rake_x_all_adverts": False}
    assert set(out.targets["weight_spec"]) == {"rake_x", "rake_x_all_adverts"}
