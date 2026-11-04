import numpy as np
import pandas as pd
import pytest

from aie import contract
from aie.methods import decay, quadrants, rake, valuation
from aie.methods.stats import weighted_quantile


def test_decay_weights_sum_to_one_and_front_load():
    t = pd.DataFrame({"vacancy_id": ["a"] * 3 + ["b"] * 2, "task_index": [1, 2, 3, 1, 2]})
    w = decay.decay_weights(t, 0.75)
    assert w.groupby(t["vacancy_id"]).sum().round(12).eq(1).all()
    assert w.iloc[0] / w.iloc[1] == pytest.approx(1 / 0.75)


def test_vacancy_moments_known_values():
    t = pd.DataFrame({"vacancy_id": ["a", "a"], "task_index": [1, 2], "score": [0.2, 0.6]})
    out = decay.vacancy_moments(t, "score", primary=0.5, comparison=1.0).iloc[0]
    # weights 2/3, 1/3
    assert out["mean_exposure"] == pytest.approx(0.2 * 2 / 3 + 0.6 / 3)
    assert out["mean_exposure_uniform"] == pytest.approx(0.4)
    assert out["var_exposure_uniform"] == pytest.approx(0.04)


def test_unscored_task_keeps_position():
    t = pd.DataFrame(
        {"vacancy_id": ["a"] * 3, "task_index": [1, 2, 3], "score": [0.4, np.nan, 0.8]}
    )
    out = decay.vacancy_moments(t, "score", primary=0.5, comparison=1.0).iloc[0]
    # weights 1, 0.5, 0.25 -> drop the middle -> 0.8, 0.2 after renormalising
    assert out["mean_exposure"] == pytest.approx(0.4 * 0.8 + 0.8 * 0.2)


def test_quadrant_table():
    thr = quadrants.Thresholds("fixed", mean_hi=0.4, high_var_lo=0.01, low_var_hi=0.02, task_hi=0.5)
    roles = pd.DataFrame(
        {
            "mean_exposure": [0.5, 0.5, 0.3, 0.3, 0.5],
            "var_exposure": [0.005, 0.02, 0.03, 0.01, 0.005],
            "n_tasks": [5, 5, 5, 5, 2],
        }
    )
    assert quadrants.classify(roles, thr).tolist() == [
        "Displacement-type",
        "Augmentation-type",
        "Concentrated",
        "Insulated",
        "Unclassified",
    ]


def test_valuation_never_double_counts():
    q = pd.Series(["Displacement-type", "Augmentation-type", "Concentrated", "Insulated"])
    v = valuation.per_fte_values(pd.Series([100.0] * 4), pd.Series([0.5] * 4), q)
    assert v["cashable"].tolist() == [100, 0, 0, 0]
    assert v["non_cashable"].tolist() == [0, 50, 50, 0]
    assert v["opportunity"].tolist() == [50] * 4


def test_weighted_quantile_matches_unweighted_median():
    assert weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == pytest.approx(2)


def test_rake_reproduces_margins():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "vacancy_id": [str(i) for i in range(500)],
            "a": rng.choice(["x", "y"], 500),
            "b": rng.choice(["p", "q", "r"], 500),
        }
    )
    margins = {"a": {"x": 60, "y": 40}, "b": {"p": 50, "q": 30, "r": 20}}
    res = rake.rake(df, margins, target_fte=100)
    assert res.diagnostics["status"] == "converged"
    assert max(res.diagnostics["max_margin_error_fte"].values()) < 1e-3
    assert res.weights["weight"].sum() == pytest.approx(100)


def test_contract_rejects_extra_columns_and_duplicates():
    with pytest.raises(contract.ContractError):
        contract.conform(pd.DataFrame({"vacancy_id": ["a"], "surprise": [1]}), "adverts")
    with pytest.raises(contract.ContractError):
        contract.conform(pd.DataFrame({"vacancy_id": ["a", "a"]}), "vacancies")


def test_prefixed_id_is_idempotent():
    s = contract.prefixed_id("nhs", pd.Series(["1", "nhs-2"]))
    assert s.tolist() == ["nhs-1", "nhs-2"]
