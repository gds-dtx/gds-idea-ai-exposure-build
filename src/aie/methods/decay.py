"""Aggregate task exposure to vacancy level with geometric task-order decay.

Tasks are extracted in advert order, and we assume earlier-listed responsibilities take more
of the role's time. Task i (1-based) gets weight w_i ∝ delta^(i-1), normalised to sum to 1
within the vacancy. delta = 1.0 is uniform. The standing default (methodology.toml) is 0.75.

Ported from gds-idea-ai-exposure/src/task_weights.py via the NHS/Education copies (identical
algorithm). Checked against CS's published vacancy_exposure.parquet: max abs difference 0.0
for both the ensemble and the gpt-oss-120b columns at delta = 0.75.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decay_weights(tasks: pd.DataFrame, delta: float) -> pd.Series:
    """Normalised decay weights aligned to `tasks.index`. `tasks` must be sorted by
    (vacancy_id, task_index)."""
    diffs = tasks.groupby("vacancy_id", sort=False)["task_index"].diff()
    if (diffs < 0).any():
        raise ValueError("tasks must be sorted by (vacancy_id, task_index)")
    seq = tasks.groupby("vacancy_id", sort=False).cumcount().to_numpy()
    raw = pd.Series(np.power(delta, seq), index=tasks.index)
    return raw / raw.groupby(tasks["vacancy_id"], sort=False).transform("sum")


def weighted_moments(tasks: pd.DataFrame, weights: pd.Series, value_col: str) -> pd.DataFrame:
    """Weighted mean and population variance per vacancy.

    NaN-safe: where some tasks are unscored, weights are renormalised over the scored ones.
    """
    x = tasks[value_col]
    g = tasks["vacancy_id"]
    present = x.notna()
    w = weights.where(present, 0.0)
    w_sum = w.groupby(g, sort=False).transform("sum")
    w = pd.Series(np.where(w_sum > 0, w / w_sum, np.nan), index=weights.index)
    xf = x.where(present, 0.0)
    m1 = (w * xf).groupby(g, sort=False).sum(min_count=1)
    m2 = (w * xf.pow(2)).groupby(g, sort=False).sum(min_count=1)
    return pd.DataFrame({"mean": m1, "var": (m2 - m1.pow(2)).clip(lower=0.0)})


def vacancy_moments(
    tasks: pd.DataFrame, value_col: str, primary: float, comparison: float
) -> pd.DataFrame:
    """One row per vacancy: n_tasks, mean/var at the primary decay and at the comparison decay.

    `tasks` needs vacancy_id, task_index and `value_col`.
    """
    t = tasks[["vacancy_id", "task_index", value_col]].sort_values(
        ["vacancy_id", "task_index"], kind="stable"
    )
    t = t.reset_index(drop=True)
    out = t.groupby("vacancy_id", sort=False).size().rename("n_tasks").to_frame()
    for delta, suffix in ((primary, ""), (comparison, "_uniform")):
        m = weighted_moments(t, decay_weights(t, delta), value_col)
        out[f"mean_exposure{suffix}"] = m["mean"]
        out[f"var_exposure{suffix}"] = m["var"]
    out["decay"] = primary
    return out.reset_index()
