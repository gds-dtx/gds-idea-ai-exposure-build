"""Role classification: the published paper's quadrant table (2026-09-30).

A role is summarised by the mean (mu) and VARIANCE (sigma^2, not std, not normalised) of its
decay-weighted task exposures. Thresholds:

    mean_hi       mu cut-off for the "high" stratum
    high_var_lo   variance cut WITHIN the high-mean stratum
    low_var_hi    variance cut WITHIN the low-mean stratum

| Category          | Condition                               |
|-------------------|-----------------------------------------|
| Displacement-type | mu >= mean_hi  AND  var <  high_var_lo  |
| Augmentation-type | mu >= mean_hi  AND  var >= high_var_lo  |
| Concentrated      | mu <  mean_hi  AND  var >= low_var_hi   |
| Insulated         | mu <  mean_hi  AND  var <  low_var_hi   |

Two threshold bases (methodology.toml):
  * fixed (PRIMARY): CS's own 1D ensemble quartiles (values in methodology.toml).
  * sector_quartiles (SENSITIVITY): the same construction re-derived on a sector's own weighted
    distribution: mean_hi = weighted 75th pct of mu; high_var_lo = weighted 25th pct of var in
    the high stratum; low_var_hi = weighted 75th pct of var in the low stratum.

Roles with fewer than MIN_TASKS tasks are "Unclassified"; roles with no score from the model in
question (a few CS adverts lack a gpt-oss score) are "Unscored". MIN_TASKS is also applied before
raking, so in practice these roles have no weight and the four categories sum to 100%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .. import settings
from .stats import weighted_quantile


@dataclass(frozen=True)
class Thresholds:
    basis: str
    mean_hi: float
    high_var_lo: float
    low_var_hi: float
    task_hi: float

    def as_dict(self) -> dict:
        return asdict(self)


def task_hi(task_scores: pd.Series, task_weights: pd.Series, m: dict | None = None) -> float:
    """Task exposure at/above which a task counts as automatable (top task_top_pct)."""
    q = (m or settings.methodology())["quadrants"]["task_top_pct"]
    return weighted_quantile(task_scores, task_weights, 1 - q)


def fixed_thresholds(task_hi_value: float, m: dict | None = None) -> Thresholds:
    q = (m or settings.methodology())["quadrants"]
    return Thresholds(
        basis="fixed",
        mean_hi=q["fixed_mean_hi"],
        high_var_lo=q["fixed_high_var_lo"],
        low_var_hi=q["fixed_low_var_hi"],
        task_hi=task_hi_value,
    )


def sector_quartile_thresholds(
    roles: pd.DataFrame,
    task_hi_value: float,
    high_mean_top: float | None = None,
    disp_low_bottom: float | None = None,
    m: dict | None = None,
) -> Thresholds:
    """Weighted within-stratum quartiles. `roles` needs mean_exposure, var_exposure, n_tasks,
    weight; only weighted roles with n_tasks >= MIN_TASKS define the cut-offs.

    `high_mean_top` / `disp_low_bottom` default to methodology.toml; pass others to sweep a
    cut-off sensitivity grid. `m` is a methodology dict (default: methodology.toml).
    """
    m = m or settings.methodology()
    q = m["quadrants"]
    top = q["high_mean_top"] if high_mean_top is None else high_mean_top
    low = q["disp_low_bottom"] if disp_low_bottom is None else disp_low_bottom
    rel = roles[(roles["n_tasks"] >= m["population"]["min_tasks"]) & roles["weight"].notna()]
    mean_hi = weighted_quantile(rel["mean_exposure"], rel["weight"], 1 - top)
    hi = rel[rel["mean_exposure"] >= mean_hi]
    lo = rel[rel["mean_exposure"] < mean_hi]
    return Thresholds(
        basis="sector_quartiles",
        mean_hi=mean_hi,
        high_var_lo=weighted_quantile(hi["var_exposure"], hi["weight"], low),
        low_var_hi=weighted_quantile(lo["var_exposure"], lo["weight"], 1 - low),
        task_hi=task_hi_value,
    )


def classify(roles: pd.DataFrame, thr: Thresholds, m: dict | None = None) -> pd.Series:
    """Quadrant label per role (aligned to `roles.index`)."""
    min_tasks = (m or settings.methodology())["population"]["min_tasks"]
    mu = roles["mean_exposure"].to_numpy(float)
    var = roles["var_exposure"].to_numpy(float)
    high = mu >= thr.mean_hi
    out = np.select(
        [
            roles["n_tasks"].to_numpy(float) < min_tasks,
            np.isnan(mu),
            high & (var < thr.high_var_lo),
            high,
            var >= thr.low_var_hi,
        ],
        ["Unclassified", "Unscored", "Displacement-type", "Augmentation-type", "Concentrated"],
        default="Insulated",
    )
    return pd.Series(out, index=roles.index, dtype="string")
