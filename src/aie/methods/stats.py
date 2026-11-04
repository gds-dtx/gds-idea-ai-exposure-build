from __future__ import annotations

import numpy as np


def weighted_quantile(values, weights, q: float) -> float:
    """Weighted quantile (q in [0, 1]) using mid-point cumulative weights.

    The same construction as NHS/Education's `data.weighted_quantile` and CS's
    model_labour_impacts.ipynb, so threshold values match theirs exactly.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = ~np.isnan(values) & ~np.isnan(weights)
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= weights.sum()
    return float(np.interp(q, cum, values))
