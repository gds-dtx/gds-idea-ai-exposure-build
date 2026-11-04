"""£ valuations, expressed PER UNIT OF WEIGHT so any weighting spec can be applied later.

    opportunity   salary x mean_exposure                 every role
    cashable      salary                                 Displacement-type (post removed)
    cashable_share salary x mean_exposure                Displacement-type (report beside cashable)
    non_cashable  salary x mean_exposure                 Augmentation-type + Concentrated

Multiply by `weight` (FTE) and sum to get a total. Displacement-type is excluded from
non-cashable so the same posts are never counted twice; Insulated is in neither.

None of these is a realised saving: mean_exposure is technical automation potential. Salary
excludes on-costs unless overhead_multiplier is changed in methodology.toml.

Ported from the NHS/Education cashable.py + monetisation.py (identical to CS's
model_labour_impacts.ipynb split).
"""

from __future__ import annotations

import pandas as pd

from .. import settings


def per_fte_values(
    salary: pd.Series, mean_exposure: pd.Series, quadrant: pd.Series, m: dict | None = None
) -> pd.DataFrame:
    v = (m or settings.methodology())["valuation"]
    cost = salary * v["overhead_multiplier"]
    share = cost * mean_exposure * v["optimism_bias"]
    is_cash = quadrant.isin(v["cashable_categories"]).fillna(False).astype(bool)
    is_non = quadrant.isin(v["non_cashable_categories"]).fillna(False).astype(bool)
    return pd.DataFrame(
        {
            "opportunity": share,
            "cashable": cost.where(is_cash, 0.0).where(cost.notna()),
            "cashable_share": share.where(is_cash, 0.0).where(share.notna()),
            "non_cashable": share.where(is_non, 0.0).where(share.notna()),
        },
        index=salary.index,
    )


def format_money(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if abs(v) >= 1e9:
        return f"£{v / 1e9:.2f}bn"
    return f"£{v / 1e6:.0f}m"
