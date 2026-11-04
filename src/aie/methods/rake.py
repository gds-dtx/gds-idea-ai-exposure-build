"""Iterative proportional fitting (raking) to one-way margins, plus diagnostics.

Dimension-agnostic: a sector passes {column: {category: target FTE}} for as many margins as its
official statistics support. Ported from the NHS rake (itself adapted from the Civil Service
rake); the Education copy was already verbatim.

The CS 3D post-stratification spec is not here yet: CS v1 adopts its published weights (see
sectors/civil_service). Porting it is part of folding gds-idea-ai-exposure in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def normalise(dist: dict[str, float]) -> dict[str, float]:
    """Scale a margin to sum to 1.0, ignoring NaN targets."""
    total = sum(v for v in dist.values() if not pd.isna(v))
    if total == 0:
        return dist
    return {k: v / total for k, v in dist.items() if not pd.isna(v)}


def filter_to_supported(
    df: pd.DataFrame, margins: dict[str, dict[str, float]]
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Drop rows whose category has no target in any margin. Returns (kept, dropped counts)."""
    mask = pd.Series(True, index=df.index)
    dropped = {}
    for col, m in margins.items():
        supported = {k for k, v in m.items() if not pd.isna(v)}
        col_mask = df[col].isin(supported)
        if (~col_mask).any():
            dropped[col] = {
                str(k): int(v) for k, v in df.loc[~col_mask, col].value_counts().items()
            }
        mask &= col_mask
    return df[mask].reset_index(drop=True), dropped


def calculate_raked_weights(
    df: pd.DataFrame,
    margins: dict[str, dict[str, float]],
    *,
    target_fte: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[pd.Series, dict]:
    """IPF to one-way margins. Every category in `df` must have a target (filter first).

    Returns (weights aligned to df.index and summing to target_fte, convergence info).
    """
    norm = {col: normalise(m) for col, m in margins.items()}
    groups = {col: list(df.groupby(col, observed=True).indices.items()) for col in margins}

    w = np.ones(len(df))
    prev_change = np.inf
    status, iterations = "max_iter", max_iter
    for iteration in range(max_iter):
        w_old = w.copy()
        for col in margins:
            target = norm[col]
            w_sum = w.sum()
            for key, idx in groups[col]:
                share = w[idx].sum() / w_sum
                if share:
                    w[idx] *= target[key] / share
        change = np.abs(w - w_old).max() / w.mean()
        if change < tol:
            status, iterations = "converged", iteration + 1
            break
        # Mutually inconsistent margins (different population totals) settle at a fixed point
        # where the per-cycle change stops shrinking: stop and report the residual.
        if abs(prev_change - change) < tol:
            status, iterations = "fixed_point", iteration + 1
            break
        prev_change = change
    return pd.Series(w * (target_fte / w.sum()), index=df.index), {
        "status": status,
        "iterations": iterations,
    }


def verify_margins(
    weights: pd.Series, df: pd.DataFrame, margins: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Max absolute FTE error per margin (≈0 on convergence)."""
    total = weights.sum()
    errors = {}
    for col, m in margins.items():
        actual = weights.groupby(df[col], observed=True).sum()
        expected = (pd.Series(normalise(m)) * total).reindex(actual.index)
        errors[col] = float((actual - expected).abs().max())
    return errors


def effective_sample_size(w: pd.Series) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2)."""
    w = w.dropna()
    return float(w.sum() ** 2 / (w**2).sum())


@dataclass
class RakeResult:
    weights: pd.DataFrame  # vacancy_id, weight (NaN where unweightable)
    targets: pd.DataFrame  # dimension, category, target_fte
    diagnostics: dict = field(default_factory=dict)


def rake(
    sample: pd.DataFrame,
    margins: dict[str, dict[str, float]],
    *,
    target_fte: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> RakeResult:
    """Full raking run on a prepared sample (vacancy_id + one column per margin)."""
    est, dropped = filter_to_supported(sample, margins)
    w, conv = calculate_raked_weights(
        est, margins, target_fte=target_fte, max_iter=max_iter, tol=tol
    )
    errors = verify_margins(w, est, margins)
    weights = sample[["vacancy_id"]].merge(
        est[["vacancy_id"]].assign(weight=w.to_numpy()), on="vacancy_id", how="left"
    )
    targets = pd.DataFrame(
        [
            {"dimension": d, "category": c, "target_fte": v}
            for d, m in margins.items()
            for c, v in m.items()
        ]
    )
    n_eff = effective_sample_size(w)
    return RakeResult(
        weights=weights,
        targets=targets,
        diagnostics={
            **conv,
            "dimensions": list(margins),
            "target_fte": float(target_fte),
            "n_sample": len(sample),
            "n_weighted": int(weights["weight"].notna().sum()),
            "weighted_fte": float(w.sum()),
            "effective_n": n_eff,
            "design_effect": len(w) / n_eff,
            "max_margin_error_fte": errors,
            "dropped_without_target": dropped,
        },
    )
