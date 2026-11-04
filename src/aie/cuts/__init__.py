"""Cuts: self-service tables over the curated layer, for every sector.

Used by the dashboard's Data Download page and the analysis repo. Three calls:

    options()            what can be chosen: sectors, releases, dimensions, measures, models
    values(col, ...)     distinct values for a filter dropdown
    run_cut(spec)        the table + metadata (result.to_csv(), result.save(name));
                         level="vacancy" gives one row per advert instead of groups
"""

from __future__ import annotations

import pandas as pd

from .. import catalog, read
from .engine import CutResult, run_cut
from .spec import (
    CORE_DIMENSIONS,
    MEASURES,
    VACANCY_COLUMNS,
    CustomThresholds,
    CutSpec,
    Filter,
)

__all__ = [
    "CustomThresholds",
    "CutResult",
    "CutSpec",
    "Filter",
    "clear_cache",
    "options",
    "run_cut",
    "values",
]

clear_cache = read.clear_cache


def options() -> dict:
    """Everything a front end needs to build its controls (JSON-serialisable)."""
    rel = catalog.releases()
    rel = rel[rel["published_at"].notna()]
    m = read.methodology()
    cross = m["comparability"]["cross_sector_score_model"]
    sectors = {}
    for slug, mine in rel.groupby("sector"):
        info = read.sector_info(slug)
        latest = mine.sort_values("published_at").iloc[-1]["release"]
        sectors[slug] = {
            "name": info["name"],
            "releases": mine[["release", "status", "published_at"]].to_dict("records"),
            "primary_score_model": info["primary_score_model"],
            "score_models": list(dict.fromkeys([info["primary_score_model"], cross])),
            "weight_specs": list(read.weight_specs(slug, latest)),
            "primary_weight_spec": info["primary_weight_spec"],
            # one-sector cuts can also group/filter by these
            "sector_dimensions": read.sector_columns(slug, latest),
        }
    return {
        "sectors": sectors,
        "dimensions": CORE_DIMENSIONS,
        "measures": MEASURES,
        # level="vacancy" cuts: one row per advert with these columns
        "vacancy_columns": VACANCY_COLUMNS,
        "threshold_bases": ["fixed", "sector_quartiles", "custom"],
        # Slider ranges for custom quadrant limits; 25 / 25 reproduces the fixed thresholds.
        "custom_thresholds": {
            "mean_top_pct": {"min": 5, "max": 50, "step": 5, "default": 25},
            "var_pct": {"min": 5, "max": 50, "step": 5, "default": 25},
            "fixed": {
                "mean_hi": m["quadrants"]["fixed_mean_hi"],
                "high_var_lo": m["quadrants"]["fixed_high_var_lo"],
                "low_var_hi": m["quadrants"]["fixed_low_var_hi"],
            },
        },
        "methodology_version": m["version"],
        "cross_sector_score_model": cross,
    }


def values(column: str, sectors: list[str], release: str = "latest", limit: int = 500) -> list:
    """Distinct values of a dimension, most common first, for filter dropdowns."""
    out = []
    for s in sectors:
        r, _ = read.resolve_release(s, release)
        if column == "sector":
            out.append(pd.Series([s]))
        else:
            out.append(read.partition("vacancies", s, r, ["vacancy_id", column])[column])
    return pd.concat(out).dropna().value_counts().index[:limit].tolist()
