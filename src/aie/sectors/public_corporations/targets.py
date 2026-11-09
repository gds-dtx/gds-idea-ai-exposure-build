"""Raking targets for public corporations, and the rake.

There is no single official table of staff by public corporation. ONS Public Sector
Employment gives only the total for all public corporations, and not for the NAO, Network
Rail, National Highways or TfL, which ONS classifies elsewhere. So each body's own published
staff number is the target: its annual report and accounts for 2024/25 (British Steel
publishes no accounts as a public body yet; its figure is the 2021 direct headcount in the
Oxford Economics report it published in 2026). Each figure, what it counts and where it is
printed are in config.toml [targets] and docs/public_corporations.md; the PDFs are copied
into the release's reference/ folder.

The rake: 1D post-stratification. One cell per body (the BBC as two cells, public service
and commercial, matching its accounts). Every eligible advert in a cell gets target / n.
Primary spec `rake_employer_salaried` counts only adverts that state a salary in n, so £
totals scale up from the salaried adverts to the whole body; `rake_employer` counts every
usable advert (then adverts without a salary add £0 and £ totals cover only the salaried share
of staff). Cells are NOT pooled across bodies, because the bodies do unrelated work; cells
with fewer than `small_cell_adverts` adverts are flagged in the diagnostics instead (Ordnance
Survey has under 10), and their figures should be quoted with that caveat or left out.

Measures differ: most targets are average FTE, but the Bank of England and the CAA publish
average headcount and British Steel a headcount, so their weights are slightly overstated in
FTE terms.
"""

from __future__ import annotations

import pandas as pd

from ..base import Sector, WeightOutput

SPEC = "rake_employer"  # every usable advert shares its organisation's staff number
SALARIED = "rake_employer_salaried"  # only adverts that state a salary (primary)


def target_table(sector: Sector) -> pd.DataFrame:
    rows = [
        {"cell": cell, "target": float(t["value"]), "measure": t["measure"], "source": t["source"]}
        for cell, t in sector.config["targets"].items()
    ]
    return pd.DataFrame(rows)


def _cell_weights(pool: pd.DataFrame, tgt: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    n = pool.groupby("org_unit").size().reindex(tgt.index, fill_value=0)
    per_advert = (tgt["target"] / n).where(n > 0)
    w = pool.set_index("vacancy_id")["org_unit"].map(per_advert)
    cells = pd.DataFrame(
        {"target": tgt["target"], "n_adverts": n, "weight_per_advert": per_advert}
    ).reset_index(names="cell")
    return w, cells


def _diagnostics(spec: str, w: pd.Series, cells: pd.DataFrame, tgt, small: int) -> dict:
    ww = w.dropna()
    cells = cells.assign(small_cell=cells["n_adverts"] < small)
    return {
        "spec": spec,
        "n_weighted": int(ww.size),
        "weighted_total": float(ww.sum()),
        "target_total": float(tgt["target"].sum()),
        "target_covered": float(cells.loc[cells["n_adverts"] > 0, "target"].sum()),
        "design_effect": float(1 + ww.var(ddof=0) / ww.mean() ** 2) if ww.size else None,
        "max_weight": float(ww.max()) if ww.size else None,
        "small_cell_threshold": small,
        "small_cells": cells.loc[cells["small_cell"], "cell"].tolist(),
        "empty_cells": cells.loc[cells["n_adverts"] == 0, "cell"].tolist(),
        "cells": cells.to_dict(orient="records"),
    }


def rake_to_employers(sector: Sector, adverts: pd.DataFrame, n_tasks: pd.Series) -> WeightOutput:
    """Two specs, both 1D post-stratification by organisation:

    rake_employer_salaried (primary)  each organisation's staff number shared over the adverts
        that STATE a salary, so £ totals cover the whole organisation (scaled up from the
        salaried adverts, not from all adverts). FTE, exposure and quadrant shares are then also
        on the salaried adverts.
    rake_employer  shared over every usable advert. Adverts without a salary count in FTE and
        exposure but add £0, so £ totals cover only the salaried share of staff.
    """
    cfg = sector.config["weighting"]
    tgt = target_table(sector).set_index("cell")
    pool = sector.min_tasks_sample(adverts, n_tasks)
    missing = set(pool["org_unit"]) - set(tgt.index)
    if missing:
        raise ValueError(f"adverts in cells with no target: {sorted(missing)}")
    small = int(cfg["small_cell_adverts"])
    primary = cfg["primary_spec"]
    pools = {SPEC: pool, SALARIED: pool[pool["salary_annual_gbp"].notna()]}
    frames, diags = [], {}
    for spec, p in pools.items():
        w, cells = _cell_weights(p, tgt)
        frames.append(
            pd.DataFrame(
                {
                    "vacancy_id": adverts["vacancy_id"],
                    "weight_spec": spec,
                    "weight": adverts["vacancy_id"].map(w).astype("float64"),
                    "is_primary": spec == primary,
                    "target_source": cfg["target_source"],
                    "target_date": cfg["target_date"],
                }
            )
        )
        diags[spec] = _diagnostics(spec, w, cells, tgt, small)
    diagnostics = {
        "method": "1D post-stratification by organisation (BBC split public service / commercial)",
        "primary_spec": primary,
        **diags[primary],
        "specs": diags,
    }
    targets = pd.concat(
        [
            pd.DataFrame(
                {
                    "spec": spec,
                    "dimension": "org_unit",
                    "category": tgt.index,
                    "target_fte": tgt["target"].to_numpy(),
                    "measure": tgt["measure"].to_numpy(),
                    "source": tgt["source"].to_numpy(),
                }
            )
            for spec in pools
        ],
        ignore_index=True,
    )
    return WeightOutput(pd.concat(frames, ignore_index=True), targets, diagnostics)
