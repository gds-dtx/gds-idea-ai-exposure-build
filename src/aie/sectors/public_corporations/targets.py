"""Raking targets for public corporations, and the rake.

There is no single official table of staff by public corporation. ONS Public Sector
Employment gives only the total for all public corporations, and not for the NAO, Network
Rail, National Highways or TfL, which ONS classifies elsewhere. So each body's own published
staff number is the target: its annual report and accounts for 2024/25 (British Steel
publishes no accounts as a public body yet; its figure is the 2021 direct headcount in the
Oxford Economics report it published in 2026). Each figure, what it counts and where it is
printed are in config.toml [targets] and docs/public_corporations.md; the PDFs are copied
into the release's reference/ folder.

The rake (rake_employer): 1D post-stratification. One cell per body (the BBC as two cells,
public service and commercial, matching its accounts). Every eligible advert in a cell gets
target / n_adverts. Cells are NOT pooled across bodies, because the bodies do unrelated work;
cells with fewer than `small_cell_adverts` adverts are flagged in the diagnostics instead
(Ordnance Survey has under 10), and their figures should be quoted with that caveat or left
out.

Measures differ: most targets are average FTE, but the Bank of England and the CAA publish
average headcount and British Steel a headcount, so their weights are slightly overstated in
FTE terms.
"""

from __future__ import annotations

import pandas as pd

from ..base import Sector, WeightOutput

SPEC = "rake_employer"


def target_table(sector: Sector) -> pd.DataFrame:
    rows = [
        {"cell": cell, "target": float(t["value"]), "measure": t["measure"], "source": t["source"]}
        for cell, t in sector.config["targets"].items()
    ]
    return pd.DataFrame(rows)


def rake_to_employers(sector: Sector, adverts: pd.DataFrame, n_tasks: pd.Series) -> WeightOutput:
    cfg = sector.config["weighting"]
    tgt = target_table(sector).set_index("cell")
    pool = sector.min_tasks_sample(adverts, n_tasks)
    missing = set(pool["org_unit"]) - set(tgt.index)
    if missing:
        raise ValueError(f"adverts in cells with no target: {sorted(missing)}")
    n = pool.groupby("org_unit").size().reindex(tgt.index, fill_value=0)
    per_advert = (tgt["target"] / n).where(n > 0)
    w = pool.set_index("vacancy_id")["org_unit"].map(per_advert)

    weights = pd.DataFrame(
        {
            "vacancy_id": adverts["vacancy_id"],
            "weight_spec": SPEC,
            "weight": adverts["vacancy_id"].map(w).astype("float64"),
            "is_primary": True,
            "target_source": cfg["target_source"],
            "target_date": cfg["target_date"],
        }
    )
    small = int(cfg["small_cell_adverts"])
    cells = pd.DataFrame(
        {"target": tgt["target"], "n_adverts": n, "weight_per_advert": per_advert}
    ).reset_index(names="cell")
    cells["small_cell"] = cells["n_adverts"] < small
    covered = cells.loc[cells["n_adverts"] > 0, "target"].sum()
    ww = weights["weight"].dropna()
    diagnostics = {
        "spec": SPEC,
        "method": "1D post-stratification by body (BBC split public service / commercial)",
        "n_weighted": int(ww.size),
        "weighted_total": float(ww.sum()),
        "target_total": float(tgt["target"].sum()),
        "target_covered": float(covered),
        "design_effect": float(1 + ww.var(ddof=0) / ww.mean() ** 2) if ww.size else None,
        "max_weight": float(ww.max()) if ww.size else None,
        "small_cell_threshold": small,
        "small_cells": cells.loc[cells["small_cell"], "cell"].tolist(),
        "empty_cells": cells.loc[cells["n_adverts"] == 0, "cell"].tolist(),
        "cells": cells.to_dict(orient="records"),
    }
    targets = pd.DataFrame(
        {
            "spec": SPEC,
            "dimension": "org_unit",
            "category": tgt.index,
            "target_fte": tgt["target"].to_numpy(),
            "measure": tgt["measure"].to_numpy(),
            "source": tgt["source"].to_numpy(),
        }
    )
    return WeightOutput(weights, targets, diagnostics)
