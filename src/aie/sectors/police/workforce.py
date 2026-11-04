"""Raking targets from the Home Office "Police workforce, England and Wales" open data table,
and the police rake.

Source: Home Office, "Police workforce, England and Wales" open data tables
(https://www.gov.uk/government/statistics/police-workforce-open-data-tables). The workforce
table has one row per year (as at 31 March) x force x sex x rank x worker type, with headcount
and FTE. We use the latest year. The exact file, date and publication are in config.toml
[weighting] and docs/police_workforce.md.

Scope: the 43 territorial forces of England and Wales plus British Transport Police. Police
Scotland, PSNI, the Civil Nuclear Constabulary and MoD Police are not in it, so their adverts
stay unweighted.

The rake (rake_force_staff): 1D post-stratification to FTE of POLICE STAFF (including
designated officers, who are staff in designated roles) and PCSOs. Police OFFICERS are left
out of the weighted population:
  * officers are 63% of police FTE, but officer recruitment runs through national entry
    routes (PCDA, DHEP), so force job boards carry few officer adverts, and those are senior
    or specialist posts (deputy and assistant chief constables, regional organised crime
    units) that do not stand for 115,000 constables;
  * only about 12% of officer adverts give a salary.
Special constables and police support volunteers have no FTE (unpaid), and volunteer or cadet
leader adverts classified as staff are dropped for the same reason.

Cells:
  * police staff: a force (unit) with at least `min_unit_adverts` eligible staff adverts is its
    own cell; every other force, including those with no adverts, is pooled by region, then
    nationally if the region is still thin;
  * PCSOs: one national cell (fewer than 100 adverts, mostly Thames Valley and Sussex).
Surrey, Sussex and their joint "Surrey & Sussex Police Collaboration" adverts form one unit,
as do Norfolk, Suffolk and their joint adverts, so the shared staff are not counted twice.

Known limitations: the weighted population covers about 37% of police FTE (staff and PCSOs),
not the whole workforce. Thames Valley supplies about 45% of the adverts, which its own cell
contains, but staff adverts in every force skew to professional posts, as in local
government.
"""

from __future__ import annotations

import io

import pandas as pd

from ... import storage
from ...methods import rake
from ..base import WeightOutput

SHEET = "Data"
COLUMNS = {
    "As at 31 March": "year",
    "Force name": "force",
    "Region": "region",
    "Worker type": "worker_type",
    "Total (headcount)": "headcount",
    "Total (FTE)": "fte",
}
# Home Office worker types -> the adverts' occupation_group. Officers, specials and volunteers
# are not raked (see the module docstring).
WORKER_TYPES = {
    "Police Staff": "Police staff",
    "Designated Officer": "Police staff",
    "Police Community Support Officer": "PCSO",
}
RAKED_GROUPS = ("Police staff", "PCSO")
# Advert force names (cleaning repo) that differ from the Home Office names.
ALIASES = {
    "Avon & Somerset": "Avon and Somerset",
    "Devon & Cornwall": "Devon and Cornwall",
    "Dyfed Powys": "Dyfed-Powys",
    "Hampshire": "Hampshire and Isle of Wight",
    "City of London": "London, City of",
    "Metropolitan": "Metropolitan Police",
}
# Forces that share staff or advertise jointly are one raking unit.
UNITS = {
    "Surrey": "Surrey and Sussex",
    "Sussex": "Surrey and Sussex",
    "Surrey & Sussex Police Collaboration": "Surrey and Sussex",
    "Norfolk": "Norfolk and Suffolk",
    "Suffolk": "Norfolk and Suffolk",
    "Norfolk and Suffolk Constabularies": "Norfolk and Suffolk",
}
NATIONAL = "National"  # BTP has no region in the table ([z])
UNPAID_TITLE = r"\bvolunteer|\bcadet"


def load(path: str) -> pd.DataFrame:
    """Latest year of the workforce table: one row per force x worker type, FTE + headcount."""
    raw = pd.read_excel(io.BytesIO(storage.read_bytes(path)), sheet_name=SHEET, engine="odf")
    t = raw.rename(columns=COLUMNS)[list(COLUMNS.values())]
    t = t[t["year"] == t["year"].max()].copy()
    for c in ("fte", "headcount"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)  # [z] = not applicable
    t["region"] = t["region"].where(~t["region"].astype(str).str.startswith("["), NATIONAL)
    return t.groupby(["year", "force", "region", "worker_type"], as_index=False)[
        ["fte", "headcount"]
    ].sum()


def unit_of(force: pd.Series) -> pd.Series:
    """Advert or Home Office force name -> raking unit."""
    f = force.astype("string").replace(ALIASES)
    return f.replace(UNITS)


def unit_targets(t: pd.DataFrame) -> pd.DataFrame:
    """unit x occupation_group targets for the raked worker types."""
    t = t.assign(
        unit=unit_of(t["force"]),
        occupation_group=t["worker_type"].map(WORKER_TYPES),
    ).dropna(subset=["occupation_group"])
    return t.groupby(["unit", "occupation_group"], as_index=False).agg(
        region=("region", "first"), fte=("fte", "sum"), headcount=("headcount", "sum")
    )


def strata(ut: pd.DataFrame, min_adverts: int) -> pd.Series:
    """Cell per unit x occupation group (see the module docstring). `ut` has unit,
    occupation_group, region, fte, n_adverts."""
    staff = ut["occupation_group"] == "Police staff"
    own = staff & (ut["n_adverts"] >= min_adverts) & (ut["fte"] > 0)
    cell = pd.Series(("Police staff | " + ut["unit"]).where(own), index=ut.index, dtype=object)
    region = "Police staff | " + ut["region"] + " (pooled)"
    cell = cell.fillna(region.where(staff))
    pooled_n = ut["n_adverts"].groupby(cell).transform("sum")
    thin = staff & ~own & (pooled_n < min_adverts)
    cell = cell.mask(thin, "Police staff | national (pooled)")
    return cell.fillna("PCSO | national (pooled)")


def rake_to_workforce(sector, adverts: pd.DataFrame, n_tasks, release: str) -> WeightOutput:
    cfg = sector.config["weighting"]
    spec = cfg["primary_spec"]
    path = sector.reference(cfg["reference_file"], release)
    t = load(path)
    ut = unit_targets(t)

    sample = sector.min_tasks_sample(adverts, n_tasks)
    sample["unit"] = unit_of(sample["org_unit"])
    unpaid = sample["job_title"].fillna("").str.contains(UNPAID_TITLE, case=False, regex=True)
    raked = (
        sample["occupation_group"].isin(RAKED_GROUPS) & sample["unit"].isin(ut["unit"]) & ~unpaid
    )
    donors = sample[raked]
    counts = donors.groupby(["unit", "occupation_group"]).size().rename("n_adverts")
    ut = ut.merge(counts, on=["unit", "occupation_group"], how="left").fillna({"n_adverts": 0})
    ut["cell"] = strata(ut, cfg["min_unit_adverts"])

    targets = ut.groupby("cell")["fte"].sum()
    with_donors = set(ut.loc[ut["n_adverts"] > 0, "cell"])
    lost = targets[~targets.index.isin(with_donors)]
    targets = targets[targets.index.isin(with_donors)]
    cell_of = ut.set_index(["unit", "occupation_group"])["cell"]
    donors = donors.assign(
        cell=[cell_of.get((u, g)) for u, g in zip(donors["unit"], donors["occupation_group"])]
    )
    res = rake.rake(
        donors[["vacancy_id", "cell"]].reset_index(drop=True),
        {"cell": targets.to_dict()},
        target_fte=float(targets.sum()),
        max_iter=cfg["ipf_max_iter"],
        tol=cfg["ipf_tol"],
    )
    w = res.weights.set_index("vacancy_id")["weight"]

    total = t.groupby("worker_type")["fte"].sum()
    not_raked = sample[~raked]
    reasons = pd.Series("raked", index=sample.index)
    reasons = reasons.mask(
        ~sample["occupation_group"].isin(RAKED_GROUPS),
        "worker type not raked: " + sample["occupation_group"].astype(str),
    )
    reasons = reasons.mask(
        sample["occupation_group"].isin(RAKED_GROUPS) & unpaid, "volunteer or cadet advert (unpaid)"
    )
    reasons = reasons.mask(sample["occupation_group"].isin(RAKED_GROUPS) & ~unpaid
                           & ~sample["unit"].isin(ut["unit"]),
                           "force not in the Home Office table")  # fmt: skip
    salaried = donors["salary_annual_gbp"].notna()
    coverage = {
        "as_at": f"31 March {int(t['year'].max())}",
        "workforce_fte_by_worker_type": {str(k): float(v) for k, v in total.items()},
        "workforce_fte_total": float(total.sum()),
        "raked_fte": float(w.sum()),
        "raked_share_of_workforce_fte": float(w.sum() / total.sum()),
        "not_represented": {str(k): float(v) for k, v in lost.items()},
        "eligible_adverts": len(sample),
        "adverts_weighted": int(w.notna().sum()),
        "adverts_by_reason": {str(k): int(v) for k, v in reasons.value_counts().items()},
        "weighted_share_with_salary": float(
            w.reindex(donors.loc[salaried, "vacancy_id"]).sum() / w.sum()
        ),
        "units_own_cell": int((~ut["cell"].str.endswith("(pooled)")).sum()),
        "largest_unit_share_of_adverts": {
            str(k): float(v)
            for k, v in (donors["unit"].value_counts(normalize=True).head(3)).items()
        },
        "cells": {
            c: {"units": int(len(g)), "adverts": int(g["n_adverts"].sum()),
                "fte": float(g["fte"].sum())}
            for c, g in ut.groupby("cell")
        },
        "not_raked_forces_top": {
            str(k): int(v)
            for k, v in not_raked.loc[~not_raked["unit"].isin(ut["unit"]), "org_unit"]
            .value_counts().head(10).items()
        },
    }  # fmt: skip
    weights = adverts[["vacancy_id"]].assign(
        weight=adverts["vacancy_id"].map(w),
        weight_spec=spec,
        is_primary=True,
        target_source=cfg["target_source"],
        target_date=cfg["target_date"],
    )
    return WeightOutput(
        weights=weights,
        targets=res.targets.assign(weight_spec=spec),
        diagnostics={
            spec: res.diagnostics,
            "coverage": coverage,
            "units": ut.to_dict("records"),
        },  # fmt: skip
        inputs=[path],
    )
