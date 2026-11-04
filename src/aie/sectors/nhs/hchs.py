"""Raking targets from NHS Digital's HCHS core tables, and the NHS rake.

Ported from gds-idea-nhs-ai-exposure-raking (targets.py, sample.py, build_weights.py).
HCHS core tables carry no disclosure suppression at the levels used, so every target is a real
FTE. Our 7 staff groups sum to ~188 FTE below the published total ("Unknown classification").
"""

from __future__ import annotations

import pandas as pd

from ...methods import rake
from ..base import WeightOutput

TARGET_STAFF_GROUPS = [
    "HCHS doctors",
    "Nurses & health visitors",
    "Midwives",
    "Ambulance staff",
    "Scientific, therapeutic & technical",
    "Support to clinical staff",
    "NHS infrastructure support",
]
NHSE_REGIONS = [
    "East of England",
    "London",
    "Midlands",
    "North East and Yorkshire",
    "North West",
    "South East",
    "South West",
]
HCHS_FOUNDATION_GRADES = ["Foundation Doctor Year 1", "Foundation Doctor Year 2"]
NON_BAND_GRADES_IN_SAMPLE = ["Very Senior Manager"]


def _hchs_staff_group(main: str, sub: str) -> str | None:
    """Map an HCHS (MAIN_STAFF_GROUP, STAFF_GROUP_1) pair to one of our 7 groups."""
    if main == "Professionally qualified clinical staff":
        if sub == "HCHS doctors - All grades":
            return "HCHS doctors"
        if sub == "Scientific, therapeutic & technical staff":
            return "Scientific, therapeutic & technical"
        if sub in ("Nurses & health visitors", "Midwives", "Ambulance staff"):
            return sub
        return None  # per-grade doctor rows (would double count)
    if main in ("Support to clinical staff", "NHS infrastructure support") and (
        sub == "All staff groups"
    ):
        return main
    return None


def _load(sector, name: str, data_month: str, release: str) -> pd.DataFrame:
    from ... import storage

    df = storage.read_csv(sector.reference(name, release), dtype=str)
    df["FTE"] = pd.to_numeric(df["FTE"], errors="coerce")
    return df[df["DATA_MONTH"] == data_month].copy()


def margins(sector, sample_grades: set[str], release: str) -> dict[str, dict[str, float]]:
    month = sector.config["weighting"]["target_date"]
    core1 = _load(sector, "hchs_core1_may2026.csv", month, release)
    core6 = _load(sector, "hchs_core6_may2026.csv", month, release)

    nat = core1[core1["DATA_LEVEL"] == "National"].copy()
    nat["grp"] = [
        _hchs_staff_group(m, s) for m, s in zip(nat["MAIN_STAFF_GROUP"], nat["STAFF_GROUP_1"])
    ]
    staff_group = (
        nat.dropna(subset=["grp"]).groupby("grp")["FTE"].sum().reindex(TARGET_STAFF_GROUPS)
    )

    reg = core1[
        (core1["DATA_LEVEL"] == "NHS England region")
        & (core1["MAIN_STAFF_GROUP"] == "All staff groups")
        & (core1["STAFF_GROUP_1"] == "All staff groups")
    ]
    region = reg.groupby("NHSE_REGION_NAME")["FTE"].sum().reindex(NHSE_REGIONS)

    # AfC bands sit in GRADE on the "All staff groups" rows; the medical grade split sits in
    # STAFF_GROUP_1 ("HCHS doctors - Consultant", ...). Grades the sample cannot distinguish
    # fold into "Other HCHS Doctor Grades" so target and sample share one taxonomy.
    non_medical = core6[
        (core6["MAIN_STAFF_GROUP"] == "All staff groups")
        & (core6["GRADE"].str.startswith("Band ") | core6["GRADE"].isin(NON_BAND_GRADES_IN_SAMPLE))
    ]
    med = core6[
        core6["STAFF_GROUP_1"].str.startswith("HCHS doctors - ")
        & (core6["STAFF_GROUP_1"] != "HCHS doctors - All grades")
    ].copy()
    med["GRADE"] = med["STAFF_GROUP_1"].str.replace("HCHS doctors - ", "", regex=False)
    med["GRADE"] = med["GRADE"].replace({g: "Foundation Doctor" for g in HCHS_FOUNDATION_GRADES})
    med.loc[~med["GRADE"].isin(sample_grades), "GRADE"] = "Other HCHS Doctor Grades"
    grade = (
        pd.concat([non_medical.groupby("GRADE")["FTE"].sum(), med.groupby("GRADE")["FTE"].sum()])
        .groupby(level=0)
        .sum()
    )
    return {
        "staff_group": staff_group.to_dict(),
        "grade": grade.to_dict(),
        "region": region.to_dict(),
    }


def rake_to_hchs(sector, adverts: pd.DataFrame, n_tasks: pd.Series, release: str) -> WeightOutput:
    from .sector import BAND_TO_GRADE

    sample = sector.min_tasks_sample(adverts, n_tasks).rename(
        columns={"occupation_group": "staff_group"}
    )
    sample_grades = set(BAND_TO_GRADE.values()) | {"Very Senior Manager"}
    m = margins(sector, sample_grades, release)
    target_fte = sum(v for v in m["staff_group"].values() if not pd.isna(v))
    cfg = sector.config["weighting"]
    res = rake.rake(
        sample[["vacancy_id", "staff_group", "grade", "region"]],
        m,
        target_fte=target_fte,
        max_iter=cfg["ipf_max_iter"],
        tol=cfg["ipf_tol"],
    )
    weights = adverts[["vacancy_id"]].merge(res.weights, on="vacancy_id", how="left")
    weights = weights.assign(
        weight_spec="rake_1d",
        is_primary=True,
        target_source=cfg["target_source"],
        target_date=cfg["target_date"],
    )
    return WeightOutput(
        weights=weights,
        targets=res.targets.assign(weight_spec="rake_1d"),
        diagnostics={"rake_1d": res.diagnostics},
    )
