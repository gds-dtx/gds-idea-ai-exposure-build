"""NHS salary: current-year NHS Digital Staff Earnings Estimates (primary), and the advert's own
salary uprated by the AfC pay index (sensitivity column only).

Ported from gds-idea-nhs-ai-exposure-analysis (nhs_salary.py, salary.py). Monthly mean per
FTE, x12. Tiers, most specific first:
  * HCHS doctor grades: exact staff_group x grade BASIC PAY row.
  * Very Senior Manager (band is blank; identified by pay_scheme_type): "Senior managers"
    BASIC PAY, as a proxy.
  * AfC bands: staff_group x band EARNINGS (NHS Digital publishes no basic-pay-by-band table,
    so this tier includes overtime and allowances), then staff_group BASIC PAY, then
    "All staff" BASIC PAY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ... import storage

BASIC_PAY = "PUBGRP_010_BASIC_PAY_PER_FTE"
EARNINGS = "PUBGRP_020_EARNINGS"

DOCTOR_GRADE_ROWS: dict[str, list[str]] = {
    "CONSULTANT": ["HCHS doctors - Consultant"],
    "SPECIALTY_DOCTOR": ["HCHS doctors - Specialty Doctor"],
    "SPECIALTY_REGISTRAR": ["HCHS doctors - Specialty Registrar"],
    "FOUNDATION_DOCTOR": [
        "HCHS doctors - Foundation Doctor Year 1",
        "HCHS doctors - Foundation Doctor Year 2",
    ],
    "SPECIALIST": ["HCHS doctors - Associate Specialist"],
    "DOCTOR_OTHER": [
        "HCHS doctors - Core Training",
        "HCHS doctors - Staff Grade",
        "HCHS doctors - Hospital Practitioner / Clinical Assistant",
        "HCHS doctors - Other HCHS doctor grades",
    ],
}
VSM_PROXY_ROW = "Senior managers"
AFC_BAND_LABEL = {
    f"BAND_{b}": f"Band {b.lower() if len(b) > 1 else b}"
    for b in ["1", "2", "3", "4", "5", "6", "7", "8A", "8B", "8C", "8D", "9"]
}
BAND_FILE_SUBGROUPS: dict[str, list[str]] = {
    "Nurses & health visitors": ["Nurses & health visitors"],
    "Midwives": ["Midwives"],
    "Ambulance staff": ["Ambulance staff"],
    "Scientific, therapeutic & technical": ["Scientific, therapeutic & technical staff"],
    "Support to clinical staff": [
        "Support to ST&T staff",
        "Support to ambulance staff",
        "Support to doctors, nurses & midwives",
    ],
    "NHS infrastructure support": ["Central functions", "Hotel, property & estates", "Managers"],
}

# AfC headline consolidated awards (England), for the uprated-advert sensitivity only.
AFC_PAY_AWARDS = {2021: 0.030, 2022: 0.048, 2023: 0.050, 2024: 0.055, 2025: 0.036, 2026: 0.0}


def _wmean(df: pd.DataFrame) -> tuple[float, float]:
    w = df["SAMPLE_SIZE"].astype(float)
    if w.sum() == 0:
        return np.nan, 0.0
    return float((df["AMOUNT"] * w).sum() / w.sum()), float(w.sum())


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["DATE"] == df["DATE"].max()].copy()


def build_lookup(sector, release: str) -> pd.DataFrame:
    """One row per (staff_group, band, pay_scheme_type) with estimated_salary + salary_source."""
    min_n = sector.config["salary"]["min_cell_sample"]
    est = _latest(
        storage.read_csv(
            sector.reference("earnings_estimates_jul2026.csv", release), dtype={"STAFF_GROUP": str}
        )
    )
    est = est[est["PAYMENT_TYPE"] == BASIC_PAY]
    sg_pay = {}
    for grp, g in est.groupby("STAFF_GROUP"):
        sal, n = _wmean(g)
        sg_pay[grp] = (sal * 12, n)

    band = _latest(
        storage.read_csv(
            sector.reference("earnings_by_band_jul2026.csv", release),
            dtype={"STAFF_GROUP": str, "AFC_BAND": str},
        )
    )
    band = band[
        (band["PAYMENT_TYPE"] == EARNINGS)
        & ~band["AFC_BAND"].isin(["All AfC bands", "Non AfC band"])
        & (band["STAFF_GROUP"] != "All staff groups")
    ]
    label_to_code = {v: k for k, v in AFC_BAND_LABEL.items()}

    rows = []
    for grade, src in DOCTOR_GRADE_ROWS.items():
        cells = [sg_pay[s] for s in src if s in sg_pay]
        n = sum(c[1] for c in cells)
        sal = np.nan if n == 0 else sum(c[0] * c[1] for c in cells) / n
        rows.append(("HCHS doctors", grade, pd.NA, sal, "doctor_grade_basic_pay"))
    rows.append(
        (
            pd.NA,
            pd.NA,
            "VERY_SENIOR_MANAGER",
            sg_pay[VSM_PROXY_ROW][0],
            "senior_manager_basic_pay_proxy",
        )
    )
    for target, src_labels in BAND_FILE_SUBGROUPS.items():
        sub = band[band["STAFF_GROUP"].isin(src_labels)]
        cells = {}
        for label, cell in sub.groupby("AFC_BAND"):
            if label in label_to_code:
                sal, n = _wmean(cell)
                cells[label_to_code[label]] = (sal * 12, n)
        tier2 = sg_pay.get(target, (np.nan, 0))[0]
        for code in AFC_BAND_LABEL:
            if code in cells and cells[code][1] >= min_n:
                rows.append((target, code, pd.NA, cells[code][0], "staff_group_x_band_earnings"))
            elif pd.notna(tier2):
                rows.append((target, code, pd.NA, tier2, "staff_group_basic_pay"))
            else:
                rows.append(
                    (target, code, pd.NA, sg_pay["All staff"][0], "all_staff_basic_pay_fallback")
                )
    return pd.DataFrame(
        rows,
        columns=["staff_group", "band", "pay_scheme_type", "estimated_salary", "salary_source"],
    )


def attach(v: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """estimated_salary + salary_source aligned to `v` (staff_group, pay_scheme_band,
    pay_scheme_type). VSM matches on pay_scheme_type alone."""
    rest = lookup[lookup["pay_scheme_type"].isna()].drop(columns="pay_scheme_type")
    out = (
        v.rename(columns={"pay_scheme_band": "band"})
        .reset_index()
        .merge(rest, on=["staff_group", "band"], how="left")
        .set_index("index")
        .reindex(v.index)
    )
    vsm = lookup[lookup["pay_scheme_type"] == "VERY_SENIOR_MANAGER"].iloc[0]
    is_vsm = v["pay_scheme_type"] == "VERY_SENIOR_MANAGER"
    out.loc[is_vsm, "estimated_salary"] = vsm["estimated_salary"]
    out.loc[is_vsm, "salary_source"] = vsm["salary_source"]
    return out[["estimated_salary", "salary_source"]]


def advert_annual_salary(v: pd.DataFrame) -> pd.Series:
    """Advert basic salary: fixed, else range midpoint; hourly x 37.5 x 52.14; keep £12k-£250k."""
    fixed = pd.to_numeric(v["fixed_salary"], errors="coerce")
    lo = pd.to_numeric(v["salary_range_min"], errors="coerce")
    hi = pd.to_numeric(v["salary_range_max"], errors="coerce")
    sal = fixed.where(fixed.notna(), (lo + hi) / 2)
    hourly = v["salary_time_unit"].eq("an hour")
    sal = sal.where(~hourly, sal * 37.5 * 52.14)
    sal = sal.where(v["salary_time_unit"].isin(["a year", "an hour"]))
    return sal.where((sal >= 12000) & (sal <= 250000))


def uprated_advert_salary(v: pd.DataFrame, base_year: int) -> pd.Series:
    factor = {}
    for y in AFC_PAY_AWARDS:
        f = 1.0
        for yy in range(y + 1, base_year + 1):
            f *= 1 + AFC_PAY_AWARDS.get(yy, 0.0)
        factor[y] = f
    return advert_annual_salary(v) * v["year"].map(factor)
