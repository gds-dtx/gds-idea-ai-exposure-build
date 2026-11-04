"""The cut request: everything a front end needs to send, as one JSON-serialisable object.

The dashboard page builds a CutSpec from its dropdowns, calls `run_cut(spec)`, shows
`result.data`, and offers `result.to_csv()` + `result.meta` as the download. The same spec
saved as JSON reproduces the same numbers later (it records release and methodology version).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Columns any cut can group or filter by. One-sector cuts can also use that sector's own
# vacancies columns (options()["sectors"][s]["sector_dimensions"], e.g. is_school_role).
CORE_DIMENSIONS = [
    "sector",
    "year",
    "employer",
    "org_unit",
    "occupation_group",
    "grade",
    "region",
    "salary_source",
    "quadrant",
]

MEASURES = {
    "n_adverts": "Adverts in the weighted population",
    "fte": "Sum of weights (FTE; advert count for unweighted sectors)",
    "fte_with_salary": "FTE that has a salary, so is included in the £ measures",
    "pct_fte": "Share of the cut's total FTE",
    "mean_exposure": "Weighted mean decay-weighted exposure",
    "paybill_gbp": "Weight x salary",
    "opportunity_gbp": "Weight x salary x mean exposure (every role)",
    "cashable_gbp": "Full salary of Displacement-type roles",
    "cashable_share_gbp": "Salary x exposure of Displacement-type roles (report beside cashable)",
    "non_cashable_gbp": "Salary x exposure of Augmentation-type + Concentrated roles",
    "opportunity_pct_of_paybill": "Opportunity / paybill",
    "opportunity_per_fte_gbp": "Opportunity / FTE with a salary",
}
DEFAULT_MEASURES = [
    "n_adverts", "fte", "mean_exposure", "paybill_gbp", "opportunity_gbp",
    "cashable_gbp", "non_cashable_gbp",
]  # fmt: skip


# Columns of a vacancy-level cut (level="vacancy"): one row per advert, its own figures.
# One-sector cuts can also ask for that sector's own vacancies columns.
VACANCY_COLUMNS = {
    "sector": "Sector",
    "vacancy_id": "Advert identifier",
    "job_title": "Job title as advertised",
    "posted_date": "Date the advert was posted",
    "year": "Year the advert was posted",
    "employer": "Employer",
    "org_unit": "Organisation (department, trust, council, force)",
    "occupation_group": "Occupation group (profession, staff group)",
    "grade": "Grade",
    "region": "Region",
    "salary_annual_gbp": "Stated annual salary (blank when the advert gave none)",
    "salary_source": "Where the salary came from",
    "fte": "Weight: the FTE this advert stands for (1 for unweighted sectors)",
    "n_tasks": "Tasks extracted from the advert",
    "mean_exposure": "Decay-weighted mean exposure of the advert's tasks",
    "var_exposure": "Decay-weighted variance of the advert's task exposures",
    "quadrant": "Quadrant under the chosen limits",
    "opportunity_per_fte_gbp": "Salary x mean exposure",
    "cashable_per_fte_gbp": "Full salary if Displacement-type, else 0",
    "non_cashable_per_fte_gbp": "Salary x exposure if Augmentation-type or Concentrated, else 0",
}
DEFAULT_VACANCY_COLUMNS = [
    "vacancy_id", "job_title", "year", "org_unit", "occupation_group", "grade", "region",
    "salary_annual_gbp", "fte", "mean_exposure", "quadrant",
    "cashable_per_fte_gbp", "non_cashable_per_fte_gbp",
]  # fmt: skip


class Filter(BaseModel):
    column: str
    op: Literal["in", "not_in", "eq", "between", "is_null", "not_null"] = "in"
    values: list = Field(default_factory=list)


class CustomThresholds(BaseModel):
    """Custom quadrant limits: the one sensitivity the dashboard offers.

    Give percentiles (what the page's sliders set) OR the three values. Percentiles are read
    off the same distribution as the fixed thresholds (CS's ensemble roles, primary weights),
    so 25 / 25 reproduces the fixed thresholds, and the result applies to every sector:
      mean_top_pct  the "high" stratum is the top X% of roles by mean exposure
      var_pct       Displacement-type = bottom X% by variance within the high stratum;
                    Concentrated      = top X% by variance within the low stratum
    """

    mean_top_pct: float | None = Field(None, ge=5, le=50)
    var_pct: float | None = Field(None, ge=5, le=50)
    mean_hi: float | None = Field(None, ge=0, le=1)
    high_var_lo: float | None = Field(None, ge=0, le=0.25)
    low_var_hi: float | None = Field(None, ge=0, le=0.25)

    @model_validator(mode="after")
    def _one_mode(self):
        pcts = (self.mean_top_pct, self.var_pct)
        vals = (self.mean_hi, self.high_var_lo, self.low_var_hi)
        if all(v is not None for v in pcts) and all(v is None for v in vals):
            return self
        if all(v is not None for v in vals) and all(v is None for v in pcts):
            return self
        raise ValueError("give mean_top_pct + var_pct, or mean_hi + high_var_lo + low_var_hi")

    @property
    def by_percentile(self) -> bool:
        return self.mean_top_pct is not None


class CutSpec(BaseModel):
    sectors: list[str]
    # "group": one row per group_by combination (the default). "vacancy": one row per advert
    # passing the filters, with `columns`; group_by, measures and min_cell_adverts are unused.
    level: Literal["group", "vacancy"] = "group"
    columns: list[str] = Field(default_factory=lambda: list(DEFAULT_VACANCY_COLUMNS))
    # "latest" or {sector: release}
    release: str | dict[str, str] = "latest"
    group_by: list[str] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=lambda: list(DEFAULT_MEASURES))
    # None: the sector's own primary model for a one-sector cut; the cross-sector model otherwise
    score_model: str | None = None
    # "primary" or a named spec (rake_1d, rake_3d, unweighted)
    weight_spec: str = "primary"
    # fixed (primary), the sector's own quartiles, or `custom_thresholds` (a sensitivity)
    threshold_basis: Literal["fixed", "sector_quartiles", "custom"] = "fixed"
    custom_thresholds: CustomThresholds | None = None
    # Only the current methodology is published; a past one is recomputed with `aie metrics`.
    methodology_version: str | None = None
    # Groups with fewer adverts than this have their measures blanked (not dropped).
    min_cell_adverts: int = 10
    label: str | None = None

    @field_validator("sectors")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("choose at least one sector")
        return v

    @field_validator("measures")
    @classmethod
    def _known_measures(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(MEASURES)
        if unknown:
            raise ValueError(f"unknown measures: {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def _custom_needs_limits(self):
        if (self.threshold_basis == "custom") != (self.custom_thresholds is not None):
            raise ValueError('custom_thresholds goes with threshold_basis="custom" (and only then)')
        return self

    @property
    def cross_sector(self) -> bool:
        return len(self.sectors) > 1
