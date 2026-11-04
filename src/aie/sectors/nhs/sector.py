"""NHS: HCHS workforce in England, from NHS Jobs / TRAC adverts.

Mapping onto the contract:
  employer          employer_name (trust / practice)
  org_unit          employer_name (`organisation` is ~0.1% populated)
  occupation_group  staff_group (the 7 HCHS groups, assigned in the old cleaning repo)
  grade             HCHS grade label from pay_scheme_band (Band 1-9, doctor grades, VSM)
  region            NHS England region, from employer postcode area
  salary            NHS Digital Staff Earnings Estimates lookup (see salary.py)

In scope: rake years 2021-2026 and an England postcode (HCHS is England-only; the feed is
UK-wide). Raked 1D to HCHS May-2026 on staff_group, grade and region.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ... import storage
from ..base import CleanOutput, ScoreOutput, Sector, WeightOutput, no_funnel
from . import hchs, salary

BAND_TO_GRADE = {
    "BAND_1": "Band 1",
    "BAND_2": "Band 2",
    "BAND_3": "Band 3",
    "BAND_4": "Band 4",
    "BAND_5": "Band 5",
    "BAND_6": "Band 6",
    "BAND_7": "Band 7",
    "BAND_8A": "Band 8a",
    "BAND_8B": "Band 8b",
    "BAND_8C": "Band 8c",
    "BAND_8D": "Band 8d",
    "BAND_9": "Band 9",
    "CONSULTANT": "Consultant",
    "SPECIALTY_DOCTOR": "Specialty Doctor",
    "SPECIALTY_REGISTRAR": "Specialty Registrar",
    "FOUNDATION_DOCTOR": "Foundation Doctor",
    "SPECIALIST": "Associate Specialist",
    "DOCTOR_OTHER": "Other HCHS Doctor Grades",
}
VSM = "VERY_SENIOR_MANAGER"

# Wales, Scotland, Northern Ireland, Channel Islands, Isle of Man. SY (Shropshire/Powys) is
# kept: its NHS England adverts are Shrewsbury/Telford.
NON_ENGLAND_POSTCODE_AREAS = frozenset(
    "CF LD LL NP SA AB DD DG EH FK G HS IV KA KW KY ML PA PH TD ZE BT GY JE IM".split()
)

VACANCY_COLS = [
    "vacancy_id", "id", "job_title", "employer_name", "employer_address_postcode",
    "created_date", "closing_date", "year", "staff_group", "pay_scheme_type", "pay_scheme_band",
    "contract_type", "full_time", "part_time", "fixed_salary", "salary_range_min",
    "salary_range_max", "salary_time_unit", "job_text",
]  # fmt: skip


def postcode_area(postcode: pd.Series) -> pd.Series:
    return (
        postcode.astype("string")
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.extract(r"^([A-Z]{1,2})")[0]
    )


class NHS(Sector):
    slug = "nhs"
    package_dir = Path(__file__).parent

    def clean(self, release: str) -> CleanOutput:
        src = self.legacy("vacancies")
        v = storage.read_parquet(src, columns=VACANCY_COLS)

        area = postcode_area(v["employer_address_postcode"])
        lk = storage.read_csv(
            self.reference("postcode_area_to_nhse_region.csv", release), dtype=str
        )
        region = area.map(dict(zip(lk["area"].str.upper(), lk["nhse_region"])))

        grade = v["pay_scheme_band"].map(BAND_TO_GRADE)
        grade = grade.where(grade.notna() | (v["pay_scheme_type"] != VSM), "Very Senior Manager")

        lookup = salary.build_lookup(self, release)
        sal = salary.attach(v[["staff_group", "pay_scheme_band", "pay_scheme_type"]], lookup)

        years = set(self.config["weighting"]["rake_years"])
        non_england = area.isin(NON_ENGLAND_POSTCODE_AREAS)
        in_years = v["year"].isin(years)
        exclusion = pd.Series(pd.NA, index=v.index, dtype="string")
        exclusion = exclusion.mask(non_england, "outside England (HCHS is England-only)")
        exclusion = exclusion.mask(~in_years, "outside rake years 2021-2026")

        vid = self.ids(v["vacancy_id"])
        adverts = pd.DataFrame(
            {
                "vacancy_id": vid,
                "source_vacancy_id": v["vacancy_id"].astype("string"),
                "source": "NHS Jobs",
                "posted_date": pd.to_datetime(v["created_date"], errors="coerce"),
                "year": v["year"].astype("Int64"),
                "job_title": v["job_title"],
                "employer": v["employer_name"],
                "org_unit": v["employer_name"],
                "occupation_group": v["staff_group"],
                "grade": grade,
                "region": region,
                "salary_annual_gbp": sal["estimated_salary"],
                "salary_source": sal["salary_source"],
                "headcount": 1,
                "in_scope": exclusion.isna(),
                "exclusion_reason": exclusion,
            }
        )
        ext = pd.DataFrame(
            {
                "vacancy_id": vid,
                "nhs_jobs_id": v["id"],
                "pay_scheme_type": v["pay_scheme_type"],
                "pay_scheme_band": v["pay_scheme_band"],
                "employer_postcode": v["employer_address_postcode"],
                "closing_date": v["closing_date"],
                "contract_type": v["contract_type"],
                "full_time": v["full_time"],
                "part_time": v["part_time"],
                "advert_salary_annual_gbp": salary.advert_annual_salary(v),
                # Salary-basis sensitivity: the advert's own salary, AfC-uprated to base year.
                "salary_advert_uprated_gbp": salary.uprated_advert_salary(
                    v, self.config["salary"]["base_year"]
                ),
            }
        )
        text = pd.DataFrame({"vacancy_id": vid, "advert_text": v["job_text"]})
        return CleanOutput(
            adverts, ext, text, no_funnel(len(v)), inputs=[src, self.reference("", release)]
        )

    def score(self, release: str) -> ScoreOutput:
        src = self.legacy("tasks")
        t = storage.read_parquet(src)
        vid = self.ids(t["vacancy_id"])
        model = self.primary_score_model
        tasks = pd.DataFrame(
            {
                "vacancy_id": vid,
                "task_index": t["task_index"],
                "task_text": t["task"],
                "exposure_score": t["automation_potential"],
                "score_model": model,
                "task_label": pd.NA,  # NHS full run was not labelled
                "scoring_run_id": "nhs-full_run-2026-09",
            }
        )
        model_scores = tasks[["vacancy_id", "task_index", "exposure_score"]].assign(model=model)
        reasoning = pd.DataFrame(
            {"vacancy_id": vid, "task_index": t["task_index"], "reasoning": t["reasoning"]}
        )
        return ScoreOutput(tasks, model_scores, reasoning, inputs=[src])

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        return hchs.rake_to_hchs(self, adverts, n_tasks, release)


SECTOR = NHS
