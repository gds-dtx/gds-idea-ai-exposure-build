"""NHS: HCHS workforce in England, from NHS Jobs / TRAC adverts.

Mapping onto the contract:
  employer          employer_name (trust / practice)
  org_unit          employer_name (`organisation` is ~0.1% populated)
  occupation_group  staff_group (the 7 HCHS groups, assigned in the old cleaning repo)
  grade             HCHS grade label from pay_scheme_band (Band 1-9, doctor grades, VSM)
  region            NHS England region, from employer postcode area
  salary            NHS Digital Staff Earnings Estimates lookup (see salary.py)

Built from raw (since 2026-11-24): cleaning.py and staff_groups.py hold the cleaning and
staff-group rules, ported from gds-idea-nhs-ai-exposure-cleaning. Task scores are the 2026
Bedrock run, kept in raw/ as a PROCESSED input (config.toml [raw.notes]).

In scope: rake years 2021-2026 and an England postcode (HCHS is England-only; the feed is
UK-wide). Raked 1D to HCHS May-2026 on staff_group, grade and region.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ... import settings, storage
from ..base import CleanOutput, ScoreOutput, Sector, WeightOutput
from . import cleaning, hchs, regions, salary
from .staff_groups import assign_staff_group

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
        raw = settings.raw_dir(self.slug, release)
        files = [
            (int(y), storage.join(raw, f)) for y, f in self.config["raw"]["year_files"].items()
        ]
        v, funnel = cleaning.clean_years(files)
        v["staff_group"] = assign_staff_group(v)

        area = postcode_area(v["employer_address_postcode"])
        region = area.map(regions.area_to_nhse_region())

        grade = v["pay_scheme_band"].map(BAND_TO_GRADE)
        grade = grade.where(grade.notna() | (v["pay_scheme_type"] != VSM), "Very Senior Manager")

        lookup = salary.build_lookup(self, release)
        sal = salary.attach(v[["staff_group", "pay_scheme_band", "pay_scheme_type"]], lookup)

        years = set(self.config["weighting"]["rake_years"])
        non_england = area.isin(NON_ENGLAND_POSTCODE_AREAS)
        in_years = v["year"].isin(years)
        # A cleaning reason comes first; then the year rule over the England rule (as before).
        exclusion = v["exclusion_reason"].astype("string")
        exclusion = exclusion.mask(exclusion.isna() & ~in_years, "outside rake years 2021-2026")
        exclusion = exclusion.mask(
            exclusion.isna() & non_england, "outside England (HCHS is England-only)"
        )

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
        sent = v["input_text"].notna()  # what the extraction prompt was given
        text = pd.DataFrame({"vacancy_id": vid[sent], "advert_text": v.loc[sent, "input_text"]})
        return CleanOutput(adverts, ext, text, funnel, inputs=[raw, self.reference("", release)])

    def score(self, release: str) -> ScoreOutput:
        # Scored on Bedrock (gpt-oss-120b) in 2026 by the original cleaning repo, from the same
        # cleaned adverts; the output is kept in raw/ as a PROCESSED input rather than re-run.
        src = storage.join(
            settings.raw_dir(self.slug, release), self.config["scoring"]["stored_scores"]
        )
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
