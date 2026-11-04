"""Local government: council-advertised jobs from regional job boards.

Mapping onto the contract:
  employer / org_unit  council, as advertised
  occupation_group     not built yet (no taxonomy agreed)
  region               the matched council's region in QPSES (England regions + Wales)
  salary               advert range midpoint, annual £12k-£250k only
Sector columns: qpses_unit (the raking unit the council belongs to, e.g. "Bromsgrove +
Redditch") and authority_type (QPSES type: London Borough, Met District, Shire County, ...).

FLAG, kept in for now: about 13% of adverts have school-type titles (teachers, teaching
assistants, heads, SENCOs, posts named after a school or academy). They are council-advertised
posts in council-maintained schools. They stay in local government with `is_school_role` in
curated vacancies so any cut can include or exclude them. To discuss: these staff are probably
also in DfE's School Workforce figures that Education rakes to.

Weights: raked to council headcount from ONS QPSES (LGA per-authority tables), see qpses.py.
Adverts from councils QPSES does not cover (Scottish and NI councils, town and parish
councils, other bodies) are kept but get no weight.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ... import storage
from ..base import (
    CleanOutput,
    ScoreOutput,
    Sector,
    WeightOutput,
    adopt_labelled_tasks,
    advert_midpoint_salary,
    no_funnel,
    year_of,
)
from . import qpses

SCHOOL_TITLE_PATTERNS = {
    "teacher": r"\bteacher\b(?!.*assistant)|\bteachers\b",
    "teaching_assistant": r"\b(?:teaching|learning|classroom) (?:support )?assistants?\b",
    "head_or_deputy": r"\b(?:head ?teacher|deputy head|assistant head)\b",
    "school_in_title": r"\bschool\b|\bacademy\b",
    "senco": r"\bsend?co\b",
}


def school_role_type(titles: pd.Series) -> pd.Series:
    """First matching school-type pattern per title, else NA."""
    t = titles.fillna("").astype(str)
    out = pd.Series(pd.NA, index=t.index, dtype="string")
    for label, pat in reversed(SCHOOL_TITLE_PATTERNS.items()):
        out = out.mask(t.str.contains(pat, case=False, regex=True), label)
    return out


class LocalGovt(Sector):
    slug = "local-govt"
    package_dir = Path(__file__).parent

    def clean(self, release: str) -> CleanOutput:
        src = self.legacy("vacancies")
        v = storage.read_parquet(src)
        vid = self.ids(v["vacancy_id"])
        posted = pd.to_datetime(v["date"], errors="coerce")
        salary = advert_midpoint_salary(v["salary_min"], v["salary_max"])
        adverts = pd.DataFrame(
            {
                "vacancy_id": vid,
                "source_vacancy_id": v["vacancy_id"],
                "source": v["source"],
                "posted_date": posted,
                "year": year_of(v["date"]),
                "job_title": v["job_title"],
                "employer": v["council"],
                "org_unit": v["council"],
                "occupation_group": pd.NA,
                "grade": pd.NA,
                "region": pd.NA,  # filled from QPSES below
                "salary_annual_gbp": salary,
                "salary_source": pd.Series("advert_midpoint", index=v.index).where(salary.notna()),
                "headcount": 1,
                "in_scope": True,
                "exclusion_reason": pd.NA,
            }
        )
        q = qpses.load(self.reference(self.config["weighting"]["reference_file"], release))
        per, _ = qpses.assign(adverts["employer"], q)
        adverts["region"] = per["region"]
        school = school_role_type(v["job_title"])
        ext = pd.DataFrame(
            {
                "vacancy_id": vid,
                "board_id": v["id"],
                "url": v["url"],
                "hours": v["hours"],
                "salary_min_raw": v["salary_min"],
                "salary_max_raw": v["salary_max"],
                "is_school_role": school.notna(),
                "school_role_type": school,
                "qpses_unit": per["qpses_unit"],
                "authority_type": per["authority_type"],
            }
        )
        ref = self.reference(self.config["weighting"]["reference_file"], release)
        return CleanOutput(adverts, ext, None, no_funnel(len(v)), inputs=[src, ref])

    def score(self, release: str) -> ScoreOutput:
        return adopt_labelled_tasks(self, "local-govt-full_run-2026-10")

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        return qpses.rake_to_qpses(self, adverts, n_tasks, release)


SECTOR = LocalGovt
