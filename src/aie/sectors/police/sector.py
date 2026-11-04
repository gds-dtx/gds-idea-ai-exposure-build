"""Police: force job-board adverts (officer, staff and PCSO roles).

Mapping onto the contract:
  employer / org_unit  canonical force name (normalised in the old cleaning repo)
  occupation_group     worker type from the title: Police officer / PCSO / Special constable /
                       Police staff (the Home Office workforce categories; title-based, so
                       approximate)
  region               the force's ONS region or nation (config.toml [force_region])
  salary               advert range midpoint, annual £12k-£250k only; where the salary fields
                       are empty, the salary the advert TEXT states (salary_text.py), never an
                       estimate

In scope: UK police forces only, and not recruitment-route pages (transferee schemes, talent
pools, PCDA intake pages), which describe a process rather than a role.

Weights: police staff and PCSOs raked to Home Office workforce FTE by force (workforce.py).
Officers, special constables and volunteers are not weighted.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ... import settings, storage
from ..base import (
    CleanOutput,
    ScoreOutput,
    Sector,
    WeightOutput,
    adopt_labelled_tasks,
    advert_midpoint_salary,
    no_funnel,
    to_bool,
    year_of,
)
from . import workforce
from .salary_text import salary_from_text

_STAFF = r"\bpolice staff\b|\bstaff investigator\b|\binvestigat(?:or|ion) officer\b"
_PCSO = r"\bpcso\b|community support officer"
_SPECIAL = r"\bspecial constab|\bspecials?\b(?! branch)"
_OFFICER = (
    r"\b(?:police constable|detective|sergeant|inspector|superintendent|chief constable|"
    r"\bpc\b|\bdc\b|\bds\b|constable)"
)


def worker_type(titles: pd.Series) -> pd.Series:
    t = titles.fillna("").astype(str)
    out = pd.Series("Police staff", index=t.index, dtype="string")
    officer = t.str.contains(_OFFICER, case=False, regex=True) & ~t.str.contains(
        _STAFF, case=False, regex=True
    )
    out = out.mask(officer, "Police officer")
    out = out.mask(t.str.contains(_SPECIAL, case=False, regex=True), "Special constable")
    out = out.mask(t.str.contains(_PCSO, case=False, regex=True), "PCSO")
    return out


class Police(Sector):
    slug = "police"
    package_dir = Path(__file__).parent

    def clean(self, release: str) -> CleanOutput:
        src = self.legacy("vacancies")
        v = storage.read_parquet(src)
        vid = self.ids(v["vacancy_id"])
        in_force = to_bool(v["in_scope"]).fillna(False)
        recruit = to_bool(v["is_recruitment_process"]).fillna(False)
        exclusion = pd.Series(pd.NA, index=v.index, dtype="string")
        exclusion = exclusion.mask(recruit, "recruitment-route page, not a role")
        exclusion = exclusion.mask(~in_force, "not a UK police force")
        salary = advert_midpoint_salary(v["salary_min"], v["salary_max"])
        from_text = self._salary_from_advert_text(v, release).where(salary.isna())
        source = pd.Series("advert_midpoint", index=v.index).where(salary.notna())
        source = source.mask(from_text.notna(), "advert_text")
        salary = salary.fillna(from_text)
        adverts = pd.DataFrame(
            {
                "vacancy_id": vid,
                "source_vacancy_id": v["vacancy_id"],
                "source": v["source"],
                "posted_date": pd.to_datetime(v["date"], errors="coerce"),
                "year": year_of(v["date"]),
                "job_title": v["job_title"],
                "employer": v["force"],
                "org_unit": v["force"],
                "occupation_group": worker_type(v["job_title"]),
                "grade": pd.NA,
                "region": v["force"].map(self.config["force_region"]),
                "salary_annual_gbp": salary,
                "salary_source": source,
                "headcount": 1,
                "in_scope": exclusion.isna(),
                "exclusion_reason": exclusion,
            }
        )
        ext = pd.DataFrame(
            {
                "vacancy_id": vid,
                "board_id": v["id"],
                "url": v["url"],
                "force_raw": v["force_raw"],
                "is_uk_force": in_force,
                "is_recruitment_process": recruit,
                "hours": v["hours"],
                "salary_min_raw": v["salary_min"],
                "salary_max_raw": v["salary_max"],
            }
        )
        return CleanOutput(
            adverts, ext, None, no_funnel(len(v)), inputs=[src, self._raw_path(release)]
        )

    def _raw_path(self, release: str) -> str:
        name = Path(self.config["raw"]["files"][0]).name
        return storage.join(settings.raw_dir(self.slug, release), name)

    def _salary_from_advert_text(self, v: pd.DataFrame, release: str) -> pd.Series:
        """Salary stated in the raw advert text (jd_text), matched on source + board id."""
        raw = pd.read_csv(
            io.BytesIO(storage.read_bytes(self._raw_path(release))),
            usecols=["source", "id", "jd_text"],
            dtype=str,
        )
        key = raw["source"] + "|" + raw["id"]
        text = pd.Series(raw["jd_text"].to_numpy(), index=key)
        text = text[~text.index.duplicated()]
        found = (v["source"].astype(str) + "|" + v["id"].astype(str)).map(text)
        return pd.Series(found.map(salary_from_text).to_numpy(), index=v.index, dtype="float64")

    def score(self, release: str) -> ScoreOutput:
        return adopt_labelled_tasks(self, "police-full_run-2026-10")

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        return workforce.rake_to_workforce(self, adverts, n_tasks, release)


SECTOR = Police
