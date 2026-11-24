"""Police: force job-board adverts (officer, staff and PCSO roles).

Mapping onto the contract:
  employer / org_unit  canonical force name (cleaning.normalise_force)
  occupation_group     worker type from the title: Police officer / PCSO / Special constable /
                       Police staff (the Home Office workforce categories; title-based, so
                       approximate)
  region               the force's ONS region or nation (config.toml [force_region])
  salary               advert range midpoint, annual £12k-£250k only; where the salary fields
                       are empty, the salary the advert TEXT states (salary_text.py), never an
                       estimate

Built from raw (since 2026-11-24): cleaning.py holds the cleaning rules, ported from the original
cleaning repo. Task scores are the October 2026 Bedrock run, kept in raw/ as an input.

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
    year_of,
)
from . import cleaning, workforce
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
        src = self._raw_path(release)
        raw = pd.read_csv(io.BytesIO(storage.read_bytes(src)), usecols=cleaning.RAW_COLS, dtype=str)
        # The source ids collide across boards, so the id is the row position in the raw file
        # (as in the original cleaning repo, so ids match earlier outputs).
        raw["vacancy_id"] = [f"pol-{i:05d}" for i in range(len(raw))]
        v = cleaning.clean_adverts(raw)
        vid = self.ids(v["vacancy_id"])
        exclusion = v["exclusion_reason"].mask(
            v["exclusion_reason"].isna() & ~v["is_uk_force"], "not a UK police force"
        )
        salary = advert_midpoint_salary(v["salary_min"], v["salary_max"])
        from_text = v["jd_text"].map(salary_from_text).astype("float64").where(salary.isna())
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
                "is_uk_force": v["is_uk_force"],
                "is_recruitment_process": v["is_recruitment_process"],
                "hours": v["hours"],
                "salary_min_raw": v["salary_min"],
                "salary_max_raw": v["salary_max"],
            }
        )
        # What the extraction prompt is given (title + cleaned text), for adverts sent for scoring.
        sent = v["exclusion_reason"].isna()
        text = pd.DataFrame(
            {
                "vacancy_id": vid[sent],
                "advert_text": "Job title: " + v.loc[sent, "job_title"] + "\n\n"
                + v.loc[sent, "job_text"],
            }
        )  # fmt: skip
        return CleanOutput(adverts, ext, text, cleaning.funnel(v), inputs=[src])

    def _raw_path(self, release: str, name: str = "all_police_jobs_combined.csv") -> str:
        return storage.join(settings.raw_dir(self.slug, release), name)

    def score(self, release: str) -> ScoreOutput:
        # Scored on Bedrock (gpt-oss-120b) in October 2026 by the original cleaning repo, from the
        # same cleaned adverts; the output is kept in raw/ as an input rather than re-run.
        src = self._raw_path(release, self.config["scoring"]["stored_scores"])
        return adopt_labelled_tasks(self, "police-full_run-2026-10", src)

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        return workforce.rake_to_workforce(self, adverts, n_tasks, release)


SECTOR = Police
