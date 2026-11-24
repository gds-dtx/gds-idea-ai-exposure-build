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

Built from raw (since 2026-11-24): clean_adverts below holds the cleaning rules, ported from the
original cleaning repo. Task scores are the October 2026 Bedrock run, kept in raw/ as a
PROCESSED input (see config.toml [raw.notes]).

Weights: raked to council FTE from ONS QPSES (LGA per-authority tables), see qpses.py.
Adverts from councils QPSES does not cover (Scottish and NI councils, town and parish
councils, other bodies) are kept but get no weight.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ... import settings, storage
from ..advert_cleaning import exclusion_reasons, funnel, strip_html
from ..base import (
    CleanOutput,
    ScoreOutput,
    Sector,
    WeightOutput,
    adopt_labelled_tasks,
    advert_midpoint_salary,
    year_of,
)
from . import qpses

RAW_COLS = [
    "source", "id", "url", "job_title", "council", "salary_min", "salary_max",
    "hours", "date", "jd_text", "document_text",
]  # fmt: skip


def clean_adverts(raw: pd.DataFrame) -> pd.DataFrame:
    """Every raw row, in file order, with job_text and exclusion_reason (NA = sent for scoring).

    Ported from gds-idea-local-govt-ai-exposure-cleaning: advert text is jd_text plus any
    attached document text; the shared job-board rules (advert_cleaning.py) with title + text
    as the repeat key. `raw` must already carry `vacancy_id`.
    """
    d = raw.copy().reset_index(drop=True)
    jd = d["jd_text"].map(strip_html)
    doc = d["document_text"].map(strip_html)
    d["job_text"] = (jd + " " + doc).str.strip()
    d["exclusion_reason"] = exclusion_reasons(d)
    return d


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
        src = self._raw_path(release)
        raw = pd.read_csv(io.BytesIO(storage.read_bytes(src)), usecols=RAW_COLS, dtype=str)
        # The source ids collide across boards, so the id is the row position in the raw file
        # (as in the original cleaning repo, so ids match earlier outputs).
        raw["vacancy_id"] = [f"lg-{i:07d}" for i in range(len(raw))]
        v = clean_adverts(raw)
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
                "in_scope": v["exclusion_reason"].isna(),
                "exclusion_reason": v["exclusion_reason"],
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
        # What the extraction prompt is given (title + cleaned text), for adverts sent for scoring.
        sent = v["exclusion_reason"].isna()
        text = pd.DataFrame(
            {
                "vacancy_id": vid[sent],
                "advert_text": "Job title: " + v.loc[sent, "job_title"] + "\n\n"
                + v.loc[sent, "job_text"],
            }
        )  # fmt: skip
        ref = self.reference(self.config["weighting"]["reference_file"], release)
        return CleanOutput(adverts, ext, text, funnel(v), inputs=[src, ref])

    def _raw_path(self, release: str, name: str = "all_local_council_jobs.csv") -> str:
        return storage.join(settings.raw_dir(self.slug, release), name)

    def score(self, release: str) -> ScoreOutput:
        # Scored on Bedrock (gpt-oss-120b) in October 2026 by the original cleaning repo, from the
        # same cleaned adverts; the output is kept in raw/ as an input rather than re-run.
        src = self._raw_path(release, self.config["scoring"]["stored_scores"])
        return adopt_labelled_tasks(self, "local-govt-full_run-2026-10", src)

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        return qpses.rake_to_qpses(self, adverts, n_tasks, release)


SECTOR = LocalGovt
