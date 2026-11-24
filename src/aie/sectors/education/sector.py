"""Education: state-funded school workforce in England, from Teaching Vacancies adverts.

Mapping onto the contract:
  employer / org_unit  matched school name (GIAS match is partial: ~27% confirmed)
  occupation_group     dfe_category (the 11 DfE School Workforce grade/post categories)
  grade                not separately available (the teacher grade is part of dfe_category)
  region               school_region from the GIAS match (partial)
  salary               DfE Teacher pay (TPS) median for the 4 teaching categories, year-matched;
                       the advert's own annual salary for support staff (no official source)

Built from raw (since 2026-11-24): cleaning.py and the modules it calls (salary, categories,
role_groups, schools) hold the rules, ported from gds-idea-education-ai-exposure-cleaning. Task
scores and labels are the October 2026 Bedrock run, kept in raw/ as a PROCESSED input.

Raked 1D to DfE School Workforce 2025/26 on dfe_category and phase. Known gaps (see README):
no employer-type field, so independent-school adverts cannot be screened out; special/PRU and
centrally employed staff (11.8% of FTE) have no phase equivalent in the adverts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ... import settings, storage
from ...methods import rake
from ..base import CleanOutput, ScoreOutput, Sector, WeightOutput
from . import cleaning, schools

# {academic-year start: (classroom, other leadership, head)}, £ median FTE gross pay, DfE
# "Teacher pay - TPS", read 2026-10-08. DfE blends deputy and assistant heads.
TPS_MEDIAN_PAY: dict[int, tuple[int, int, int]] = {
    2010: (35_769, 50_359, 59_809),
    2011: (35_601, 50_359, 59_842),
    2012: (35_447, 50_359, 61_288),
    2013: (35_802, 50_863, 61_901),
    2014: (36_161, 51_372, 62_521),
    2015: (36_523, 51_381, 63_152),
    2016: (36_889, 51_519, 64_864),
    2017: (37_258, 51_991, 66_016),
    2018: (38_004, 52_771, 67_008),
    2019: (39_050, 54_400, 68_851),
    2020: (40_444, 56_720, 70_745),
    2021: (40_960, 56_721, 70_745),
    2022: (42_923, 59_558, 74_283),
    2023: (46_177, 63_430, 79_074),
    2024: (49_037, 66_919, 83_236),
    2025: (51_047, 69_596, 86_803),
}
CATEGORY_TO_SLOT = {
    "Classroom teacher": 0,
    "Deputy head teacher": 1,
    "Assistant head teacher": 1,
    "Head teacher": 2,
}


def teacher_pay(dfe_category: pd.Series, year: pd.Series) -> pd.Series:
    """External median FTE pay for teaching categories (NaN for support staff)."""
    y = year.clip(lower=min(TPS_MEDIAN_PAY), upper=max(TPS_MEDIAN_PAY))
    out = pd.Series(np.nan, index=dfe_category.index, dtype="float64")
    for category, slot in CATEGORY_TO_SLOT.items():
        mask = dfe_category.eq(category) & y.notna()
        out.loc[mask] = y.loc[mask].astype(int).map(lambda yr, s=slot: TPS_MEDIAN_PAY[yr][s])
    return out


class Education(Sector):
    slug = "education"
    package_dir = Path(__file__).parent

    def clean(self, release: str) -> CleanOutput:
        raw_dir = settings.raw_dir(self.slug, release)
        shards = [storage.join(raw_dir, Path(f).name) for f in self.config["raw"]["files"]
                  if Path(f).name.startswith("total_vacancies_published-")]  # fmt: skip
        gias_src = self.reference(self.config["weighting"]["gias_file"], release)
        with storage.open_input(gias_src) as fh:
            gias = schools.load_gias(fh)
        v = cleaning.clean_adverts(cleaning.read_shards(shards), gias)
        year = v["year"].astype("Int64")

        clean_basis = v["salary_basis"].eq("annual") & ~v["salary_is_actual"].fillna(False).astype(
            bool
        )
        advert = v[["salary_min", "salary_max"]].mean(axis=1, skipna=True).where(clean_basis)
        tps = teacher_pay(v["dfe_category"], year.astype("float64"))
        salary = tps.fillna(advert)
        source = np.where(
            tps.notna(), "dfe_tps_external", np.where(advert.notna(), "advert_parsed", None)
        )

        school = v["candidate_school_name"].where(v["school_match_urn"].notna())
        vid = self.ids(v["vacancy_id"])
        adverts = pd.DataFrame(
            {
                "vacancy_id": vid,
                "source_vacancy_id": v["vacancy_id"],
                "source": "Teaching Vacancies (DfE)",
                "posted_date": pd.to_datetime(v["first_published_on"], errors="coerce"),
                "year": year,
                "job_title": v["job_title"],
                "employer": school,
                "org_unit": school,
                "occupation_group": v["dfe_category"],
                "grade": pd.NA,
                "region": v["school_region"],
                "salary_annual_gbp": salary,
                "salary_source": source,
                "headcount": 1,
                "in_scope": v["exclusion_reason"].isna(),
                "exclusion_reason": v["exclusion_reason"],
            }
        )
        phase_map = self.config["weighting"]["phase_map"]
        ext = pd.DataFrame(
            {
                "vacancy_id": vid,
                "is_teaching": v["is_teaching"],
                "phase": v["primary_education_phase"].map(phase_map),
                "primary_education_phase": v["primary_education_phase"],
                "education_phase": v["education_phase"],
                "primary_job_role": v["primary_job_role"],
                "job_type": v["job_type_clean"],
                "working_patterns": v["working_patterns"],
                "advert_salary_annual_gbp": advert,
                "school_urn": v["school_match_urn"],
                "school_la": v["school_la"],
                "school_phase_gias": v["school_phase_gias"],
                "school_name_ambiguous": v["school_name_ambiguous"],
                "state_funded_confirmed": v["state_funded_confirmed"],
            }
        )
        sent = v["exclusion_reason"].isna()  # the extraction prompt was given title + text
        text = pd.DataFrame(
            {
                "vacancy_id": vid[sent],
                "advert_text": "Job title: " + v.loc[sent, "job_title"] + "\n\n"
                + v.loc[sent, "job_text"],
            }
        )  # fmt: skip
        return CleanOutput(adverts, ext, text, cleaning.funnel(v), inputs=[raw_dir, gias_src])

    def score(self, release: str) -> ScoreOutput:
        # Scored and labelled on Bedrock (gpt-oss-120b) in October 2026 by the original cleaning
        # repo, from the same cleaned adverts; kept in raw/ as a PROCESSED input, not re-run.
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
                "task_label": t["label"],
                "scoring_run_id": "education-full_run-2026-10",
            }
        )
        model_scores = tasks[["vacancy_id", "task_index", "exposure_score"]].assign(model=model)
        reasoning = pd.DataFrame(
            {
                "vacancy_id": vid,
                "task_index": t["task_index"],
                "reasoning": t["reasoning"],
                "label_reasoning": t["label_reasoning"],
            }
        )
        return ScoreOutput(tasks, model_scores, reasoning, inputs=[src])

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        cfg = self.config["weighting"]
        sample = self.min_tasks_sample(adverts, n_tasks).merge(
            ext[["vacancy_id", "phase"]], on="vacancy_id", how="left"
        )
        sample = sample.rename(columns={"occupation_group": "dfe_category"})
        margins = {d: dict(cfg["targets"][d]) for d in cfg["dimensions"]}
        res = rake.rake(
            sample[["vacancy_id", *cfg["dimensions"]]],
            margins,
            target_fte=sum(margins["dfe_category"].values()),
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


SECTOR = Education
