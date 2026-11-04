"""Civil Service: from Civil Service Jobs adverts (GRID), scored by an 8-model ensemble.

Mapping onto the contract:
  employer          department as advertised
  org_unit          cs_org (CS Statistics organisation, used for raking and salary)
  occupation_group  cs_profession (profession-corrected)
  grade             cs_grade
  region            vacancy_region
  salary            CS Statistics salary lookup (estimated_salary from the canonical dataset)
  headcount         number_of_jobs_available

Weights: rake_1d (primary) and rake_3d, adopted from gds-idea-ai-exposure for v1.
Scores: curated tasks carry the ensemble mean (CS headline) as exposure_score, plus a
score_<model> column for each of the 8 models, including score_gpt_oss_120b for cross-sector cuts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ... import storage
from ..base import CleanOutput, ScoreOutput, Sector, WeightOutput, no_funnel

ADVERT_COLS = [
    "vacancy_id", "vacancy_title", "department", "job_grade", "profession", "cs_org",
    "cs_profession", "cs_grade", "number_of_jobs_available", "most_recent_posting_live",
    "most_recent_posting_live_imputed", "closing_date", "approach", "business_area",
    "business_area_detail", "vacancy_postcode", "vacancy_city", "vacancy_region",
    "salary_minimum", "salary_maximum_optional", "cs_profession_original",
    "profession_reassignment_method", "job_summary", "job_description",
]  # fmt: skip


class CivilService(Sector):
    slug = "civil-service"
    package_dir = Path(__file__).parent

    def clean(self, release: str) -> CleanOutput:
        a_src, c_src = self.legacy("adverts"), self.legacy("canonical")
        a = storage.read_parquet(a_src, columns=ADVERT_COLS)
        c = storage.read_parquet(c_src, columns=["vacancy_id", "estimated_salary", "salary_source"])
        a = a.merge(c, on="vacancy_id", how="left", validate="one_to_one")
        for col in ("department", "cs_org", "cs_profession", "cs_grade"):
            a[col] = a[col].astype("string")

        posted = pd.to_datetime(a["most_recent_posting_live"], errors="coerce")
        vid = self.ids(a["vacancy_id"])
        adverts = pd.DataFrame(
            {
                "vacancy_id": vid,
                "source_vacancy_id": a["vacancy_id"],
                "source": "Civil Service Jobs",
                "posted_date": posted,
                "year": posted.dt.year.astype("Int64"),
                "job_title": a["vacancy_title"],
                "employer": a["department"],
                "org_unit": a["cs_org"],
                "occupation_group": a["cs_profession"],
                "grade": a["cs_grade"],
                "region": a["vacancy_region"],
                "salary_annual_gbp": a["estimated_salary"],
                "salary_source": a["salary_source"],
                "headcount": a["number_of_jobs_available"].astype("Int64"),
                "in_scope": True,
                "exclusion_reason": pd.NA,
            }
        )
        ext = a.drop(
            columns=[
                "vacancy_title", "cs_org", "cs_profession", "cs_grade", "vacancy_region",
                "estimated_salary", "salary_source", "number_of_jobs_available",
                "most_recent_posting_live", "job_summary", "job_description",
            ]
        ).assign(vacancy_id=vid)  # fmt: skip
        text = pd.DataFrame(
            {
                "vacancy_id": vid,
                "advert_text": (
                    a["job_summary"].fillna("") + "\n\n" + a["job_description"].fillna("")
                ).str.strip(),
            }
        )
        return CleanOutput(adverts, ext, text, no_funnel(len(a)), inputs=[a_src, c_src])

    def score(self, release: str) -> ScoreOutput:
        src = self.legacy("tasks")
        models = self.config["scoring"]["models"]
        t = storage.read_parquet(
            src, columns=["vacancy_id", "task_number", "task", *models, "mean_ap"]
        )
        vid = self.ids(t["vacancy_id"])
        tasks = pd.DataFrame(
            {
                "vacancy_id": vid,
                "task_index": t["task_number"].astype("Int64"),
                "task_text": t["task"],
                "exposure_score": t["mean_ap"],
                "score_model": self.primary_score_model,
                "task_label": pd.NA,
                "scoring_run_id": "cs-grid-8model-2026-08",
            }
        )
        long = (
            t[["task_number", *models]]
            .assign(vacancy_id=vid)
            .melt(
                id_vars=["vacancy_id", "task_number"], var_name="model", value_name="exposure_score"
            )
            .rename(columns={"task_number": "task_index"})
        )
        # Unscored task x model cells are kept as nulls so decay keeps each task's position.
        long["model"] = long["model"].replace(self.config["scoring"]["rename_models"])
        return ScoreOutput(tasks, long, None, inputs=[src])

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        src = self.legacy("weights")
        w = storage.read_parquet(src)
        w["vacancy_id"] = self.ids(w["vacancy_id"])
        cfg = self.config["weighting"]
        frames = []
        for spec, col in (("rake_1d", "weight_1d"), ("rake_3d", "weight_3d")):
            frames.append(
                adverts[["vacancy_id"]]
                .merge(w[["vacancy_id", col]], on="vacancy_id", how="left")
                .rename(columns={col: "weight"})
                .assign(
                    weight_spec=spec,
                    is_primary=spec == cfg["primary_spec"],
                    target_source=cfg["target_source"],
                    target_date=cfg["target_date"],
                )
            )
        weights = pd.concat(frames, ignore_index=True)
        diag = {
            spec: {
                "n_weighted": int(g["weight"].notna().sum()),
                "weighted_fte": float(g["weight"].sum()),
            }
            for spec, g in weights.groupby("weight_spec")
        }
        diag["note"] = "adopted from gds-idea-ai-exposure stage 3 (v2/raw/vacancy_weights.parquet)"
        return WeightOutput(weights, None, diag, inputs=[src])


SECTOR = CivilService
