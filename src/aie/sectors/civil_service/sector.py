"""Civil Service: from Civil Service Jobs adverts (GRID), scored by an 8-model ensemble.

Mapping onto the contract:
  employer          department as advertised
  org_unit          cs_org (CS Statistics organisation, used for raking and salary)
  occupation_group  cs_profession (profession-corrected)
  grade             cs_grade
  region            vacancy_region
  salary            CS Statistics salary lookup (weighting.salary_lookup)
  headcount         number_of_jobs_available

Release v2 is built from raw here (since 2026-11-25): grid_v2.py runs the original pipeline's
cleaning, post-processing and profession correction (ported from gds-idea-ai-exposure, each
module from the commit release v2 was made with), and weighting.py its raking and salary
lookup. The task scores and the profession LLM run are stored PROCESSED inputs in raw/.
Rebuilt this way, v2 reproduces the release first adopted from ai-exposure/v2/ exactly.

Weights: rake_1d (primary) and rake_3d, on adverts with at least MIN_TASKS tasks.
Scores: curated tasks carry the ensemble mean (CS headline) as exposure_score, plus a
score_<model> column for each of the 8 models, including score_gpt_oss_120b for cross-sector cuts.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd

from ... import settings, storage
from ..base import CleanOutput, ScoreOutput, Sector, WeightOutput, no_funnel
from . import grid_v2, professions, stats_settings, weighting

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

    def _raw(self, release: str, name: str) -> str:
        return storage.join(settings.raw_dir(self.slug, release), name)

    def _use_stats(self, release: str) -> str:
        """Copy the release's Civil Service Statistics tables to a local folder and point
        cs_stats / rake / salary at it."""
        folder = tempfile.mkdtemp(prefix="aie-cs-stats-")
        for name in self.config["reference_sources"]:
            Path(folder, name).write_bytes(storage.read_bytes(self.reference(name, release)))
        stats_settings.use_stats_dir(folder)
        return folder

    def post_processed(self, release: str) -> pd.DataFrame:
        """The post-processed, profession-corrected adverts (grid_v2.build) from raw/."""
        raw = self.config["raw"]
        labels = pd.concat(
            [
                professions.parse_batch_output(
                    storage.read_bytes(self._raw(release, f)).decode().splitlines()
                )
                for f in raw["profession_label_files"]
            ],
            ignore_index=True,
        )
        scored = storage.read_parquet(self._raw(release, raw["scores_file"]), ["vacancy_id"])
        new_fields = pd.read_excel(
            io.BytesIO(storage.read_bytes(self._raw(release, raw["new_fields_file"])))
        )
        return grid_v2.build(
            io.BytesIO(storage.read_bytes(self._raw(release, raw["grid_file"]))),
            io.BytesIO(storage.read_bytes(self._raw(release, raw["professions_file"]))),
            new_fields,
            scored["vacancy_id"].astype(str).unique(),
            labels,
        )

    def clean(self, release: str) -> CleanOutput:
        g = self.post_processed(release)
        self._use_stats(release)
        lookup = weighting.salary_lookup(g)
        a = g[ADVERT_COLS].copy()
        key = ["cs_org", "cs_grade", "cs_profession"]
        sal = (
            a[key]
            .astype(str)
            .merge(lookup[[*key, "estimated_salary", "salary_source"]], on=key, how="left")
        )
        a["estimated_salary"] = sal["estimated_salary"].to_numpy()
        a["salary_source"] = sal["salary_source"].to_numpy()
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
        raw_dir = settings.raw_dir(self.slug, release)
        return CleanOutput(
            adverts, ext, text, no_funnel(len(a), "post-processed GRID adverts"),
            inputs=[raw_dir, self.reference("", release)],
        )  # fmt: skip

    def score(self, release: str) -> ScoreOutput:
        # The August 2026 extraction (Sonnet 4.6) and eight-model scoring, kept in raw/ as a
        # PROCESSED input rather than re-run.
        src = self._raw(release, self.config["raw"]["scores_file"])
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
        self._use_stats(release)
        sample = self.min_tasks_sample(adverts, n_tasks)
        grid = pd.DataFrame(
            {
                "vacancy_id": sample["source_vacancy_id"].astype(str),
                "cs_org": sample["org_unit"].astype(str),
                "cs_grade": sample["grade"].astype(str),
                "cs_profession": sample["occupation_group"].astype(str),
                "number_of_jobs_available": sample["headcount"].astype("int64"),
            }
        )
        w, diag = weighting.raking_weights(grid)
        w = w.assign(vacancy_id=self.ids(w["vacancy_id"]))
        cfg = self.config["weighting"]
        frames = [
            adverts[["vacancy_id"]]
            .merge(w[["vacancy_id", col]], on="vacancy_id", how="left")
            .rename(columns={col: "weight"})
            .assign(
                weight_spec=spec,
                is_primary=spec == cfg["primary_spec"],
                target_source=cfg["target_source"],
                target_date=cfg["target_date"],
            )
            for spec, col in (("rake_1d", "weight_1d"), ("rake_3d", "weight_3d"))
        ]
        diag["min_tasks_sample"] = int(len(sample))
        return WeightOutput(pd.concat(frames, ignore_index=True), None, diag)


SECTOR = CivilService
