"""Release v2's advert pipeline, from the raw GRID files to the post-processed, profession-
corrected adverts (the original's grid_post_processed_profession_corrected.parquet).

Steps and the original code each comes from (gds-idea-ai-exposure):
  0  clean the GRID export (grid_cleaning.py; data_cleaning.py at e246377)
  1c join the GRID team's extra fields, keep scored adverts, flag and drop duplicates and
     test titles (post_processing.py; data_quality.py and 1c_grid_post_processing.ipynb at
     c2a10a5)
  1d replace 'Other' professions with the stored LLM classification (professions.py)

Between steps the original saved and re-read parquet files; that round trip is kept (in
memory) so column types match exactly.
"""

from __future__ import annotations

import io

import pandas as pd

from . import grid_cleaning, post_processing, professions


def _parquet_round_trip(df: pd.DataFrame) -> pd.DataFrame:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return pd.read_parquet(buf)


def clean_grid(grid_file, professions_file) -> pd.DataFrame:
    """Step 0."""
    raw = grid_cleaning.load_raw_grid(grid_file, professions_file)
    return _parquet_round_trip(grid_cleaning.clean_raw_grid_data(raw, verbose=False))


JOIN_COLS = [
    "vacancy_id", "approach", "number_of_jobs_available", "business_area", "business_area_detail",
    "vacancy_postcode", "vacancy_city", "vacancy_region", "salary_minimum",
    "salary_maximum_optional",
]  # fmt: skip


def post_process(grid: pd.DataFrame, new_fields: pd.DataFrame, scored_ids) -> pd.DataFrame:
    """Step 1c, as the notebook ran it. `new_fields` is dsit_full_dept.xlsx (the GRID team's
    extra fields); `scored_ids` are the adverts that have task scores."""
    new_df = new_fields.copy()
    new_df["vacancy_id"] = new_df["vacancy_id"].astype(str)
    new_df = new_df[new_df["vacancy_id"].isin(list(grid["vacancy_id"].unique()))]
    new_df = new_df[new_df["vacancy_id"].isin(list(scored_ids))]
    assert new_df["vacancy_id"].duplicated().sum() == 0, "duplicate vacancy ids in new fields"

    joint = grid.merge(new_df[JOIN_COLS], on="vacancy_id", how="inner")
    joint = joint.sort_values(by="number_of_jobs_available", ascending=False)
    pp = post_processing
    joint = pp.remove_invalid_jobs_available(joint, verbose=False)
    joint = pp.clean_new_salary_fields(joint)
    joint = pp.clean_new_location_and_business_fields(joint)
    joint = pp.ensure_most_recent_posting_live(joint)
    joint = pp.cap_imputed_posting_dates(joint)
    joint = pp.flag_test_and_copy_titles(joint)
    joint = pp.flag_error_titles(joint)
    joint = pp.assign_duplicate_groups(joint)

    final = pp.apply_final_filters(joint)
    final = final.sort_values(by="number_of_jobs_available", ascending=False)
    present = [c for c in pp._DIAGNOSTIC_COLUMNS_TO_DROP if c in final.columns]
    return _parquet_round_trip(final.drop(columns=present))


def build(grid_file, professions_file, new_fields, scored_ids, labels) -> pd.DataFrame:
    """Steps 0, 1c and 1d: the post-processed, profession-corrected adverts."""
    grid = clean_grid(grid_file, professions_file)
    pp = post_process(grid, new_fields, scored_ids)
    return professions.apply_labels(pp, labels)
