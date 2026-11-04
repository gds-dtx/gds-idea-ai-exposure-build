"""Read curated tables: the one loader the pipeline, the analysis repo and notebooks share.

    from aie import read
    v = read.load("vacancies", ["nhs", "education"], release="v1")
    m = read.load("metrics", ["nhs"], release="v1", columns=["vacancy_id", "quadrant"])

Tables: vacancies, tasks, metrics (see aie.contract). Only the requested sector x release
files (and columns) are read, through `aie.storage`,
so this works the same on S3 and on a local AIE_ROOT. Partitions are cached in-process.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache

import pandas as pd

from . import catalog, settings, storage


def resolve_release(sector: str, release: str = "latest") -> tuple[str, str | None]:
    """(release, warning). 'latest' = newest published, else newest draft (with a warning)."""
    return catalog.resolve_release(sector, release)


@lru_cache(maxsize=64)
def columns(table: str, sector: str, release: str) -> tuple[str, ...]:
    """Columns actually present in one sector's file (core + that sector's extras)."""
    return tuple(storage.parquet_columns(settings.curated_file(table, sector, release)))


@lru_cache(maxsize=64)
def _partition(table: str, sector: str, release: str, cols: tuple[str, ...] | None) -> pd.DataFrame:
    present = columns(table, sector, release)
    want = [c for c in cols if c in present] if cols else None
    df = storage.read_parquet(settings.curated_file(table, sector, release), columns=want)
    for c in cols or ():
        if c not in df.columns:  # e.g. weight_rake_3d or a sector column this sector lacks
            df[c] = pd.NA
    return df.assign(sector=sector, release=release)


def partition(
    table: str, sector: str, release: str, columns: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """One sector x release of one table (cached; do not mutate the result). Requested
    columns the sector does not have come back as nulls."""
    return _partition(table, sector, release, tuple(columns) if columns else None)


def load(
    table: str,
    sectors: list[str],
    release: str | dict[str, str] = "latest",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Rows of `table` for the given sectors, with `sector` and `release` columns added."""
    frames = []
    for s in sectors:
        want = release.get(s, "latest") if isinstance(release, dict) else release
        r, _ = resolve_release(s, want)
        frames.append(partition(table, s, r, columns))
    return pd.concat(frames, ignore_index=True)


def load_many(
    tables: list[str], sectors: list[str], release: str | dict[str, str] = "latest"
) -> dict[str, pd.DataFrame]:
    return {t: load(t, sectors, release) for t in tables}


def clear_cache() -> None:
    _partition.cache_clear()
    columns.cache_clear()


def _meta(sector: str, release: str) -> dict:
    return storage.read_json(storage.join(settings.curated_dir(sector, release), "_meta.json"))


def sector_info(sector: str) -> dict:
    """Row of catalog/sectors.csv (name, primary model, primary weight spec, raking target)."""
    df = storage.read_csv(storage.join(settings.catalog_dir(), "sectors.csv"), dtype=str)
    rows = df[df["sector"] == sector]
    if rows.empty:
        raise KeyError(f"{sector} not in catalog/sectors.csv")
    return rows.iloc[0].to_dict()


def weight_specs(sector: str, release: str) -> dict[str, dict]:
    """{spec: {is_primary, target_source, target_date}}. The primary spec is the `weight`
    column of vacancies; any other spec is `weight_<spec>`."""
    return _meta(sector, release)["files"]["vacancies.parquet"]["weight_specs"]


def weight_column(sector: str, release: str, spec: str = "primary") -> str:
    specs = weight_specs(sector, release)
    if spec == "primary" or specs.get(spec, {}).get("is_primary"):
        return "weight"
    if spec not in specs:
        raise ValueError(f"{sector} {release}: no weight spec {spec!r} (has {sorted(specs)})")
    return f"weight_{spec}"


def thresholds(sector: str, release: str) -> pd.DataFrame:
    """The quadrant thresholds metrics was built with (per model and basis)."""
    return pd.DataFrame(_meta(sector, release)["files"]["metrics.parquet"]["thresholds"])


def sector_columns(sector: str, release: str) -> list[str]:
    """vacancies columns specific to this sector (beyond the core and weight columns)."""
    from .contract import TABLES

    return [
        c
        for c in columns("vacancies", sector, release)
        if c not in TABLES["vacancies"] and not c.startswith("weight_")
    ]


def score_column(sector: str, release: str, model: str) -> str:
    """tasks column holding `model`'s score: exposure_score for the primary model."""
    from .contract import model_column

    if sector_info(sector)["primary_score_model"] == model:
        return "exposure_score"
    col = model_column(model)
    if col not in columns("tasks", sector, release):
        raise ValueError(f"{sector} {release}: no {model} task scores")
    return col


def methodology() -> dict:
    """The methodology published with the data (catalog/methodology.toml), else the local one."""
    uri = storage.join(settings.catalog_dir(), "methodology.toml")
    if storage.exists(uri):
        return tomllib.loads(storage.read_bytes(uri).decode())
    return settings.methodology()
