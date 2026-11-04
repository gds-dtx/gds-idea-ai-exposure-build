"""Run one build stage for one sector and write it to build/sector=<s>/release=<r>/<layer>/.

  raw        copy source files unchanged into build/sector=<s>/release=<r>/raw/   (L0)
  reference  copy official statistics into build/sector=<s>/release=<r>/reference/
  clean      adverts.parquet, adverts_ext.parquet, advert_text.parquet, funnel.csv   (L1)
  score      tasks.parquet, model_scores.parquet, task_reasoning.parquet             (L2)
  weight     vacancy_weights.parquet, targets.parquet, rake_diagnostics.json         (L3)

Each layer reads only the layer before it (from storage, never from another repo's folder).
"""

from __future__ import annotations

from datetime import date
from pathlib import PurePosixPath

import pandas as pd

from . import catalog, contract, settings, storage
from .sectors import Sector

STAGES = ("raw", "reference", "clean", "score", "weight")


def _write(df, folder, name, sector, layer, release, inputs=None, extra=None) -> None:
    uri = storage.write_table(
        df, folder, name, sector=sector.slug, layer=layer, release=release,
        inputs=inputs, extra=extra,
    )  # fmt: skip
    print(f"  wrote {uri}  ({len(df):,} rows)")


def copy_raw(sector: Sector, release: str) -> list[str]:
    dest_dir = settings.raw_dir(sector.slug, release)
    out = []
    for src in sector.raw_sources():
        dest = storage.join(dest_dir, PurePosixPath(src).name)
        if storage.exists(dest):
            print(f"  exists {dest}")
        else:
            storage.copy(src, dest)
            print(f"  copied {src} -> {dest}")
        out.append(dest)
    if not out:
        print(f"  {sector.slug}: no raw files configured (see config.toml [raw])")
        return out
    meta_uri = storage.join(dest_dir, "_meta.json")
    meta = storage.read_json(meta_uri) if storage.exists(meta_uri) else {"copied": {}}
    for src, dest in zip(sector.raw_sources(), out):
        meta["copied"].setdefault(
            PurePosixPath(dest).name, {"source": src, "on": date.today().isoformat()}
        )
    storage.write_json(meta | {"sector": sector.slug, "release": release, "layer": "raw"}, meta_uri)
    return out


def copy_reference(sector: Sector, release: str) -> None:
    for name, src in sector.reference_sources().items():
        dest = sector.reference(name, release)
        storage.copy(src, dest)
        print(f"  copied {src} -> {dest}")


def clean(sector: Sector, release: str) -> None:
    out = sector.clean(release)
    folder = settings.build_dir(sector.slug, "clean", release)
    adverts = contract.conform(out.adverts, "adverts")
    _write(adverts, folder, "adverts.parquet", sector, "clean", release, out.inputs)
    ext = contract.conform_ext(out.ext)
    _write(ext, folder, "adverts_ext.parquet", sector, "clean", release, out.inputs)
    if out.text is not None:
        _write(out.text, folder, "advert_text.parquet", sector, "clean", release, out.inputs)
    if out.funnel is not None:
        _write(out.funnel, folder, "funnel.csv", sector, "clean", release, out.inputs)
    catalog.record_stage(sector, release, "clean")


def score(sector: Sector, release: str) -> None:
    out = sector.score(release)
    folder = settings.build_dir(sector.slug, "scored", release)
    tasks = contract.conform(out.tasks, "task_scores")
    _write(tasks, folder, "tasks.parquet", sector, "scored", release, out.inputs)
    ms = contract.conform(out.model_scores, "model_scores")
    _write(ms, folder, "model_scores.parquet", sector, "scored", release, out.inputs)
    if out.reasoning is not None:
        _write(
            out.reasoning, folder, "task_reasoning.parquet", sector, "scored", release, out.inputs
        )
    catalog.record_stage(sector, release, "scored")


def n_tasks_by_vacancy(sector: Sector, release: str) -> pd.Series:
    tasks = storage.read_parquet(
        storage.join(settings.build_dir(sector.slug, "scored", release), "tasks.parquet"),
        columns=["vacancy_id"],
    )
    return tasks.groupby("vacancy_id").size()


def weight(sector: Sector, release: str) -> None:
    clean_dir = settings.build_dir(sector.slug, "clean", release)
    adverts = storage.read_parquet(storage.join(clean_dir, "adverts.parquet"))
    ext = storage.read_parquet(storage.join(clean_dir, "adverts_ext.parquet"))
    out = sector.weight(release, adverts, ext, n_tasks_by_vacancy(sector, release))
    folder = settings.build_dir(sector.slug, "weights", release)
    w = contract.conform(out.weights, "weights")
    inputs = [clean_dir, settings.build_dir(sector.slug, "scored", release), *out.inputs]
    _write(w, folder, "vacancy_weights.parquet", sector, "weights", release, inputs)
    if out.targets is not None:
        _write(out.targets, folder, "targets.parquet", sector, "weights", release, inputs)
    storage.write_json(out.diagnostics, storage.join(folder, "rake_diagnostics.json"))
    for spec, g in w.groupby("weight_spec"):
        print(f"  {spec}: {g['weight'].notna().sum():,} weighted, total {g['weight'].sum():,.0f}")
    catalog.record_stage(sector, release, "weights")


def run(sector: Sector, stage: str, release: str) -> None:
    print(f"[{sector.slug}] {stage} release={release}")
    sector.check_release(release)
    if stage == "raw":
        copy_raw(sector, release)
    elif stage == "reference":
        copy_reference(sector, release)
    elif stage == "clean":
        clean(sector, release)
    elif stage == "score":
        score(sector, release)
    elif stage == "weight":
        weight(sector, release)
    else:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
