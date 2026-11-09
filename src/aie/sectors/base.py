"""What every sector provides, and the helpers they share.

A sector is a folder under `aie/sectors/` with:
  config.toml   names, sources, model, weighting spec, raking targets, LLM system prompts
  sector.py     a `Sector` subclass: clean(), score(), weight()
  README.md     funnel, mapping decisions, known caveats

Each stage returns plain DataFrames in the standard shape; `aie.build` writes them to
build/sector=<s>/release=<r>/<layer>/ and `aie.publish` turns them into the curated tables.

v1 of every sector ADOPTS the outputs that already exist (read-only, from the pre-scaffold
S3 folders) and maps them onto the contract; nothing is re-scored. Folding each old repo's
clean-from-raw code in, so v2 can be rebuilt from raw, happens once this approach is agreed.
"""

from __future__ import annotations

import os
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .. import contract, settings, storage


@dataclass
class CleanOutput:
    adverts: pd.DataFrame  # contract "adverts" columns (n_tasks filled at publish)
    ext: pd.DataFrame  # vacancy_id + sector-specific columns (become extra vacancies columns)
    text: pd.DataFrame | None = None  # vacancy_id, advert_text (input to scoring)
    funnel: pd.DataFrame | None = None  # step, rows
    inputs: list[str] = field(default_factory=list)


@dataclass
class ScoreOutput:
    tasks: pd.DataFrame  # contract "task_scores" columns
    model_scores: pd.DataFrame  # contract "model_scores" columns (long)
    reasoning: pd.DataFrame | None = None  # vacancy_id, task_index, reasoning, label_reasoning
    inputs: list[str] = field(default_factory=list)


@dataclass
class WeightOutput:
    weights: pd.DataFrame  # vacancy_weights columns
    targets: pd.DataFrame | None = None  # spec, dimension, category, target_fte
    diagnostics: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


class Sector(ABC):
    slug: str
    package_dir: Path

    def __init__(self) -> None:
        with (self.package_dir / "config.toml").open("rb") as fh:
            self.config = tomllib.load(fh)

    # --- identity ---------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def primary_score_model(self) -> str:
        return self.config["scoring"]["primary_model"]

    @property
    def primary_weight_spec(self) -> str:
        return self.config["weighting"]["primary_spec"]

    def legacy(self, key: str) -> str:
        """URI of a pre-scaffold file this sector's v1 adopts (read-only)."""
        return resolve_legacy(self.config["legacy"][key])

    def reference(self, name: str, release: str) -> str:
        return storage.join(settings.reference_dir(self.slug, release), name)

    def check_release(self, release: str) -> None:
        """Refuse to build a release the config marks as data-only (or does not list)."""
        releases = self.config.get("releases")
        if releases is None:
            return
        how = releases.get(release)
        if how is None:
            raise ValueError(
                f"{self.slug}: release {release!r} not in config [releases] {releases}"
            )
        if how == "legacy":
            raise ValueError(
                f"{self.slug} {release} is a legacy release: its data is kept under "
                f"build/sector={self.slug}/release={release}/ but it is not built by this repo"
            )

    def weights_on_salaried(self, release: str) -> bool:
        """True if this release shares each raking target over adverts that state a salary
        only (config.toml [release_settings.<release>] weight_on_salaried), so £ totals scale
        up from salaried adverts. See aie.build.weight."""
        rs = self.config.get("release_settings", {}).get(release, {})
        return bool(rs.get("weight_on_salaried", False))

    def ids(self, s: pd.Series) -> pd.Series:
        return contract.prefixed_id(self.slug, s)

    # --- stages -----------------------------------------------------------------------------
    def raw_sources(self) -> list[str]:
        """Files copied unchanged into build/sector=<s>/raw/ (L0)."""
        return [resolve_legacy(p) for p in self.config.get("raw", {}).get("files", [])]

    def reference_sources(self) -> dict[str, str]:
        """{reference file name: source URI} copied into build/sector=<s>/reference/."""
        return {k: resolve_legacy(v) for k, v in self.config.get("reference_sources", {}).items()}

    @abstractmethod
    def clean(self, release: str) -> CleanOutput: ...

    @abstractmethod
    def score(self, release: str) -> ScoreOutput: ...

    @abstractmethod
    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        """L3. `n_tasks` is indexed by vacancy_id (from the scored layer)."""

    # --- shared helpers ---------------------------------------------------------------------
    def unweighted(self, adverts: pd.DataFrame, n_tasks: pd.Series, reason: str) -> WeightOutput:
        """Placeholder weights: each in-scope advert with enough tasks counts its headcount.

        Totals built on these are SAMPLE totals (adverts), not workforce totals.
        """
        min_tasks = settings.methodology()["population"]["min_tasks"]
        eligible = adverts["in_scope"].fillna(False).astype(bool) & (
            adverts["vacancy_id"].map(n_tasks).fillna(0) >= min_tasks
        )
        w = adverts["headcount"].astype("float64").where(eligible)
        return WeightOutput(
            weights=pd.DataFrame(
                {
                    "vacancy_id": adverts["vacancy_id"],
                    "weight_spec": "unweighted",
                    "weight": w,
                    "is_primary": True,
                    "target_source": f"none: {reason}",
                    "target_date": pd.NA,
                }
            ),
            diagnostics={
                "spec": "unweighted",
                "reason": reason,
                "n_weighted": int(w.notna().sum()),
                "weighted_total": float(w.sum()),
            },
        )

    def min_tasks_sample(self, adverts: pd.DataFrame, n_tasks: pd.Series) -> pd.DataFrame:
        """In-scope adverts with at least MIN_TASKS tasks: the raking donor pool."""
        min_tasks = settings.methodology()["population"]["min_tasks"]
        a = adverts[adverts["in_scope"].fillna(False).astype(bool)].copy()
        a["n_tasks"] = a["vacancy_id"].map(n_tasks)
        return a[a["n_tasks"] >= min_tasks].reset_index(drop=True)


def resolve_legacy(path: str) -> str:
    """Config paths are relative to the pre-scaffold S3 root unless absolute, s3:// or ~/."""
    if "://" in path or path.startswith("/"):
        return path
    if path.startswith("~"):
        return os.path.expanduser(path)
    return f"{settings.LEGACY_ROOT}/{path}"


def adopt_labelled_tasks(sector: Sector, run_id: str) -> ScoreOutput:
    """Adopt a pre-scaffold labelled_tasks_full.parquet (Education/LG/police layout)."""
    src = sector.legacy("tasks")
    t = storage.read_parquet(src)
    vid = sector.ids(t["vacancy_id"])
    model = sector.primary_score_model
    tasks = pd.DataFrame(
        {
            "vacancy_id": vid,
            "task_index": t["task_index"],
            "task_text": t["task"],
            "exposure_score": t["automation_potential"],
            "score_model": model,
            "task_label": t["label"],
            "scoring_run_id": run_id,
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


# --- parsing helpers used by more than one sector -------------------------------------------


def parse_money(s: pd.Series) -> pd.Series:
    """First number in a free-text salary field ('£24,561 per annum' -> 24561.0)."""
    txt = s.astype("string").str.replace("£", "", regex=False)
    num = txt.str.extract(r"([\d,]+(?:\.\d+)?)")[0].str.replace(",", "", regex=False)
    return pd.to_numeric(num, errors="coerce")


def advert_midpoint_salary(lo: pd.Series, hi: pd.Series, floor=12_000, cap=250_000) -> pd.Series:
    """Midpoint of an advertised range (or whichever end is present), annual £ only.

    Values outside [floor, cap] are treated as hourly/daily rates or errors and dropped.
    """
    lo_n, hi_n = parse_money(lo), parse_money(hi)
    mid = pd.concat([lo_n, hi_n], axis=1).mean(axis=1, skipna=True)
    return mid.where((mid >= floor) & (mid <= cap))


def year_of(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce").dt.year.astype("Int64")


def to_bool(s: pd.Series) -> pd.Series:
    return s.map({True: True, False: False, "True": True, "False": False}).astype("boolean")


def no_funnel(n: int, label: str = "adopted from pre-scaffold output") -> pd.DataFrame:
    return pd.DataFrame({"step": [label], "rows": [n]})
