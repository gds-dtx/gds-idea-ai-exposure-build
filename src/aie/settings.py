"""Where things live, and the methodology settings.

Two roots, both overridable by environment variable so tests and local runs can point at a
folder instead of S3:

  AIE_ROOT        s3://gds-idea-econ/ai-exposure-updated-scaffold   (this project's data)
  AIE_BATCH_ROOT  s3://gds-idea-econ/batch/ai-exposure              (Bedrock batch scratch)

Batch scratch sits OUTSIDE the project folder, under the bucket-wide `batch/` prefix, so one
lifecycle rule can expire batch I/O for every project.
"""

from __future__ import annotations

import copy
import os
import tomllib
from functools import cache
from pathlib import Path

BUCKET = "gds-idea-econ"
REGION = "eu-west-2"

DEFAULT_ROOT = f"s3://{BUCKET}/ai-exposure-updated-scaffold"
DEFAULT_BATCH_ROOT = f"s3://{BUCKET}/batch/ai-exposure"

# Read-only: where the pre-scaffold outputs live. Only the v1 `adopt` steps read from here.
LEGACY_ROOT = f"s3://{BUCKET}/ai-exposure"

SECTORS = ("civil-service", "nhs", "education", "local-govt", "police")


def root() -> str:
    return os.environ.get("AIE_ROOT", DEFAULT_ROOT).rstrip("/")


def batch_root() -> str:
    return os.environ.get("AIE_BATCH_ROOT", DEFAULT_BATCH_ROOT).rstrip("/")


def methodology_path() -> Path:
    override = os.environ.get("AIE_METHODOLOGY")
    if override:
        return Path(override)
    repo_copy = Path(__file__).resolve().parents[2] / "methodology.toml"
    return repo_copy if repo_copy.exists() else Path(__file__).with_name("methodology.toml")


@cache
def _methodology_file() -> dict:
    with methodology_path().open("rb") as fh:
        return tomllib.load(fh)


METHOD_SECTIONS = ("decay", "population", "quadrants", "valuation", "comparability")


def methodology(version: str | None = None) -> dict:
    """The current methodology (decay, MIN_TASKS, thresholds, valuation rules), or a past
    version's full settings from its [history.<version>] entry in methodology.toml."""
    current = _methodology_file()
    if version in (None, current["version"]):
        return current
    past = current.get("history", {}).get(version)
    if past is None:
        known = [current["version"], *current.get("history", {})]
        raise ValueError(f"unknown methodology version {version!r} (known: {known})")
    return {**{k: current[k] for k in METHOD_SECTIONS}, **past, "version": version}


# Settings that move the CS reference distribution the fixed thresholds are taken from.
REDERIVE_FIXED = {"decay.primary", "quadrants.high_mean_top", "quadrants.disp_low_bottom"}


def with_changes(changes: dict, label: str, base: dict | None = None) -> dict:
    """`base` (default: current) with `changes` applied, for a sensitivity scenario.

    `changes` uses dotted keys, e.g. {"decay.primary": 0.9, "quadrants.high_mean_top": 0.3}.
    The result's version is "<base>+<label>", so its numbers can never pass for the primary.

    Fixed thresholds ("option b", agreed 2026-10-28): they are CS's ensemble 1D quartiles at
    the primary decay. A change to decay or to the quartile percentiles therefore re-derives
    them the same way and applies them to every sector, unless the scenario sets fixed_*
    itself. Population settings (MIN_TASKS) act before raking, so they need a new release.
    """
    m = copy.deepcopy(base or methodology())
    for key, value in changes.items():
        section, _, name = key.partition(".")
        if section == "population":
            raise ValueError(f"{key} acts before raking: build a new release instead")
        if section not in METHOD_SECTIONS or name not in m[section]:
            raise ValueError(f"unknown methodology setting {key!r}")
        m[section][name] = value
    if REDERIVE_FIXED & set(changes) and not any(".fixed_" in k for k in changes):
        m["quadrants"]["fixed_source"] = "rederive"
    m["version"] = f"{m['version']}+{label}"
    m.pop("history", None)
    m.pop("changelog", None)
    return m


# --- S3 layout ------------------------------------------------------------------------------
# Both trees read the same way: pick the sector, then the release, then the files. Raw and
# reference files belong to a release, because each release is built from its own inputs.
#   build/sector=<s>/release=<r>/{raw,reference,clean,scored,weights}/
#   curated/sector=<s>/release=<r>/{vacancies,tasks,metrics}.parquet


def sector_build_dir(sector: str) -> str:
    return f"{root()}/build/sector={sector}"


def release_build_dir(sector: str, release: str) -> str:
    return f"{sector_build_dir(sector)}/release={release}"


def raw_dir(sector: str, release: str) -> str:
    """L0: the files this release was built from, exactly as received."""
    return f"{release_build_dir(sector, release)}/raw"


def reference_dir(sector: str, release: str) -> str:
    """Official statistics this release uses (raking targets, pay tables)."""
    return f"{release_build_dir(sector, release)}/reference"


def build_dir(sector: str, layer: str, release: str) -> str:
    assert layer in ("clean", "scored", "weights"), layer
    return f"{release_build_dir(sector, release)}/{layer}"


def curated_dir(sector: str, release: str) -> str:
    return f"{root()}/curated/sector={sector}/release={release}"


def curated_file(table: str, sector: str, release: str) -> str:
    return f"{curated_dir(sector, release)}/{table}.parquet"


def cuts_dir(scope: str) -> str:
    return f"{root()}/cuts/{scope}"


def catalog_dir() -> str:
    return f"{root()}/catalog"


def batch_dir(owner: str, job_group: str) -> str:
    """Batch scratch for one job group, e.g. batch_dir('nhs', 'v2-extract')."""
    return f"{batch_root()}/{owner}/{job_group}"
