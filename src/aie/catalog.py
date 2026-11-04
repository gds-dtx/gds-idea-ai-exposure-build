"""catalog/: what exists and what is current.

  sectors.csv        one row per sector (from each config.toml)
  releases.csv       one row per sector x release: status, stage timestamps, provenance
  cuts_manifest.csv  one row per saved cut, across sectors (written by the pipeline repo)
  methodology.toml   the methodology the published metrics were built with

Release status: draft (being built) -> published (signed off) -> superseded. Analysis asks for
"latest published"; if a sector has none yet, readers fall back to the latest draft and say so.
`legacy` marks a release kept as data only (CS v1): never curated, never picked as "latest".
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from . import settings, storage

RELEASE_COLS = [
    "sector", "release", "status", "methodology_version", "primary_score_model",
    "primary_weight_spec", "clean_at", "scored_at", "weights_at", "published_at",
    "git_commit", "notes",
]  # fmt: skip
STATUSES = ("draft", "published", "superseded", "legacy")


def _uri(name: str) -> str:
    return storage.join(settings.catalog_dir(), name)


def releases() -> pd.DataFrame:
    uri = _uri("releases.csv")
    if not storage.exists(uri):
        return pd.DataFrame(columns=RELEASE_COLS)
    return storage.read_csv(uri, dtype=str).reindex(columns=RELEASE_COLS)


def record_stage(sector, release: str, stage: str, notes: str | None = None) -> None:
    """Upsert the (sector, release) row and stamp `<stage>_at`."""
    df = releases()
    key = (df["sector"] == sector.slug) & (df["release"] == release)
    if not key.any():
        row = {c: pd.NA for c in RELEASE_COLS} | {
            "sector": sector.slug,
            "release": release,
            "status": "draft",
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        key = (df["sector"] == sector.slug) & (df["release"] == release)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    df.loc[key, f"{stage}_at"] = now
    df.loc[key, "methodology_version"] = settings.methodology()["version"]
    df.loc[key, "primary_score_model"] = sector.primary_score_model
    df.loc[key, "primary_weight_spec"] = sector.primary_weight_spec
    df.loc[key, "git_commit"] = storage.git_commit()
    if notes:
        df.loc[key, "notes"] = notes
    storage.write_csv(df, _uri("releases.csv"))


def record_legacy(sector: str, release: str, notes: str) -> None:
    """Add (or update) a data-only release row."""
    df = releases()
    df = df[~((df["sector"] == sector) & (df["release"] == release))]
    row = {c: pd.NA for c in RELEASE_COLS} | {
        "sector": sector, "release": release, "status": "legacy", "notes": notes,
    }  # fmt: skip
    storage.write_csv(pd.concat([df, pd.DataFrame([row])], ignore_index=True), _uri("releases.csv"))


def drop_release(sector: str, release: str) -> None:
    """Remove a row (e.g. a release that was registered under the wrong name)."""
    df = releases()
    storage.write_csv(
        df[~((df["sector"] == sector) & (df["release"] == release))], _uri("releases.csv")
    )


def set_status(sector: str, release: str, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    df = releases()
    key = (df["sector"] == sector) & (df["release"] == release)
    if not key.any():
        raise KeyError(f"no release {sector}/{release} in the catalog")
    if status == "published":
        older = (df["sector"] == sector) & (df["status"] == "published") & ~key
        df.loc[older, "status"] = "superseded"
    df.loc[key, "status"] = status
    storage.write_csv(df, _uri("releases.csv"))


def resolve_release(sector: str, release: str = "latest") -> tuple[str, str | None]:
    """(release, warning). 'latest' = newest published, else newest draft (with a warning)."""
    if release != "latest":
        return release, None
    df = releases()
    df = df[(df["sector"] == sector) & df["published_at"].notna()]
    if df.empty:
        raise KeyError(f"{sector}: nothing published to curated/ yet")
    pub = df[df["status"] == "published"]
    if len(pub):
        return pub.sort_values("published_at")["release"].iloc[-1], None
    r = df.sort_values("published_at")["release"].iloc[-1]
    return r, f"{sector}: no signed-off release yet, using draft {r}"


def write_sectors_table(sectors: list) -> None:
    rows = [
        {
            "sector": s.slug,
            "name": s.name,
            "source": s.config["source"],
            "owner": s.config.get("owner", ""),
            "primary_score_model": s.primary_score_model,
            "primary_weight_spec": s.primary_weight_spec,
            "raking_target": s.config["weighting"].get(
                "target_source", s.config["weighting"].get("placeholder_reason", "")
            ),
        }
        for s in sectors
    ]
    storage.write_csv(pd.DataFrame(rows), _uri("sectors.csv"))


def append_cut(row: dict) -> None:
    uri = _uri("cuts_manifest.csv")
    df = storage.read_csv(uri, dtype=str) if storage.exists(uri) else pd.DataFrame()
    storage.write_csv(pd.concat([df, pd.DataFrame([row])], ignore_index=True), uri)


def write_methodology() -> None:
    """Publish methodology.toml beside the data, so readers need not install this repo."""
    storage.write_bytes(settings.methodology_path().read_bytes(), _uri("methodology.toml"))
