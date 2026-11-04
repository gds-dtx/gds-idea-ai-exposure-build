"""Read and write tables on S3 or a local folder, and the `_meta.json` sidecar every folder gets.

All paths are URIs: `s3://bucket/key` or a plain local path. Nothing else in the package
touches boto3 or pyarrow filesystems directly.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from . import settings


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


@cache
def _boto():
    import boto3

    return boto3.Session(region_name=settings.REGION)


def _s3fs() -> pafs.S3FileSystem:
    c = _boto().get_credentials().get_frozen_credentials()
    return pafs.S3FileSystem(
        access_key=c.access_key,
        secret_key=c.secret_key,
        session_token=c.token,
        region=settings.REGION,
    )


def _fs_path(uri: str) -> tuple[pafs.FileSystem, str]:
    if is_s3(uri):
        return _s3fs(), uri.removeprefix("s3://")
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def join(*parts: str) -> str:
    return "/".join(p.strip("/") if i else p.rstrip("/") for i, p in enumerate(parts))


def exists(uri: str) -> bool:
    fs, path = _fs_path(uri)
    return fs.get_file_info(path).type != pafs.FileType.NotFound


def read_parquet(uri: str, columns: list[str] | None = None) -> pd.DataFrame:
    fs, path = _fs_path(uri)
    return pq.read_table(path, filesystem=fs, columns=columns).to_pandas()


def parquet_columns(uri: str) -> list[str]:
    fs, path = _fs_path(uri)
    return pq.ParquetFile(fs.open_input_file(path)).schema_arrow.names


def write_parquet(df: pd.DataFrame, uri: str) -> None:
    fs, path = _fs_path(uri)
    if not is_s3(uri):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, filesystem=fs)


def read_bytes(uri: str) -> bytes:
    if uri.startswith("https://"):  # official statistics fetched straight from the publisher
        import ssl
        import urllib.request

        import truststore  # the OS trust store (uv's Python has none; corporate proxies)

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        req = urllib.request.Request(uri, headers={"User-Agent": "aie-build (DSIT)"})
        with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
            return r.read()
    fs, path = _fs_path(uri)
    with fs.open_input_stream(path) as fh:
        return fh.read()


def write_bytes(data: bytes, uri: str) -> None:
    fs, path = _fs_path(uri)
    if not is_s3(uri):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with fs.open_output_stream(path) as fh:
        fh.write(data)


def read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def write_json(obj: dict, uri: str) -> None:
    write_bytes(json.dumps(obj, indent=2, default=str).encode(), uri)


def read_csv(uri: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_bytes(uri)), **kw)


def write_csv(df: pd.DataFrame, uri: str) -> None:
    write_bytes(df.to_csv(index=False).encode(), uri)


def copy(src: str, dest: str) -> None:
    """Copy one object. S3 -> S3 is a server-side copy (no download)."""
    if is_s3(src) and is_s3(dest):
        sb, sk = _split(src)
        db, dk = _split(dest)
        _boto().client("s3").copy({"Bucket": sb, "Key": sk}, db, dk)
        return
    write_bytes(read_bytes(src), dest)


def list_files(prefix_uri: str) -> list[str]:
    fs, path = _fs_path(prefix_uri)
    sel = pafs.FileSelector(path, recursive=True, allow_not_found=True)
    scheme = "s3://" if is_s3(prefix_uri) else ""
    return sorted(scheme + f.path for f in fs.get_file_info(sel) if f.type == pafs.FileType.File)


# --- provenance -----------------------------------------------------------------------------


@cache
def git_commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return (sha or "uncommitted") + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_table(
    df: pd.DataFrame,
    folder: str,
    name: str,
    *,
    sector: str,
    layer: str,
    release: str,
    inputs: list[str] | None = None,
    extra: dict | None = None,
) -> str:
    """Write `folder/name` and record it in `folder/_meta.json` (merged, one per folder)."""
    uri = join(folder, name)
    if name.endswith(".parquet"):
        write_parquet(df, uri)
    elif name.endswith(".csv"):
        write_csv(df, uri)
    else:
        raise ValueError(f"unsupported table format: {name}")

    meta_uri = join(folder, "_meta.json")
    meta = read_json(meta_uri) if exists(meta_uri) else {"files": {}}
    meta.update(
        {
            "sector": sector,
            "layer": layer,
            "release": release,
            "git_commit": git_commit(),
            "methodology_version": settings.methodology()["version"],
            "updated_utc": datetime.now(UTC).isoformat(),
        }
    )
    meta["files"][name] = {
        "rows": len(df),
        "columns": list(df.columns),
        "inputs": inputs or [],
        **(extra or {}),
    }
    write_json(meta, meta_uri)
    return uri
