"""Where the Civil Service Statistics tables are read from, for cs_stats / rake / salary.

The original modules read constants.RAW_DATA_DIR and the URL table from constants.py. Here
`constants` carries the same names; the sector points STATS_DIR at a local copy of the
release's reference/ files before calling them (use_stats_dir).
"""

from __future__ import annotations

from types import SimpleNamespace

from .mappings import (
    COVERAGE_THRESHOLD_3D,
    CS_STATS_BROWSER_URLS,
    DEPARTMENTS,
    EXTRAPOLATION_THRESHOLD,
)

constants = SimpleNamespace(
    RAW_DATA_DIR=None,
    CS_STATS_BROWSER_URLS=CS_STATS_BROWSER_URLS,
    DEPARTMENTS=DEPARTMENTS,
    EXTRAPOLATION_THRESHOLD=EXTRAPOLATION_THRESHOLD,
    COVERAGE_THRESHOLD_3D=COVERAGE_THRESHOLD_3D,
)


def use_stats_dir(path: str) -> None:
    """Read the statistics tables (organisations.csv, grades.csv, ...) from `path`."""
    constants.RAW_DATA_DIR = str(path)
