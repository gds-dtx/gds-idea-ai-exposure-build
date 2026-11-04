"""Sector registry. Adding a sector = copy a folder, edit config.toml and sector.py, add it here."""

from __future__ import annotations

from importlib import import_module

from .base import Sector

_MODULES = {
    "civil-service": "civil_service",
    "nhs": "nhs",
    "education": "education",
    "local-govt": "local_govt",
    "police": "police",
}


def get(slug: str) -> Sector:
    if slug not in _MODULES:
        raise KeyError(f"unknown sector {slug!r}; expected one of {sorted(_MODULES)}")
    return import_module(f".{_MODULES[slug]}.sector", __name__).SECTOR()


def all_slugs() -> list[str]:
    return list(_MODULES)
