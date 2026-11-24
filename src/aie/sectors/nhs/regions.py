"""Postcode area -> NHS England region, for the region dimension of the NHS rake.

Ported from gds-idea-nhs-ai-exposure-raking (scripts/build_region_lookup.py), which wrote the
same table to a CSV that earlier releases read from reference/. It is now built here.

Each English postcode area (the letter prefix, e.g. 'LS', 'SW') is assigned to its dominant ONS
region, and the 9 ONS regions collapse exactly into the 7 NHS England regions:

    North East + Yorkshire and The Humber -> North East and Yorkshire
    East Midlands + West Midlands          -> Midlands
    (the other five map one-to-one)

This is an area-level approximation: a few areas straddle a regional boundary (mostly around
London), so some adverts get the neighbouring region. The rake still matches every region total
exactly; only the mix within a region is affected. A full-postcode lookup (ONS Postcode
Directory, `pcds` -> `rgn`) would remove the approximation.

Welsh, Scottish, Northern Irish, Channel Islands and Isle of Man areas are not listed: those
adverts are outside England and out of scope.
"""

from __future__ import annotations

import pandas as pd

# English postcode areas by ONS region (dominant assignment).
AREA_BY_ONS_REGION: dict[str, list[str]] = {
    "North East": ["DH", "DL", "NE", "SR", "TS"],
    "North West": ["BB", "BL", "CA", "CH", "CW", "FY", "L", "LA", "M", "OL",
                   "PR", "SK", "WA", "WN"],
    "Yorkshire and The Humber": ["BD", "DN", "HD", "HG", "HU", "HX", "LS", "S", "WF", "YO"],
    "East Midlands": ["DE", "LE", "LN", "NG", "NN"],
    # SY straddles Shropshire (England) and Powys (Wales); NHS England adverts there are
    # Shrewsbury/Telford, so it is assigned to the Midlands.
    "West Midlands": ["B", "CV", "DY", "HR", "ST", "SY", "TF", "WR", "WS", "WV"],
    "East of England": ["AL", "CB", "CM", "CO", "IP", "LU", "NR", "PE", "SG", "SS", "WD"],
    "London": ["BR", "CR", "E", "EC", "EN", "HA", "IG", "N", "NW", "RM", "SE",
               "SM", "SW", "TW", "UB", "W", "WC"],
    "South East": ["BN", "CT", "DA", "GU", "HP", "KT", "ME", "MK", "OX", "PO",
                   "RG", "RH", "SL", "SO", "TN"],
    "South West": ["BA", "BH", "BS", "DT", "EX", "GL", "PL", "SN", "SP", "TA", "TQ", "TR"],
}  # fmt: skip

# 9 ONS regions -> 7 NHS England regions (exact).
REGION_TO_NHSE = {
    "North East": "North East and Yorkshire",
    "Yorkshire and The Humber": "North East and Yorkshire",
    "East Midlands": "Midlands",
    "West Midlands": "Midlands",
    "North West": "North West",
    "East of England": "East of England",
    "London": "London",
    "South East": "South East",
    "South West": "South West",
}


def lookup() -> pd.DataFrame:
    """One row per English postcode area: area, ons_region, nhse_region."""
    rows = [
        {"area": area, "ons_region": ons, "nhse_region": REGION_TO_NHSE[ons]}
        for ons, areas in AREA_BY_ONS_REGION.items()
        for area in areas
    ]
    df = pd.DataFrame(rows).sort_values("area").reset_index(drop=True)
    if df["area"].duplicated().any():
        raise ValueError("a postcode area maps to more than one region")
    return df


def area_to_nhse_region() -> dict[str, str]:
    lk = lookup()
    return dict(zip(lk["area"], lk["nhse_region"], strict=True))
