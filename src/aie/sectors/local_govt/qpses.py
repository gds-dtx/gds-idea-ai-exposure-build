"""Raking targets from the ONS Quarterly Public Sector Employment Survey (QPSES), and the
local government rake.

Source: Local Government Association, "ONS Quarterly Public Sector Employment Survey"
(https://www.local.gov.uk/publications/ons-quarterly-public-sector-employment-survey). Each
quarter the LGA publishes the ONS returns for every English and Welsh principal authority
as an xlsx. We use the "Final Individual" sheet: one row per authority with LA number, name,
type, region, headcount, FTE and sex / working pattern / contract splits. The exact file,
quarter and publication date are in config.toml [weighting] and in docs/qpses.md.

Scope, as far as the file shows:
  * England and Wales principal councils only (no Scottish or NI councils, no town or parish
    councils, no fire, police or combined authorities).
  * It appears to INCLUDE staff in council-maintained schools (Hampshire 35k, Leeds 25k and
    Kent 22k are too large for non-school staff), and so excludes academies. The LGA file
    does not say so; this is inferred. School-type adverts therefore stay in the rake.
  * Some authorities report under pre-reorganisation names (Cumbria and its six districts
    for Cumberland + Westmorland and Furness) or report 0 because a partner employs their
    staff (Babergh via Mid Suffolk, Wandsworth via Richmond, West Oxfordshire and Forest of
    Dean via Publica). Shared-staffing councils are pooled into one unit, below.

Matching: an advert's council string is normalised (team prefixes such as "Tenancy
Services - Newcastle City Council" dropped, "council", "borough", "city" etc. removed) and
matched to a QPSES name, then ALIASES, then "A and B" joint employers. Councils that ever
appear together (a joint employer, a shared service or a reorganised area) form one raking
UNIT whose target is the sum of their headcounts, so no member is counted twice.

The rake (rake_council): 1D post-stratification to QPSES **FTE**, not headcount. Salaries
are full-time annual, and NHS and Education rake to FTE too, so paybill and £ figures stay
comparable. (Weighting to headcount overstated paybill by about 28%, since 46% of council
staff are part-time.) Cells:
  * a council (unit) with at least `min_unit_adverts` eligible adverts is its own cell;
  * every other council, including those with no adverts, is pooled by region x authority
    type, then by region if that is still thin. A council's few adverts are skewed to senior
    posts, so they must not stand for the whole council alone;
  * a pool with no adverts at all (a council with none, and no small neighbours of its type)
    has its FTE shared out across the own-cell councils of its region and type, in
    proportion to their FTE. They keep their own cells, so each council's adverts still stand
    for that council's size (release v2; v1 pooled the whole region and type instead).
Totals cover ENGLAND's councils (config `regions`), like NHS and Education. Welsh councils are
in QPSES but our ~300 Welsh adverts are mostly senior posts, so Wales is left unweighted.

Known limitation: council cells fix WHERE the workforce is, not WHAT it does. Adverts skew to
professional and senior posts, so the weighted mean salary is well above
local government's actual average. An occupation or pay margin (APS / ASHE) is the next step.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from ... import storage
from ...methods import rake
from ..base import WeightOutput

SHEET = "Final Individual"
COLUMNS = {
    "LA number": "la_number",
    "LGA Name": "qpses_name",
    "Type": "authority_type",
    "Region": "region",
    "Total Headcount": "headcount",
    "Total FTE": "fte",
    "Notes": "qpses_note",
}

_DROP = (
    r"\b(the|council|councils|county|city|metropolitan|metropolitian|borough|district|"
    r"of|royal|london|mbc|cbc|bc|dc|cc|met|unitary|authority|jobsite|jobs)\b"
)

CUMBRIA = [
    "Cumbria", "Allerdale", "Barrow-in-Furness", "Carlisle", "Copeland", "Eden",
    "South Lakeland",
]  # fmt: skip
NORTH_YORKSHIRE_DISTRICTS = [
    "craven", "hambleton", "harrogate", "richmondshire", "ryedale", "scarborough", "selby",
]  # fmt: skip
SOMERSET_DISTRICTS = ["mendip", "sedgemoor", "south somerset", "somerset west and taunton"]

# normalised advert key -> QPSES names. Names that changed, abbreviations, and pre-2023
# districts whose adverts now belong to the new unitary.
ALIASES: dict[str, list[str]] = {
    "newcastle": ["Newcastle upon Tyne"],
    "hull": ["Kingston upon Hull"],
    "richmond": ["Richmond upon Thames"],
    "st albans and": ["St Albans"],
    "st helens": ["St. Helens"],
    "vale glamorgan": ["The Vale of Glamorgan"],
    "uttlesford essex": ["Uttlesford"],
    "cumberland": CUMBRIA,
    "westmorland and furness": CUMBRIA,
    **{k: ["North Yorkshire"] for k in NORTH_YORKSHIRE_DISTRICTS},
    **{k: ["Somerset"] for k in SOMERSET_DISTRICTS},
}


def normalise(name: str) -> str:
    s = str(name).split(" - ")[-1].lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", lambda m: " " + m.group(0)[1:-1] + " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(_DROP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load(path: str) -> pd.DataFrame:
    """One row per authority from the LGA's QPSES workbook."""
    raw = pd.read_excel(io.BytesIO(storage.read_bytes(path)), sheet_name=SHEET)
    q = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna(subset=["authority_type"])
    q["la_number"] = q["la_number"].astype(int)
    q["headcount"] = pd.to_numeric(q["headcount"]).astype(float)
    q["fte"] = pd.to_numeric(q["fte"]).astype(float)
    q["key"] = q["qpses_name"].map(normalise)
    if q["key"].duplicated().any():
        raise ValueError(f"QPSES names collide after normalising: {q[q.key.duplicated()]}")
    return q.reset_index(drop=True)


def match(councils: pd.Series, q: pd.DataFrame) -> pd.Series:
    """QPSES names (a list, empty if none) for each advert council string."""
    by_key = dict(zip(q["key"], q["qpses_name"]))

    def one(name: str) -> list[str]:
        k = normalise(name)
        if k in by_key:
            return [by_key[k]]
        if k in ALIASES:
            return ALIASES[k]
        parts = [p.strip() for p in re.split(r"\band\b", k) if p.strip()]
        if len(parts) > 1:  # a joint employer, e.g. "Bromsgrove and Redditch Councils"
            found = [by_key.get(p) or (ALIASES.get(p) or [None])[0] for p in parts]
            if all(found):
                return sorted(set(found))
        return []

    uniq = pd.Series(councils.dropna().unique())
    lookup = dict(zip(uniq, uniq.map(one)))
    return pd.Series(
        [lookup.get(c, []) if isinstance(c, str) else [] for c in councils],
        index=councils.index,
        dtype=object,
    )


# A joint advert pools its councils only if joint adverts are at least this share of all the
# members' adverts (a real shared workforce), or a member reports 0 headcount. Otherwise the
# few joint adverts are left unweighted rather than merging two large councils.
MIN_JOINT_SHARE = 0.10


def units(matches: pd.Series, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pool councils into raking units. Returns (one row per QPSES authority with its `unit`,
    e.g. "Bromsgrove + Redditch"; and per advert the names it counts towards, empty where a
    joint advert was not pooled)."""
    parent = {n: n for n in q["qpses_name"]}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def join(names):
        for other in names[1:]:
            parent[find(other)] = find(names[0])

    alias_groups = {tuple(v) for v in ALIASES.values() if len(v) > 1}
    for g in alias_groups:  # reorganised areas reported under old names: always one unit
        join(list(g))
    keys = pd.Series([tuple(sorted(n)) for n in matches], index=matches.index, dtype=object)
    single = keys[keys.map(len) == 1].map(lambda k: k[0]).value_counts()
    zero = set(q.loc[q["headcount"] == 0, "qpses_name"])
    pooled = set()
    for k, n_joint in keys[keys.map(len) > 1].value_counts().items():
        if k in {tuple(sorted(g)) for g in alias_groups}:
            pooled.add(k)
            continue
        n_all = n_joint + sum(single.get(x, 0) for x in k)
        if n_joint / n_all >= MIN_JOINT_SHARE or zero & set(k):
            join(list(k))
            pooled.add(k)
    counts_towards = pd.Series(
        [list(k) if len(k) == 1 or k in pooled else [] for k in keys],
        index=matches.index,
        dtype=object,
    )
    out = q.copy()
    root = out["qpses_name"].map(find)
    members = out.groupby(root)["qpses_name"].apply(lambda s: " + ".join(sorted(s)))
    out["unit"] = root.map(members)
    return out, counts_towards


def unit_table(u: pd.DataFrame) -> pd.DataFrame:
    """One row per raking unit: headcount and FTE, and the region and type of its largest
    member (a county + districts unit such as Cumbria takes the county's)."""
    g = u.groupby("unit")
    t = g.agg(headcount=("headcount", "sum"), fte=("fte", "sum"), n_authorities=("unit", "size"))
    largest = u.sort_values("headcount", ascending=False).drop_duplicates("unit").set_index("unit")
    t["region"] = largest["region"]
    t["authority_type"] = largest["authority_type"]
    return t.reset_index()


def assign(councils: pd.Series, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per advert: qpses_unit, region, authority_type (NA where no QPSES match). Plus the
    unit table. Used by clean (to fill region) and by weight (the targets)."""
    u, m = units(match(councils, q), q)
    ut = unit_table(u)
    name_to_unit = dict(zip(u["qpses_name"], u["unit"]))
    unit = m.map(lambda names: name_to_unit[names[0]] if len(names) else pd.NA).astype("string")
    by_unit = ut.set_index("unit")
    per = pd.DataFrame(
        {
            "qpses_unit": unit,
            "region": unit.map(by_unit["region"]).astype("string"),
            "authority_type": unit.map(by_unit["authority_type"]).astype("string"),
        },
        index=councils.index,
    )
    return per, ut


def strata(ut: pd.DataFrame, n_adverts: pd.Series, min_adverts: int) -> pd.Series:
    """Raking cell per unit. A unit with at least `min_adverts` eligible adverts and some FTE
    is its own cell. Every other unit (few adverts, none, or zero FTE) is pooled with the
    others of its region and authority type, then of its region if that is still thin, so
    one advert never stands for a whole council and councils with no adverts are still
    represented by similar councils."""
    n = ut["unit"].map(n_adverts).fillna(0)
    own = (n >= min_adverts) & (ut["fte"] > 0)
    cell = pd.Series(ut["unit"].where(own), index=ut.index, dtype=object)
    region_type = ut["region"] + " | " + ut["authority_type"] + " (pooled)"
    cell = cell.fillna(region_type)
    # A pool with no adverts at all (e.g. a met district with none, whose neighbours of the
    # same type all have their own cells) is shared out over those neighbours: see share_out.
    empty = n.groupby(cell).transform("sum").eq(0) & ~own
    has_own = region_type.isin(set(region_type[own]))
    shared = empty & has_own
    cell = cell.mask(shared, ut["region"] + " | " + ut["authority_type"] + SHARED)
    pooled_n = n.groupby(cell).transform("sum")
    thin = ~own & ~shared & (pooled_n < min_adverts)
    cell = cell.mask(thin, ut["region"] + " (pooled)")
    return cell


SHARED = " (shared out)"


def share_out(ut: pd.DataFrame, measure: str) -> tuple[pd.Series, dict]:
    """Cell targets, with each "(shared out)" cell's FTE added to the own-cell councils of the
    same region and type in proportion to their FTE. Returns (targets, what was shared)."""
    targets = ut.groupby("cell")[measure].sum()
    detail = {}
    for c in [c for c in targets.index if c.endswith(SHARED)]:
        members = ut[ut["cell"] == c]
        r, t = members["region"].iloc[0], members["authority_type"].iloc[0]
        to = ut[(ut["region"] == r) & (ut["authority_type"] == t) & (ut["cell"] == ut["unit"])]
        share = to.set_index("cell")[measure] / to[measure].sum()
        targets = targets.add(share * targets[c], fill_value=0).drop(c)
        detail[c] = {"units": members["unit"].tolist(), measure: float(members[measure].sum()),
                     "shared_over": to["unit"].tolist()}  # fmt: skip
    return targets, detail


def rake_to_qpses(sector, adverts: pd.DataFrame, n_tasks, release: str) -> WeightOutput:
    cfg = sector.config["weighting"]
    measure = cfg["target_measure"]  # "fte": salaries are full-time, as for NHS and Education
    q = load(sector.reference(cfg["reference_file"], release))
    per, ut = assign(adverts["org_unit"], q)
    # England only, like NHS and Education. Welsh adverts (about 300, mostly senior posts on
    # national boards) stay in the data, unweighted.
    ut = ut[ut["region"].isin(cfg["regions"])].reset_index(drop=True)
    q = q[q["region"].isin(cfg["regions"])]
    per["qpses_unit"] = per["qpses_unit"].where(per["qpses_unit"].isin(ut["unit"]))

    sample = sector.min_tasks_sample(adverts.assign(unit=per["qpses_unit"].to_numpy()), n_tasks)
    has_unit = sample["unit"].notna()
    ut["n_adverts"] = ut["unit"].map(sample.loc[has_unit, "unit"].value_counts()).fillna(0)
    ut["cell"] = strata(ut, ut.set_index("unit")["n_adverts"], cfg["min_unit_adverts"])
    targets, shared = share_out(ut, measure)
    cells_with_donors = set(ut.loc[ut["n_adverts"] > 0, "cell"])
    lost = targets[~targets.index.isin(cells_with_donors)]
    targets = targets[targets.index.isin(cells_with_donors)]

    donors = sample[has_unit].assign(cell=lambda d: d["unit"].map(ut.set_index("unit")["cell"]))
    res = rake.rake(
        donors[["vacancy_id", "cell"]].reset_index(drop=True),
        {"cell": targets.to_dict()},
        target_fte=float(targets.sum()),
        max_iter=cfg["ipf_max_iter"],
        tol=cfg["ipf_tol"],
    )
    w = res.weights.set_index("vacancy_id")["weight"]
    own = ut[ut["cell"] == ut["unit"]]
    coverage = {
        "target_measure": measure,
        "qpses_total_headcount": float(q["headcount"].sum()),
        "qpses_total_fte": float(q["fte"].sum()),
        "weighted_total": float(w.sum()),
        "share_of_total_weighted": float(w.sum() / q[measure].sum()),
        "not_represented": {str(k): float(v) for k, v in lost.items()},
        "units_in_qpses": len(ut),
        "units_with_adverts": int((ut["n_adverts"] > 0).sum()),
        "units_own_cell": len(own),
        "shared_out": shared,
        f"{measure}_in_own_cells": float(own[measure].sum()),
        "pooled_cells": {
            c: {"units": int(len(g)), "with_adverts": int((g["n_adverts"] > 0).sum()),
                "adverts": int(g["n_adverts"].sum()), measure: float(g[measure].sum())}
            for c, g in ut[ut["cell"].str.endswith("(pooled)")].groupby("cell")
        },
        "eligible_adverts": len(sample),
        "eligible_adverts_weighted": int(w.notna().sum()),
        "unmatched_councils_top": {
            str(k): int(v)
            for k, v in sample.loc[~has_unit, "org_unit"].value_counts().head(30).items()
        },
    }  # fmt: skip
    weights = adverts[["vacancy_id"]].assign(
        weight=adverts["vacancy_id"].map(w),
        weight_spec="rake_council",
        is_primary=True,
        target_source=cfg["target_source"],
        target_date=cfg["target_date"],
    )
    return WeightOutput(
        weights=weights,
        targets=res.targets.assign(weight_spec="rake_council"),
        diagnostics={
            "rake_council": res.diagnostics,
            "coverage": coverage,
            "units": ut.to_dict("records"),
        },
        inputs=[sector.reference(cfg["reference_file"], release)],
    )
