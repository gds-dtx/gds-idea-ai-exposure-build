"""Profession correction (release v2, step 1d): replace GRID's 'Other' profession with an LLM's
classification of the advert.

Ported from gds-idea-ai-exposure (src/1d_batch_remap_professions.ipynb and
src/profession_prompts.py at commit c2a10a5). The LLM run (Qwen 3 235B on Bedrock, every
advert whose profession was 'Other', given only its title and description) is not repeated:
its two batch output files are kept in raw/ as PROCESSED inputs and parsed here the way the
original parser did. A record counts only if the model returned a profession from
PROFESSION_CATEGORIES; failed records keep 'Other'.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd

PROFESSION_CATEGORIES: tuple[str, ...] = (
    "Operational Delivery", "Policy", "Digital and Data", "Project Delivery",
    "Finance", "Corporate Finance", "Commercial", "Human Resources",
    "Communications", "Legal", "Security", "Property", "Science and Engineering",
    "Intelligence Analysis", "Counter Fraud", "Statistics", "Economics",
    "Social Research", "Operational Research",
    "Knowledge and Information Management", "Internal Audit", "Planning",
    "Planning Inspectors", "Tax", "Veterinarian",
    "Actuary", "Clinical", "Geography", "Inspector of Education and Training",
    "Occupational Psychology", "Risk Management",
    "Other",  # a valid answer: the advert does not say enough to tell
)  # fmt: skip


def parse_batch_output(lines: Iterable[str]) -> pd.DataFrame:
    """One row per record of a Bedrock batch output (.jsonl.out): vacancy_id, llm_profession
    (None when the call failed or the answer was not a valid category)."""
    rows = []
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        profession = None
        try:
            call = rec["modelOutput"]["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(call["function"]["arguments"])
            if args.get("profession") in PROFESSION_CATEGORIES and isinstance(
                args.get("reasoning"), str
            ):
                profession = args["profession"]
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        rows.append({"vacancy_id": str(rec["recordId"]), "llm_profession": profession})
    return pd.DataFrame(rows, columns=["vacancy_id", "llm_profession"])


def apply_labels(grid: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Replace cs_profession with the LLM's answer where the call succeeded; keep the original
    in cs_profession_original and record profession_reassignment_method."""
    ok = labels[labels["llm_profession"].notna()]
    lookup = dict(zip(ok["vacancy_id"], ok["llm_profession"], strict=True))
    g = grid.copy()
    g["vacancy_id"] = g["vacancy_id"].astype(str)
    g["cs_profession"] = g["cs_profession"].astype(str)
    replace = g["vacancy_id"].isin(lookup)
    g["cs_profession_original"] = g["cs_profession"]
    g.loc[replace, "cs_profession"] = g.loc[replace, "vacancy_id"].map(lookup)
    g["profession_reassignment_method"] = "original"
    g.loc[replace, "profession_reassignment_method"] = "llm"
    g["cs_profession"] = g["cs_profession"].astype("category")
    return g
