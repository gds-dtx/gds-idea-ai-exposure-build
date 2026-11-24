import json

import pandas as pd

from aie.sectors.civil_service import mappings, professions


def test_cleaning_tables_add_the_scored_era_bodies():
    extra = set(mappings.CLEANING_DEPARTMENTS) - set(mappings.DEPARTMENTS)
    assert extra == {
        "institute-for-apprenticeships-and-technical-education",
        "national-infrastructure-commission",
    }


def _record(vid, args):
    call = {"function": {"arguments": json.dumps(args), "name": "ProfessionClassification"}}
    out = {"choices": [{"message": {"tool_calls": [call]}}]}
    return json.dumps({"recordId": vid, "modelOutput": out})


def test_profession_labels_parse_and_apply():
    lines = [
        _record("1", {"profession": "Finance", "reasoning": "budgets"}),
        _record("2", {"profession": "Astronaut", "reasoning": "not a category"}),
        json.dumps({"recordId": "3", "modelOutput": None}),
    ]
    labels = professions.parse_batch_output(lines)
    assert labels["llm_profession"].notna().tolist() == [True, False, False]
    grid = pd.DataFrame(
        {"vacancy_id": ["1", "2", "3", "4"], "cs_profession": ["Other"] * 3 + ["Policy"]}
    )
    out = professions.apply_labels(grid, labels)
    assert out["cs_profession"].astype(str).tolist() == ["Finance", "Other", "Other", "Policy"]
    assert out["profession_reassignment_method"].tolist() == [
        "llm",
        "original",
        "original",
        "original",
    ]
