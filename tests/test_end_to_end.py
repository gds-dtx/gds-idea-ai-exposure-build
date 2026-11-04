"""A tiny synthetic sector through every layer on a local AIE_ROOT, read back with aie.read."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aie import build, catalog, publish, read
from aie.sectors.base import CleanOutput, ScoreOutput, Sector, WeightOutput

CONFIG = """
name = "Toy"
source = "synthetic"
[scoring]
primary_model = "gpt-oss-120b"
[weighting]
primary_spec = "unweighted"
placeholder_reason = "test"
"""


class Toy(Sector):
    slug = "police"  # a registered slug

    def __init__(self, tmp: Path):
        self.package_dir = tmp
        (tmp / "config.toml").write_text(CONFIG)
        super().__init__()

    def clean(self, release):
        n = 60
        ids = [f"pol-{i}" for i in range(n)]
        adverts = pd.DataFrame(
            {
                "vacancy_id": ids,
                "job_title": "Analyst",
                "occupation_group": ["Police staff", "Police officer"] * (n // 2),
                "region": "London",
                "salary_annual_gbp": 30_000.0,
                "headcount": 1,
                "in_scope": [True] * (n - 1) + [False],
                "exclusion_reason": [pd.NA] * (n - 1) + ["not a UK police force"],
            }
        )
        ext = pd.DataFrame({"vacancy_id": ids, "is_flagged": [i % 3 == 0 for i in range(n)]})
        return CleanOutput(adverts, ext)

    def score(self, release):
        rng = np.random.default_rng(1)
        rows = [
            (f"pol-{i}", k, f"task {k}", float(rng.choice(np.arange(11) / 10)))
            for i in range(60)
            for k in range(1, 2 + i % 6)  # 1..6 tasks, so some fall under MIN_TASKS
        ]
        t = pd.DataFrame(rows, columns=["vacancy_id", "task_index", "task_text", "exposure_score"])
        t["score_model"] = "gpt-oss-120b"
        ms = t[["vacancy_id", "task_index", "exposure_score"]].assign(model="gpt-oss-120b")
        return ScoreOutput(t, ms)

    def weight(self, release, adverts, ext, n_tasks) -> WeightOutput:
        return self.unweighted(adverts, n_tasks, "test")


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv("AIE_ROOT", str(tmp_path / "root"))
    read.clear_cache()
    return Toy(tmp_path)


def test_every_layer(toy):
    for stage in ("clean", "score", "weight"):
        build.run(toy, stage, "v1")
    publish.publish(toy, "v1")
    catalog.write_sectors_table([toy])
    catalog.write_methodology()

    rel = catalog.releases()
    assert rel.loc[0, "status"] == "draft" and rel.loc[0, "published_at"]
    assert read.resolve_release("police") == (
        "v1",
        "police: no signed-off release yet, using draft v1",
    )

    dim = read.load("vacancies", ["police"])
    n_eligible = sum(1 for i in range(59) if 1 + i % 6 >= 3)
    assert dim["in_scope"].sum() == n_eligible
    assert set(dim.loc[~dim["in_scope"], "exclusion_reason"]) == {
        "not a UK police force", "fewer than 3 tasks",
    }  # fmt: skip

    assert "is_flagged" in read.sector_columns("police", "v1")
    assert dim["weight"].notna().sum() == n_eligible
    assert read.weight_column("police", "v1") == "weight"
    tasks = read.load("tasks", ["police"])
    assert read.score_column("police", "v1", "gpt-oss-120b") == "exposure_score"
    assert [c for c in tasks.columns if c.startswith("score_") and c != "score_model"] == []
    assert set(read.thresholds("police", "v1")["basis"]) == {"fixed", "sector_quartiles"}

    m = read.load("metrics", ["police"])
    assert set(m["score_model"]) == {"gpt-oss-120b"}
    eligible = m[m["vacancy_id"].isin(dim.loc[dim["in_scope"], "vacancy_id"])]
    assert set(eligible["quadrant"]) <= {
        "Displacement-type", "Augmentation-type", "Concentrated", "Insulated",
    }  # fmt: skip
    assert read.sector_info("police")["primary_weight_spec"] == "unweighted"
    assert read.methodology()["version"] == "m1"

    catalog.set_status("police", "v1", "published")
    assert read.resolve_release("police") == ("v1", None)
