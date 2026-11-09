# AI exposure: build

Builds AI exposure data for public-sector workforces (Civil Service, NHS, Education, local
government, police and public corporations), from job adverts to a set of curated tables, and
provides the shared methods and cut engine that analysis and dashboard code use to read them.

The package is `aie`. Other projects install it as a library for the curated loader
(`aie.read`), the catalog, the shared methods and the cut engine (`aie.cuts`).

> **Status: in development.** Methods and releases are not yet signed off.

## The six layers

```
 L0 raw ──► L1 clean ──► L2 scored ──► L3 weighted ──► L4 curated ──► L5 cuts / outputs
 adverts    standard      tasks +       raking to        one schema      self-service
 as         advert        exposure      official         for every       cuts, headline
 received   table         scores (LLM)  workforce stats  sector          tables
```

Each layer reads only the layer before it. Fixes and method tweaks rebuild and republish a
sector's **current** release in place; a new release (`v1`, `v2`, …) is made only when the team
decides to keep the old one alongside (as for Civil Service v1 → v2). Every folder gets a
`_meta.json` recording its inputs, git commit and methodology version.

| Layer | Where | What |
|---|---|---|
| L0 raw | `build/sector=<s>/release=<r>/raw/` | The files this release was built from, exactly as received |
| reference | `build/sector=<s>/release=<r>/reference/` | Official statistics this release uses for raking and pay |
| L1 clean | `build/sector=<s>/release=<r>/clean/` | `adverts.parquet` (contract columns), `adverts_ext.parquet`, `advert_text.parquet`, `funnel.csv` |
| L2 scored | `build/sector=<s>/release=<r>/scored/` | `tasks.parquet`, `model_scores.parquet`, `task_reasoning.parquet` |
| L3 weighted | `build/sector=<s>/release=<r>/weights/` | `vacancy_weights.parquet` (one row per advert per spec), `targets.parquet`, `rake_diagnostics.json` |
| L4 curated | `curated/sector=<s>/release=<r>/` | `vacancies.parquet`, `tasks.parquet`, `metrics.parquet`. **The only thing analysis reads.** See the contract below |
| L5 cuts / outputs | `cuts/<scope>/`, `outputs/<scope>/` | Saved cuts (`aie.cuts`) and analysis tables, each with metadata |
| catalog | `catalog/` | `sectors.csv`, `releases.csv` (draft / published / superseded / legacy), `methodology.toml`, `cuts_manifest.csv` |

All paths are relative to the data root, `AIE_ROOT`.

## The curated contract (`src/aie/contract.py`)

Three files per sector and release, in `curated/sector=<s>/release=<r>/`:

| File | One row per | Columns |
|---|---|---|
| `vacancies.parquet` | advert | **Core** (same in every sector): vacancy_id (`cs-` `nhs-` `edu-` `lg-` `pol-`), job_title, employer, org_unit, occupation_group, grade, region, salary_annual_gbp, salary_source, headcount, n_tasks, in_scope, exclusion_reason, `weight` (primary spec). **Then** `weight_<spec>` for other specs, then the sector's own columns |
| `tasks.parquet` | task | **Core**: vacancy_id, task_index, task_text, task_label, exposure_score (the sector's primary model), score_model. **Then** `score_<model>` for every other model |
| `metrics.parquet` | advert × model (current methodology only) | Decay-weighted mean and variance of exposure, quadrant, and per-FTE valuation columns |

`metrics` is kept separate because it is the only file rebuilt when the methodology changes.
`_meta.json` records which weight spec `weight` is, the raking target of each spec and the
thresholds used.

## Methodology lives in one file

`methodology.toml` holds every methodology setting (task decay, minimum tasks per advert,
quadrant thresholds, valuation rules, the cross-sector model). Curated tables hold the current
version only. A past version is kept as settings in the file's `[history]` section, with a
`[[changelog]]` entry, and can be reproduced exactly because published releases never change:

```bash
uv run aie metrics nhs --release v1 --methodology m1
uv run aie metrics nhs --release v1 --methodology m1 --out m1_nhs.parquet
```

To change the primary: copy the current settings into `[history.<old version>]`, edit them,
bump `version`, add a changelog entry, then run `aie publish all`. Nothing upstream needs
rebuilding.

Sensitivity scenarios are not published here. They are computed on the fly with
`aie.metrics.from_curated(sector, release, settings.with_changes({...}, label))`.

## Sectors

Each sector folder (`src/aie/sectors/<x>/`) has a `config.toml` (its releases, sources and
raking targets) and a `sector.py`. Adding a sector means copying a folder. Notes on the
official statistics used for weighting are in `docs/`:

- Local government: ONS QPSES via the LGA ([docs/qpses.md](docs/qpses.md))
- Police: Home Office police workforce ([docs/police_workforce.md](docs/police_workforce.md))
- Raking design for local government and police ([docs/raking_local_govt_police.md](docs/raking_local_govt_police.md))
- Public corporations: each body's annual report ([docs/public_corporations.md](docs/public_corporations.md))

## Running it

```bash
uv sync
aie build nhs raw --release v1       # source files -> build/sector=nhs/release=v1/raw/
aie build nhs reference --release v1 # official stats -> .../release=v1/reference/
aie build nhs clean --release v1     # L1
aie build nhs score --release v1     # L2
aie build nhs weight --release v1    # L3
aie publish nhs --release v1         # L4 curated: vacancies, tasks, metrics
aie release nhs v1 --status published
aie catalog
```

Set `AIE_ROOT=/some/local/folder` to run against a local folder. The tests do this.

Scoring a new release with an LLM batch job needs a confirmed cost estimate. Jobs under 100
records are refused, and failed records can be resubmitted:

```bash
aie llm nhs extract submit --release v2                    # prints the estimate, refuses
aie llm nhs extract submit --release v2 --confirm-usd 420  # after the spend is agreed
aie llm nhs extract status --release v2
aie llm nhs extract collect --release v2                   # -> build/sector=nhs/release=v2/scored/
aie llm nhs extract retry --release v2 --confirm-usd 5
```

## Reading the curated data

```python
from aie import read
v = read.load("vacancies", ["nhs", "education"], release="v1")   # pin the release
m = read.load("metrics", ["nhs"], release="v1", columns=["vacancy_id", "score_model", "quadrant"])
read.thresholds("nhs", "v1"); read.weight_specs("civil-service", "v1"); read.sector_columns("local-govt", "v1")
```

Cuts (group by, filter, one row per advert) are `aie.cuts`; see [docs/cuts.md](docs/cuts.md).

## Layout

```
methodology.toml          the one place methodology lives
src/aie/
  settings.py storage.py  where things live; S3/local IO and _meta.json
  contract.py             curated tables and columns
  catalog.py              releases, status, cuts manifest
  build.py publish.py     stages and the curated publish
  read.py                 the curated loader
  methods/                decay, quadrants, valuation, rake (one copy each)
  llm/                    shared prompts + resumable LLM batch runs
  sectors/<sector>/       config.toml + sector.py (+ sector helpers)
docs/                     method notes
tests/
```
