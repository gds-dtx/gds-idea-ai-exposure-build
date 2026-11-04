"""aie: one command line for every sector.

  aie build <sector|all> <stage> --release v1        stage: raw reference clean score weight
  aie publish <sector|all> --release v1              build layers -> curated/ (+ metrics)
  aie metrics <sector> --release v1 --methodology m1  recompute a past methodology's metrics
                                                     (headline totals; --out FILE saves them)
  aie release <sector> v1 --status published         sign a release off in the catalog
  aie catalog                                        what is built and published
  aie cut --spec cut.json [--out file.csv] [--save name]
                                                     a table from a JSON CutSpec (aie.cuts)
  aie cut --options                                  what a cut can choose (JSON)
  aie llm <sector> <extract|label> <submit|status|collect|retry> --release v2 [--confirm-usd N]

AIE_ROOT / AIE_BATCH_ROOT override the S3 roots (e.g. a local folder for testing).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import build, catalog, settings
from .sectors import all_slugs
from .sectors import get as get_sector


def _sectors(arg: str) -> list:
    return [get_sector(s) for s in (all_slugs() if arg == "all" else [arg])]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="aie", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="run one build stage")
    b.add_argument("sector")
    b.add_argument("stage", choices=build.STAGES)
    b.add_argument("--release", default="v1")

    pb = sub.add_parser("publish", help="build layers -> curated tables")
    pb.add_argument("sector")
    pb.add_argument("--release", default="v1")

    mt = sub.add_parser("metrics", help="recompute metrics for a past methodology version")
    mt.add_argument("sector")
    mt.add_argument("--release", default="latest")
    mt.add_argument("--methodology", help="version, e.g. m1 (default: current)")
    mt.add_argument("--model", help="scoring model (default: the sector's primary)")
    mt.add_argument("--out", help="write the metrics rows here (.parquet, local or s3://)")

    r = sub.add_parser("release", help="set a release's status")
    r.add_argument("sector")
    r.add_argument("release")
    r.add_argument("--status", required=True, choices=catalog.STATUSES)

    sub.add_parser("catalog", help="show releases")

    ct = sub.add_parser("cut", help="a table from a JSON CutSpec (what Data Download runs)")
    ct.add_argument("--spec", help="CutSpec as JSON")
    ct.add_argument("--options", action="store_true", help="print what a cut can choose")
    ct.add_argument("--out", help="also write the CSV here")
    ct.add_argument("--save", help="save to cuts/ under this name")

    llm = sub.add_parser("llm", help="score a new release on Bedrock (batch)")
    llm.add_argument("sector")
    llm.add_argument("stage", choices=["extract", "label"])
    llm.add_argument("action", choices=["submit", "status", "collect", "retry"])
    llm.add_argument("--release", required=True)
    llm.add_argument("--confirm-usd", type=float)

    a = p.parse_args(argv)
    print(f"AIE_ROOT={settings.root()}", file=sys.stderr)

    if a.cmd == "build":
        for s in _sectors(a.sector):
            build.run(s, a.stage, a.release)
    elif a.cmd == "publish":
        from .publish import publish

        sectors = _sectors(a.sector)
        catalog.write_sectors_table([get_sector(x) for x in all_slugs()])
        catalog.write_methodology()
        for s in sectors:
            publish(s, a.release)
    elif a.cmd == "metrics":
        from . import metrics

        m = settings.methodology(a.methodology)
        rows, thr = metrics.from_curated(a.sector, a.release, m, a.model)
        print(thr.drop(columns=["methodology_version"]).to_string(index=False))
        for k, v in metrics.totals(rows, a.sector, a.release).items():
            print(f"  {k:<20} {v:>20,.0f}")
        if a.out:
            rows.to_parquet(a.out, index=False)
            print(f"wrote {len(rows):,} rows -> {a.out}")
    elif a.cmd == "release":
        catalog.set_status(a.sector, a.release, a.status)
        print(f"{a.sector} {a.release} -> {a.status}")
    elif a.cmd == "catalog":
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(catalog.releases().to_string(index=False))
    elif a.cmd == "cut":
        _cut(a)
    elif a.cmd == "llm":
        from .llm import batch

        batch.require_creds()
        run = batch.Run(get_sector(a.sector), a.release, a.stage)
        if a.action == "submit":
            batch.submit(run, a.confirm_usd)
        elif a.action == "status":
            batch.status(run)
        elif a.action == "collect":
            batch.collect(run)
        else:
            ids = batch.failed_ids(run)
            print(f"{len(ids):,} failed records to resubmit")
            # Resubmitting needs them out of the "already submitted" set.
            m = run.manifest()
            for ch in m["chunks"]:
                ch["vacancy_ids"] = [v for v in ch.get("vacancy_ids", []) if v not in ids]
            run.save(m)
            batch.submit(run, a.confirm_usd, only_ids=ids)


def _cut(a) -> None:
    import json
    from pathlib import Path

    from .cuts import CutSpec, options, run_cut

    if a.options:
        print(json.dumps(options(), indent=2, default=str))
        return
    if not a.spec:
        raise SystemExit("aie cut: give --spec cut.json or --options")
    res = run_cut(CutSpec.model_validate(json.loads(Path(a.spec).read_text())))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.data.to_string(index=False))
    for w in res.meta["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    if a.out:
        Path(a.out).write_bytes(res.to_csv())
    if a.save:
        print(f"saved {res.save(a.save)}")


if __name__ == "__main__":
    main()
