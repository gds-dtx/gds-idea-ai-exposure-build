"""Resumable Bedrock batch runs (via llmbo), for scoring a NEW release (v2 onwards).

All batch I/O goes under the batch root (settings.batch_root()), never into the project folder:

    <AIE_BATCH_ROOT>/<sector>/<release>-<stage>/
        input/<job>.jsonl   output/<job-uid>/...   manifest.json

so scratch can be expired separately. Collected, parsed results
are written to the scored layer (build/sector=<s>/release=<r>/scored/), which is what is kept.

Team rules built in: a cost estimate must be confirmed before submit, jobs under 100 records
are refused (Bedrock batch minimum), job names are unique, failed records can be resubmitted
with `retry`, and every command is safe to re-run after a credential refresh.

Ported from the NHS/local-govt/police run_full.py runners (identical design).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from .. import settings, storage
from ..sectors import Sector
from .labelling import TaskLabellingOutput, label_tasks
from .prompts import AutomationScoredTaskOutput, task_extract_and_score_ai_automation

# The Bedrock batch service role, from the environment (not kept in the code).
ROLE_ARN = os.environ.get("AIE_BATCH_ROLE_ARN", "")
CHUNK = 50_000
MIN_RECORDS = 100
COLLECTABLE = {"Completed", "PartiallyCompleted"}
# gpt-oss-120b batch pricing, $/token (50% of on-demand). Rough chars-per-token for estimates.
PRICE_IN, PRICE_OUT = 0.15e-6 * 0.5, 0.60e-6 * 0.5
CHARS_PER_TOKEN = 4
EST_OUT_TOKENS = {"extract": 1_500, "label": 600}


@dataclass
class Run:
    sector: Sector
    release: str
    stage: str  # extract | label

    @property
    def prefix_uri(self) -> str:
        return settings.batch_dir(self.sector.slug, f"{self.release}-{self.stage}")

    @property
    def manifest_uri(self) -> str:
        return storage.join(self.prefix_uri, "manifest.json")

    def manifest(self) -> dict:
        if storage.exists(self.manifest_uri):
            return storage.read_json(self.manifest_uri)
        return {"chunks": []}

    def save(self, m: dict) -> None:
        storage.write_json(m, self.manifest_uri)


# --- inputs ---------------------------------------------------------------------------------


def _pool(run: Run) -> pd.DataFrame:
    clean = settings.build_dir(run.sector.slug, "clean", run.release)
    adverts = storage.read_parquet(
        storage.join(clean, "adverts.parquet"), columns=["vacancy_id", "job_title"]
    )
    if run.stage == "extract":
        text = storage.read_parquet(storage.join(clean, "advert_text.parquet"))
        return adverts.merge(text, on="vacancy_id").dropna(subset=["advert_text"])
    tasks = storage.read_parquet(
        storage.join(settings.build_dir(run.sector.slug, "scored", run.release), "tasks.parquet"),
        columns=["vacancy_id", "task_index", "task_text"],
    )
    grouped = (
        tasks.sort_values(["vacancy_id", "task_index"])
        .groupby("vacancy_id")["task_text"]
        .apply(list)
        .rename("tasks")
        .reset_index()
    )
    return adverts.merge(grouped, on="vacancy_id")


def _inputs(run: Run, chunk: pd.DataFrame) -> dict:
    from llmbo import ModelInput

    llm = run.sector.config["llm"]
    if run.stage == "extract":
        return {
            r.vacancy_id: ModelInput(
                system=llm["extract_system_prompt"],
                messages=[
                    {
                        "role": "user",
                        "content": task_extract_and_score_ai_automation(
                            f"Job title: {r.job_title}\n\n{r.advert_text}"
                        ),
                    }
                ],
                max_tokens=16384,
            )
            for r in chunk.itertuples()
        }
    return {
        r.vacancy_id: ModelInput(
            system=llm["label_system_prompt"],
            messages=[{"role": "user", "content": label_tasks(r.job_title, r.tasks)}],
            max_tokens=4096,
        )
        for r in chunk.itertuples()
    }


def estimate_cost(run: Run, pool: pd.DataFrame) -> dict:
    col = "advert_text" if run.stage == "extract" else "tasks"
    chars = pool[col].astype(str).str.len().sum() + 6_000 * len(pool)  # prompt template
    tin = chars / CHARS_PER_TOKEN
    tout = EST_OUT_TOKENS[run.stage] * len(pool)
    return {
        "records": len(pool),
        "input_tokens_est": int(tin),
        "output_tokens_est": int(tout),
        "usd_est": round(tin * PRICE_IN + tout * PRICE_OUT, 2),
    }


# --- submit / status / collect --------------------------------------------------------------


def _inferer(run: Run):
    from llmbo import StructuredBatchInferer
    from llmbo.batch_inferer import VALID_FINISHED_STATUSES  # noqa: F401

    bucket, _, prefix = run.prefix_uri.removeprefix("s3://").partition("/")

    class PrefixedInferer(StructuredBatchInferer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.bucket_uri = f"s3://{bucket}/{prefix}"

        def push_requests_to_s3(self):
            self._write_requests_locally()
            return self.session.client("s3").upload_file(
                Filename=self._local_path(self.file_name),
                Bucket=bucket,
                Key=f"{prefix}/input/{self.file_name}",
                ExtraArgs={"ContentType": "application/json"},
            )

    return PrefixedInferer, bucket


def submit(run: Run, confirm_usd: float | None, only_ids: set[str] | None = None) -> None:
    pool = _pool(run)
    if only_ids is not None:
        pool = pool[pool["vacancy_id"].isin(only_ids)]
    m = run.manifest()
    done = {v for ch in m["chunks"] for v in ch.get("vacancy_ids", [])}
    pool = pool[~pool["vacancy_id"].isin(done)].reset_index(drop=True)
    est = estimate_cost(run, pool)
    print(f"{run.sector.slug} {run.release} {run.stage}: {est}")
    if len(pool) < MIN_RECORDS:
        raise SystemExit(f"only {len(pool)} records: Bedrock batch needs at least {MIN_RECORDS}")
    if confirm_usd is None or confirm_usd < est["usd_est"]:
        raise SystemExit(
            f"estimated ${est['usd_est']:,}: re-run with --confirm-usd {est['usd_est']} "
            "(or higher) once the spend is agreed"
        )
    Inferer, bucket = _inferer(run)
    model_id = run.sector.config["scoring"]["bedrock_model_id"]
    output_model = AutomationScoredTaskOutput if run.stage == "extract" else TaskLabellingOutput
    work = Path(tempfile.mkdtemp(prefix="aie-batch-"))
    n0 = len(m["chunks"])
    for i in range(0, len(pool), CHUNK):
        chunk = pool.iloc[i : i + CHUNK]
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        job_name = f"aie-{run.sector.slug}-{run.release}-{run.stage}-{n0 + i // CHUNK:03d}-{ts}"[
            :63
        ]
        sbi = Inferer(
            output_model=output_model,
            model_name=model_id,
            bucket_name=bucket,
            region=settings.REGION,
            job_name=job_name,
            role_arn=_role_arn(),
            output_dir=str(work),
        )
        sbi.prepare_requests(_inputs(run, chunk))
        sbi.push_requests_to_s3()
        sbi.create()
        m["chunks"].append(
            {
                "job_name": job_name,
                "job_arn": sbi.job_arn,
                "n": len(chunk),
                "vacancy_ids": chunk["vacancy_id"].tolist(),
                "submitted_at": ts,
                "collected": False,
            }
        )
        run.save(m)  # after every chunk, so an expired session loses nothing
        print(f"submitted {job_name}: {len(chunk):,} records")


def status(run: Run) -> dict:
    import boto3

    br = boto3.Session(region_name=settings.REGION).client("bedrock")
    c = Counter()
    for ch in run.manifest()["chunks"]:
        c[
            "collected"
            if ch.get("collected")
            else br.get_model_invocation_job(jobIdentifier=ch["job_arn"])["status"]
        ] += 1
    print(f"{run.sector.slug} {run.release} {run.stage}: {dict(c)}")
    return dict(c)


def _payload(mo: dict) -> tuple[dict | list | None, str | None]:
    """Tool-call payload from a gpt-oss or Anthropic record, or (None, failure status)."""
    if "choices" in mo:
        ch = mo["choices"][0]
        tcs = ch["message"].get("tool_calls")
        if not tcs:
            return None, "no_tool_call"
        try:
            return json.loads(tcs[0]["function"]["arguments"]), None
        except json.JSONDecodeError:
            return None, "truncated" if ch.get("finish_reason") == "length" else "invalid_json"
    blocks = [b for b in mo.get("content", []) if b.get("type") == "tool_use"]
    return (blocks[0]["input"], None) if blocks else (None, "no_tool_call")


def parse(path: Path, stage: str) -> tuple[list[dict], list[dict]]:
    """(rows, statuses) from one downloaded output file. Status per record: ok, empty,
    no_tool_call, truncated, invalid_json, validation_error."""
    rows, statuses = [], []
    key, model = (
        ("tasks", AutomationScoredTaskOutput)
        if stage == "extract"
        else ("labels", TaskLabellingOutput)
    )
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            vid = str(r["recordId"])
            payload, fail = _payload(r["modelOutput"])
            if fail:
                statuses.append({"vacancy_id": vid, "status": fail, "n": 0})
                continue
            if isinstance(payload, list):
                payload = {key: payload}
            elif isinstance(payload, dict) and key not in payload and key.title() in payload:
                payload = {key: payload[key.title()]}
            try:
                parsed = model(**payload)
            except (ValidationError, TypeError):
                statuses.append({"vacancy_id": vid, "status": "validation_error", "n": 0})
                continue
            items = getattr(parsed, key)
            statuses.append(
                {"vacancy_id": vid, "status": "ok" if items else "empty", "n": len(items)}
            )
            if stage == "extract":
                rows += [
                    {"vacancy_id": vid, "task_index": i, "task_text": t.task,
                     "reasoning": t.reasoning, "exposure_score": t.automation_potential}
                    for i, t in enumerate(items, 1)
                ]  # fmt: skip
            else:
                rows += [
                    {"vacancy_id": vid, "task_index": x.task_number, "task_label": x.label,
                     "label_reasoning": x.reasoning}
                    for x in items
                ]  # fmt: skip
    return rows, statuses


def collect(run: Run) -> None:
    import boto3

    sess = boto3.Session(region_name=settings.REGION)
    br, s3 = sess.client("bedrock"), sess.client("s3")
    bucket, _, prefix = run.prefix_uri.removeprefix("s3://").partition("/")
    m = run.manifest()
    work = Path(tempfile.mkdtemp(prefix="aie-collect-"))
    for ch in m["chunks"]:
        if ch.get("collected"):
            continue
        st = br.get_model_invocation_job(jobIdentifier=ch["job_arn"])["status"]
        if st not in COLLECTABLE:
            print(f"{ch['job_name']}: {st}")
            continue
        uid = ch["job_arn"].split("/")[-1]
        local = work / f"{ch['job_name']}.out"
        s3.download_file(bucket, f"{prefix}/output/{uid}/{ch['job_name']}.jsonl.out", str(local))
        rows, statuses = parse(local, run.stage)
        storage.write_parquet(
            pd.DataFrame(rows), storage.join(run.prefix_uri, "parsed", f"{ch['job_name']}.parquet")
        )
        storage.write_parquet(
            pd.DataFrame(statuses),
            storage.join(run.prefix_uri, "parsed", f"{ch['job_name']}_status.parquet"),
        )
        ch["collected"] = True
        ch["status_counts"] = dict(Counter(s["status"] for s in statuses))
        run.save(m)
        print(f"{ch['job_name']}: {st} -> {len(rows):,} rows {ch['status_counts']}")
    if m["chunks"] and all(ch.get("collected") for ch in m["chunks"]):
        _finalise(run, m)


def failed_ids(run: Run) -> set[str]:
    out = set()
    for ch in run.manifest()["chunks"]:
        uri = storage.join(run.prefix_uri, "parsed", f"{ch['job_name']}_status.parquet")
        if storage.exists(uri):
            s = storage.read_parquet(uri)
            out |= set(s.loc[~s["status"].isin(["ok", "empty"]), "vacancy_id"])
    return out


def _finalise(run: Run, m: dict) -> None:
    """Combine parsed chunks into the scored layer (contract columns)."""
    from .. import contract

    parts = [
        storage.read_parquet(storage.join(run.prefix_uri, "parsed", f"{ch['job_name']}.parquet"))
        for ch in m["chunks"]
    ]
    rows = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["vacancy_id", "task_index"], keep="last"
    )
    folder = settings.build_dir(run.sector.slug, "scored", run.release)
    model = run.sector.primary_score_model
    common = dict(
        sector=run.sector.slug, layer="scored", release=run.release, inputs=[run.prefix_uri]
    )
    if run.stage == "extract":
        tasks = rows.assign(
            score_model=model, task_label=pd.NA, scoring_run_id=f"{run.release}-extract"
        )
        storage.write_table(
            contract.conform(tasks.drop(columns="reasoning"), "task_scores"),
            folder,
            "tasks.parquet",
            **common,
        )
        ms = tasks[["vacancy_id", "task_index", "exposure_score"]].assign(model=model)
        storage.write_table(
            contract.conform(ms, "model_scores"), folder, "model_scores.parquet", **common
        )
        storage.write_table(
            rows[["vacancy_id", "task_index", "reasoning"]],
            folder,
            "task_reasoning.parquet",
            **common,
        )
    else:
        uri = storage.join(folder, "tasks.parquet")
        tasks = storage.read_parquet(uri).drop(columns="task_label")
        tasks = tasks.merge(
            rows[["vacancy_id", "task_index", "task_label"]],
            on=["vacancy_id", "task_index"],
            how="left",
        )
        storage.write_table(
            contract.conform(tasks, "task_scores"), folder, "tasks.parquet", **common
        )
    print(f"wrote scored layer for {run.sector.slug} {run.release} ({run.stage})")


def _role_arn() -> str:
    if not ROLE_ARN:
        raise SystemExit("set AIE_BATCH_ROLE_ARN to the Bedrock batch service role's ARN")
    return ROLE_ARN


def require_creds() -> None:
    if not (os.getenv("AWS_PROFILE") or os.getenv("AWS_ACCESS_KEY_ID")):
        raise SystemExit("export AWS credentials first")
