"""Generate the 100-row Method C hybrid pilot with checkpointed Alibaba calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import pandas as pd

from c_hybrid_contract import validate_response
from generate_a2_pilot_qwen import (
    DEFAULT_BASE_URL,
    MODEL_ID,
    append_jsonl,
    file_sha256,
    request_completion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "c_hybrid_qwen_pilot_v1"
EXPECTED_PLAN_SHA256 = "d323311541bed79faf01cfebe3622b78ddc2506ee76b11cddbe9bb5896b72aa5"
EXPECTED_REQUEST_PACK_SHA256 = "252f3fb18701b0d2d6af7b29efd1f7e2e6ce5e39ff2c2f915a79e916d866ae1c"
EXPECTED_PROMPT_SHA256 = "7372cb3cf6b09cc41adc670fbda492787cc357cbfdf4f82c3d4d8d6a3c618699"
EXPECTED_ROWS = 100
MAX_COMPLETION_TOKENS = 650


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_response(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("status") not in {"ok", "skip"}:
        raise ValueError("invalid_status_or_schema")
    if value["status"] == "skip":
        if not isinstance(value.get("reason"), str):
            raise ValueError("skip_without_reason")
        return value
    if not isinstance(value.get("synthetic_name"), str):
        raise ValueError("missing_synthetic_name")
    if not isinstance(value.get("attribute_updates"), dict):
        raise ValueError("missing_attribute_updates")
    return value


def compact_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in (
            "category",
            "rule_id",
            "operation",
            "source_name",
            "source_attributes",
            "changed_target_concepts",
            "source_values",
            "replacement_values",
            "context_values",
            "required_output",
        )
    }


def build_messages(
    prompt_pack: dict[str, object],
    payload: dict[str, object],
    explicit_cache: bool,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": str(prompt_pack["system"])}
    ]
    for example in prompt_pack["examples"]:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(example["input"], ensure_ascii=False, sort_keys=True),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(example["output"], ensure_ascii=False, sort_keys=True),
            }
        )
    if explicit_cache:
        stable = str(messages[-1]["content"])
        messages[-1]["content"] = [
            {
                "type": "text",
                "text": stable,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                compact_payload(payload), ensure_ascii=False, sort_keys=True
            ),
        }
    )
    return messages


def terminal_hashes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(value["opportunity_sha256"])
        for value in read_jsonl(path)
        if value.get("opportunity_sha256")
    }


def uncertain_hashes(path: Path, terminal: set[str]) -> set[str]:
    if not path.is_file():
        return set()
    latest: dict[str, str] = {}
    for event in read_jsonl(path):
        latest[str(event["opportunity_sha256"])] = str(event["event"])
    return {
        identity
        for identity, event in latest.items()
        if event == "submitted" and identity not in terminal
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO_ROOT
        / "artifacts/synthetic_v2/c_hybrid_pilot_plan_v1/pilot_plan.parquet",
    )
    parser.add_argument(
        "--request-pack",
        type=Path,
        default=REPO_ROOT
        / "artifacts/synthetic_v2/c_hybrid_pilot_plan_v1/provider_request_pack.jsonl",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=REPO_ROOT
        / "synthetic_research/c_pipeline_variants/c_prompt_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/c_hybrid_pilot_v1",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-explicit-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.workers <= 12:
        raise RuntimeError("--workers must be between 1 and 12")
    for path, expected, label in (
        (args.plan, EXPECTED_PLAN_SHA256, "pilot plan"),
        (args.request_pack, EXPECTED_REQUEST_PACK_SHA256, "provider request pack"),
        (args.prompt_pack, EXPECTED_PROMPT_SHA256, "prompt pack"),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"C {label} SHA-256 mismatch: {actual}")

    plan = pd.read_parquet(args.plan).sort_values("pilot_row_index").reset_index(drop=True)
    payloads = read_jsonl(args.request_pack)
    prompt_pack = json.loads(args.prompt_pack.read_text(encoding="utf-8"))
    if len(plan) != EXPECTED_ROWS or len(payloads) != EXPECTED_ROWS:
        raise RuntimeError("C pilot row count mismatch")
    payload_by_index = {int(value["pilot_row_index"]): value for value in payloads}
    if set(payload_by_index) != set(range(EXPECTED_ROWS)):
        raise RuntimeError("C request pack indices mismatch")
    if any(
        forbidden in args.request_pack.read_text(encoding="utf-8")
        for forbidden in ('"source_id"', '"replacement_donor_id"')
    ):
        raise RuntimeError("C provider request pack contains dataset IDs")
    if prompt_pack.get("version") != "c_hybrid_prompt_v1" or len(
        prompt_pack.get("examples", [])
    ) != 3:
        raise RuntimeError("C prompt pack identity mismatch")

    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme != "https" or not parsed_endpoint.hostname:
        raise RuntimeError("DASHSCOPE_BASE_URL must be valid HTTPS")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not args.dry_run and not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the Alibaba client: pip install -r scripts/requirements-a2.txt"
            ) from exc
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=args.timeout,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_responses.jsonl"
    journal_path = args.output_dir / "request_journal.jsonl"
    config_path = args.output_dir / "generation_config.json"
    explicit_cache = not args.no_explicit_cache
    config = {
        "profile": PROFILE,
        "model": MODEL_ID,
        "endpoint_host": parsed_endpoint.hostname,
        "plan_sha256": file_sha256(args.plan),
        "request_pack_sha256": file_sha256(args.request_pack),
        "prompt_pack_sha256": file_sha256(args.prompt_pack),
        "seed": args.seed,
        "timeout": args.timeout,
        "workers": args.workers,
        "automatic_retries": 0,
        "max_paid_attempts_per_source": 1,
        "enable_thinking": False,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "explicit_cache": explicit_cache,
        "provider_payload_contains_ids": False,
        "provider_payload_contains_validation": False,
    }
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing C output directory has different generation identity")
    if raw_path.is_file() and not config_path.is_file():
        raise RuntimeError("Refusing to mix C output without generation_config.json")

    terminal = terminal_hashes(raw_path)
    uncertain = uncertain_hashes(journal_path, terminal)
    planned = [
        row
        for row in plan.itertuples(index=False)
        if str(row.opportunity_sha256) not in terminal
        and str(row.opportunity_sha256) not in uncertain
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "profile": PROFILE,
                    "model": MODEL_ID,
                    "planned_rows": len(plan),
                    "remaining": len(planned),
                    "uncertain_not_retried": len(uncertain),
                    "workers": args.workers,
                    "automatic_retries": 0,
                    "enable_thinking": False,
                    "explicit_cache": explicit_cache,
                    "provider_payload_contains_ids": False,
                    "provider_payload_contains_validation": False,
                    "counts_by_rule": plan.rule_id.value_counts().sort_index().to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    started = time.time()
    write_lock = Lock()
    print_lock = Lock()

    def process_row(position: int, row: object) -> str:
        identity = str(row.opportunity_sha256)
        payload = payload_by_index[int(row.pilot_row_index)]
        request_seed = args.seed + int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        submitted_at = time.time()
        append_jsonl(
            journal_path,
            {
                "event": "submitted",
                "opportunity_sha256": identity,
                "attempt_number": 1,
                "submitted_at_unix": submitted_at,
                "model": MODEL_ID,
                "enable_thinking": False,
            },
            write_lock,
        )
        record: dict[str, object] = {
            "opportunity_sha256": identity,
            "pilot_row_index": int(row.pilot_row_index),
            "rule_id": str(row.rule_id),
            "category": str(row.category),
            "source_id": int(row.source_id),
            "replacement_donor_id": int(row.replacement_donor_id),
            "model": MODEL_ID,
            "seed": request_seed,
            "attempt_number": 1,
            "attempted_at_unix": submitted_at,
        }
        try:
            content, usage, transport = request_completion(
                client,
                build_messages(prompt_pack, payload, explicit_cache),
                request_seed,
                args.timeout,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            response = parse_response(content)
            accepted, reasons, diagnostics = validate_response(row._asdict(), response)
            record.update(
                {
                    "request_status": "complete",
                    "model_status": response["status"],
                    "skip_reason": response.get("reason"),
                    "accepted_by_hard_validator": accepted,
                    "reject_reasons": reasons,
                    "diagnostics": diagnostics,
                    "usage": usage,
                    "transport": transport,
                    "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "request_status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
        record["finished_at_unix"] = time.time()
        record["elapsed_seconds"] = record["finished_at_unix"] - submitted_at
        append_jsonl(raw_path, record, write_lock)
        append_jsonl(
            journal_path,
            {
                "event": str(record["request_status"]),
                "opportunity_sha256": identity,
                "attempt_number": 1,
                "finished_at_unix": record["finished_at_unix"],
                "provider_request_id": (record.get("transport") or {}).get(
                    "provider_request_id"
                )
                if isinstance(record.get("transport"), dict)
                else None,
            },
            write_lock,
        )
        with print_lock:
            print(
                f"[{position}/{len(planned)}] {row.rule_id}: {record['request_status']}",
                flush=True,
            )
        return str(record["request_status"])

    indexed = list(enumerate(planned, 1))
    parallel_start = 0
    aborted_after_warmup_error = False
    if indexed and explicit_cache:
        first_status = process_row(*indexed[0])
        parallel_start = 1
        aborted_after_warmup_error = first_status == "error"
    work = [] if aborted_after_warmup_error else indexed[parallel_start:]
    if work:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, *value) for value in work]
            for future in as_completed(futures):
                future.result()

    records = read_jsonl(raw_path)
    terminal_after = {str(value["opportunity_sha256"]) for value in records}
    uncertain_after = uncertain_hashes(journal_path, terminal_after)
    plan_by_identity = {
        str(row.opportunity_sha256): row for row in plan.itertuples(index=False)
    }
    result_rows: list[dict[str, object]] = []
    for record in records:
        row = plan_by_identity[str(record["opportunity_sha256"])]
        diagnostics = record.get("diagnostics") or {}
        result_rows.append(
            {
                "opportunity_sha256": record["opportunity_sha256"],
                "pilot_row_index": record["pilot_row_index"],
                "rule_id": record["rule_id"],
                "category": record["category"],
                "source_id": record["source_id"],
                "replacement_donor_id": record["replacement_donor_id"],
                "source_name": row.source_name,
                "synthetic_name": diagnostics.get("synthetic_name"),
                "synthetic_attributes": diagnostics.get("synthetic_attributes"),
                "sample_weight": float(row.recommended_sample_weight),
                "request_status": record["request_status"],
                "model_status": record.get("model_status"),
                "accepted_by_hard_validator": bool(
                    record.get("accepted_by_hard_validator")
                ),
                "reject_reasons_json": json.dumps(
                    record.get("reject_reasons", []), ensure_ascii=False
                ),
            }
        )
    results_path = args.output_dir / "generation_results.parquet"
    pd.DataFrame(result_rows).sort_values("pilot_row_index").to_parquet(
        results_path, index=False
    )
    usage_int = lambda value: int(value or 0)
    usage = {
        "prompt_tokens": sum(
            usage_int((record.get("usage") or {}).get("prompt_tokens"))
            for record in records
        ),
        "completion_tokens": sum(
            usage_int((record.get("usage") or {}).get("completion_tokens"))
            for record in records
        ),
        "reasoning_tokens": sum(
            usage_int(
                ((record.get("usage") or {}).get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
            )
            for record in records
        ),
        "cached_prompt_tokens": sum(
            usage_int(
                ((record.get("usage") or {}).get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                )
            )
            for record in records
        ),
    }
    request_errors = sum(record.get("request_status") == "error" for record in records)
    rejects = Counter(
        reason
        for record in records
        for reason in record.get("reject_reasons", [])
    )
    status = (
        "uncertain"
        if uncertain_after
        else "complete_with_errors"
        if request_errors
        else "complete"
    )
    manifest = {
        "status": status,
        "profile": PROFILE,
        "model": MODEL_ID,
        "planned_rows": len(plan),
        "terminal_records": len(records),
        "request_errors": request_errors,
        "model_skips": sum(record.get("model_status") == "skip" for record in records),
        "accepted_rows": sum(
            bool(record.get("accepted_by_hard_validator")) for record in records
        ),
        "reject_reasons": dict(sorted(rejects.items())),
        "uncertain_submissions": len(uncertain_after),
        "aborted_after_warmup_error": aborted_after_warmup_error,
        "usage": usage,
        "automatic_retries": 0,
        "enable_thinking": False,
        "raw_responses_sha256": file_sha256(raw_path),
        "request_journal_sha256": file_sha256(journal_path),
        "generation_results_sha256": file_sha256(results_path),
        "elapsed_seconds_this_run": time.time() - started,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
