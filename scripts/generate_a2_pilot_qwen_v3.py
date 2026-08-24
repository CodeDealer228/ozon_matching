"""Run the contract-constrained A2 v3 pilot with safe v2 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import pandas as pd

import analyze_synthetic_negative_rules as rules
from a2_fact_contract import validate_candidate
from generate_a2_pilot_qwen import (
    DEFAULT_BASE_URL,
    MODEL_ID,
    append_jsonl,
    file_sha256,
    load_completed,
    load_uncertain_submissions,
    request_completion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "a2_qwen_contract_v3"
EXPECTED_SOURCE_SHA256 = "084a79f816e2b47ace88b0dea74b9ead69c70c371603837f05cf6a63ceca2770"
EXPECTED_PLAN_SHA256 = "57a4c747c9a122cb1e29279c7c8e4fe5d3c3bc96965cc1890430b5b98483465b"
EXPECTED_FEW_SHOT_SHA256 = "00314f0b40fdc2fbc621ede716cc7e74deed02be6ae957e96e1a68aa597d200c"
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_contract(contract: dict[str, object]) -> dict[str, object]:
    """Provider payload: no IDs, raw attributes or validation content."""
    return {
        "category": contract["category"],
        "a_safe_template": contract["a_safe_template"],
        "name2": contract["name2"],
        "allowed_operations": contract["allowed_operations"],
        "protected_numeric_mentions": contract["protected_numeric_mentions"],
        "protected_identifiers": contract["protected_identifiers"],
    }


def build_messages(
    pack: dict[str, object], contract: dict[str, object], explicit_cache: bool
) -> list[dict[str, object]]:
    system = (
        "Ты выполняешь A2 hardening для product matching. A и B — разные товары из "
        "проверенной human-negative пары. Перепиши только name2, сохраняя identity B. "
        "Используй исключительно ALLOWED_OPERATIONS; placeholders вида <..._A_ONLY> "
        "запрещено копировать или раскрывать. Все protected numbers/identifiers должны "
        "остаться. Не добавляй никаких иных фактов. Если безопасного содержательного "
        "изменения нет, верни status=skip. Для status=ok верни 1-2 разных кандидата. "
        "Каждый кандидат обязан перечислить использованные fact_id и operation_id. "
        "Ответь одним JSON-объектом без markdown. Схема: "
        "{status:'ok',candidates:[{name2_prime:'...',used_fact_ids:[...],"
        "applied_operation_ids:[...]}]} или {status:'skip',reason:'...'}"
    )
    messages: list[dict[str, object]] = [{"role": "system", "content": system}]
    for example in pack["examples"]:
        messages.append({
            "role": "user",
            "content": json.dumps(example["input"], ensure_ascii=False, sort_keys=True),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["output"], ensure_ascii=False, sort_keys=True),
        })
    if explicit_cache:
        stable = str(messages[-1]["content"])
        messages[-1]["content"] = [{
            "type": "text",
            "text": stable,
            "cache_control": {"type": "ephemeral"},
        }]
    messages.append({
        "role": "user",
        "content": json.dumps(compact_contract(contract), ensure_ascii=False, sort_keys=True),
    })
    return messages


def parse_response(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("status") not in {"ok", "skip"}:
        raise ValueError("invalid_status_or_schema")
    if value["status"] == "skip":
        if not isinstance(value.get("reason"), str):
            raise ValueError("skip_without_reason")
        return value
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError("expected_one_or_two_candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name2_prime"), str):
            raise ValueError("invalid_candidate")
        if not isinstance(candidate.get("used_fact_ids"), list):
            raise ValueError("missing_used_fact_ids")
        if not isinstance(candidate.get("applied_operation_ids"), list):
            raise ValueError("missing_applied_operation_ids")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_pilot_plan" / "pilot_source_pool.parquet",
    )
    parser.add_argument(
        "--plan", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_pilot_plan" / "pilot_generation_plan.jsonl",
    )
    parser.add_argument(
        "--few-shot", type=Path,
        default=REPO_ROOT / "synthetic_research" / "a2_few_shot_v3.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_pilot_v3",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-explicit-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise RuntimeError("--workers must be between 1 and 16")
    for path, expected, label in (
        (args.source_pool, EXPECTED_SOURCE_SHA256, "source pool"),
        (args.plan, EXPECTED_PLAN_SHA256, "generation plan"),
        (args.few_shot, EXPECTED_FEW_SHOT_SHA256, "few-shot pack"),
    ):
        if file_sha256(path) != expected:
            raise RuntimeError(f"A2 v3 {label} SHA-256 mismatch")

    pool = pd.read_parquet(args.source_pool)
    plan = read_jsonl(args.plan)
    contracts = {str(value["source_pair_sha256"]): value for value in plan}
    if len(pool) != 20 or len(contracts) != 20 or set(pool["label"].unique()) != {0}:
        raise RuntimeError("Pilot identity mismatch")
    if set(map(str, pool["source_pair_sha256"])) != set(contracts):
        raise RuntimeError("Source pool and contract plan differ")
    if any(value["tier"] != "A" for value in plan):
        raise RuntimeError("Pilot contains non-Tier-A contract")
    pack = json.loads(args.few_shot.read_text(encoding="utf-8"))
    if pack["canonical_split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Few-shot split identity mismatch")
    if pack["validation_fingerprint"] != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Few-shot validation identity mismatch")

    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError("DASHSCOPE_BASE_URL must be valid HTTPS")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not args.dry_run and not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    explicit_cache = not args.no_explicit_cache
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_responses.jsonl"
    journal_path = args.output_dir / "request_journal.jsonl"
    config_path = args.output_dir / "generation_config.json"
    config = {
        "profile": PROFILE,
        "model": MODEL_ID,
        "endpoint_host": parsed.hostname,
        "source_pool_sha256": file_sha256(args.source_pool),
        "plan_sha256": file_sha256(args.plan),
        "few_shot_sha256": file_sha256(args.few_shot),
        "seed": args.seed,
        "timeout": args.timeout,
        "workers": args.workers,
        "automatic_retries": 0,
        "max_paid_attempts_per_source": 1,
        "enable_thinking": False,
        "candidate_count": "1-2",
        "explicit_cache": explicit_cache,
        "provider_payload_fields": [
            "category", "a_safe_template", "name2", "allowed_operations",
            "protected_numeric_mentions", "protected_identifiers",
        ],
        "provider_payload_contains_ids": False,
        "provider_payload_contains_validation": False,
    }
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing output directory has different generation identity")
    if raw_path.is_file() and not config_path.is_file():
        raise RuntimeError("Refusing to mix v3 with legacy output")

    completed = load_completed(raw_path)
    uncertain = load_uncertain_submissions(journal_path, completed)
    planned = [
        row for row in pool.itertuples(index=False)
        if str(row.source_pair_sha256) not in completed and str(row.source_pair_sha256) not in uncertain
    ]
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "profile": PROFILE,
            "model": MODEL_ID,
            "endpoint_host": parsed.hostname,
            "planned_rows": len(pool),
            "remaining": len(planned),
            "uncertain_not_retried": len(uncertain),
            "automatic_retries": 0,
            "enable_thinking": False,
            "workers": args.workers,
            "explicit_cache": explicit_cache,
            "provider_payload_contains_ids": False,
            "provider_payload_contains_validation": False,
            "counts_by_category": {
                str(k): int(v) for k, v in pool["category"].value_counts().sort_index().items()
            },
        }, ensure_ascii=False, indent=2))
        return

    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    write_lock, print_lock = Lock(), Lock()

    def process_row(position: int, row: object) -> None:
        source_hash = str(row.source_pair_sha256)
        contract = contracts[source_hash]
        seed = args.seed + int(hashlib.sha256(source_hash.encode()).hexdigest()[:8], 16)
        submitted_at = time.time()
        append_jsonl(journal_path, {
            "event": "submitted", "source_pair_sha256": source_hash,
            "attempt_number": 1, "submitted_at_unix": submitted_at,
            "model": MODEL_ID, "enable_thinking": False,
        }, write_lock)
        record: dict[str, object] = {
            "source_pair_sha256": source_hash,
            "category": row.category,
            "id1": int(row.id1), "id2": int(row.id2),
            "contract_sha256": contract["contract_sha256"],
            "model": MODEL_ID, "seed": seed, "attempt_number": 1,
            "attempted_at_unix": submitted_at,
        }
        try:
            content, usage, transport = request_completion(
                endpoint, str(api_key), build_messages(pack, contract, explicit_cache),
                seed, args.timeout,
            )
            response = parse_response(content)
            validated: list[dict[str, object]] = []
            if response["status"] == "ok":
                for candidate in response["candidates"]:
                    accepted, reasons, diagnostics = validate_candidate(
                        contract,
                        str(candidate["name2_prime"]),
                        candidate["used_fact_ids"],
                        candidate["applied_operation_ids"],
                    )
                    old_similarity = rules.name_similarity(row.name1, row.name2)[0]
                    new_similarity = rules.name_similarity(row.name1, str(candidate["name2_prime"]))[0]
                    validated.append({
                        **candidate,
                        "accepted_by_deterministic_validator": accepted,
                        "reject_reasons": reasons,
                        "diagnostics": diagnostics,
                        "old_name_similarity": old_similarity,
                        "new_name_similarity": new_similarity,
                        "name_similarity_delta": new_similarity - old_similarity,
                    })
            record.update({
                "request_status": "complete",
                "model_status": response["status"],
                "skip_reason": response.get("reason"),
                "candidates": validated,
                "usage": usage,
                "transport": transport,
                "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })
        except Exception as exc:
            record.update({
                "request_status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            })
        record["finished_at_unix"] = time.time()
        record["elapsed_seconds"] = record["finished_at_unix"] - submitted_at
        append_jsonl(raw_path, record, write_lock)
        append_jsonl(journal_path, {
            "event": str(record["request_status"]),
            "source_pair_sha256": source_hash,
            "attempt_number": 1,
            "finished_at_unix": record["finished_at_unix"],
            "provider_request_id": (record.get("transport") or {}).get("provider_request_id")
            if isinstance(record.get("transport"), dict) else None,
        }, write_lock)
        with print_lock:
            print(f"[{position}/{len(planned)}] {row.category}: {record['request_status']}", flush=True)

    indexed = list(enumerate(planned, 1))
    parallel_start = 0
    if indexed and explicit_cache:
        process_row(*indexed[0])
        parallel_start = 1
    work = indexed[parallel_start:]
    if work:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, *value) for value in work]
            for future in as_completed(futures):
                future.result()

    records = read_jsonl(raw_path)
    terminal = {str(value["source_pair_sha256"]) for value in records}
    uncertain_after = load_uncertain_submissions(journal_path, terminal)
    candidate_rows: list[dict[str, object]] = []
    for record in records:
        for index, candidate in enumerate(record.get("candidates", [])):
            candidate_rows.append({
                "source_pair_sha256": record["source_pair_sha256"],
                "category": record["category"],
                "candidate_index": index,
                "name2_prime": candidate["name2_prime"],
                "accepted_by_deterministic_validator": candidate["accepted_by_deterministic_validator"],
                "reject_reasons_json": json.dumps(candidate["reject_reasons"], ensure_ascii=False),
                "used_fact_ids_json": json.dumps(candidate["used_fact_ids"], ensure_ascii=False),
                "applied_operation_ids_json": json.dumps(candidate["applied_operation_ids"], ensure_ascii=False),
                "old_name_similarity": candidate["old_name_similarity"],
                "new_name_similarity": candidate["new_name_similarity"],
                "name_similarity_delta": candidate["name_similarity_delta"],
            })
    candidates_path = args.output_dir / "candidate_results.parquet"
    pd.DataFrame(candidate_rows).to_parquet(candidates_path, index=False)
    rejects = Counter(
        reason for record in records for candidate in record.get("candidates", [])
        for reason in candidate.get("reject_reasons", [])
    )
    usage = {
        "prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens", 0)) for r in records),
        "completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens", 0)) for r in records),
        "reasoning_tokens": sum(int(((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)) for r in records),
        "cached_prompt_tokens": sum(int(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)) for r in records),
    }
    manifest = {
        "status": "complete" if not uncertain_after else "uncertain",
        "profile": PROFILE, "model": MODEL_ID,
        "source_pool_sha256": file_sha256(args.source_pool),
        "plan_sha256": file_sha256(args.plan),
        "few_shot_sha256": file_sha256(args.few_shot),
        "planned_rows": len(pool), "completed_records": len(records),
        "request_errors": sum(r.get("request_status") == "error" for r in records),
        "model_skips": sum(r.get("model_status") == "skip" for r in records),
        "candidate_rows": len(candidate_rows),
        "accepted_candidates": sum(bool(c.get("accepted_by_deterministic_validator")) for r in records for c in r.get("candidates", [])),
        "reject_reasons": dict(rejects),
        "uncertain_submissions": len(uncertain_after),
        "usage": usage,
        "automatic_retries": 0, "enable_thinking": False,
        "raw_responses_sha256": file_sha256(raw_path),
        "request_journal_sha256": file_sha256(journal_path),
        "candidate_results_sha256": file_sha256(candidates_path),
        "elapsed_seconds_this_run": time.time() - started,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
