"""Generate a checkpointed A2 pilot with Alibaba Singapore qwen3.8-27b.

Secrets are read only from ``DASHSCOPE_API_KEY``.  The script never prints or
persists the key.  The safe-v2 transport profile disables reasoning, makes at
most one paid request per source, warms an explicit prompt cache once, and then
uses a bounded worker pool.  A durable submission journal prevents an
uncertain timed-out source from being sent again automatically after restart.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3.8-27b"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
EXPECTED_SOURCE_SHA256 = "2dee9a0b5b2ef54c4210b07b114346057bddc45effe728e2e43056f42505dd28"
EXPECTED_FEW_SHOT_SHA256 = "2aa9c1734fac177782c897a76780bd60b26d01e952b43c0a5a683af5257f4f2a"
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
GENERATOR_PROFILE = "a2_qwen_safe_async_v2"

COMMON_SURFACE_WORDS = {
    "товар", "для", "из", "на", "с", "со", "и", "или", "цвет", "размер",
    "женский", "женская", "женские", "мужской", "мужская", "мужские",
    "детский", "детская", "детские", "комплект", "набор", "шт", "штук",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_helpers():
    path = REPO_ROOT / "scripts" / "analyze_synthetic_negative_rules.py"
    spec = importlib.util.spec_from_file_location("synthetic_negative_rules_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokens(value: object, helpers) -> set[str]:
    return set(re.findall(r"[0-9a-zа-я]+", helpers.normalize_text(value)))


def numeric_tokens(value: object, helpers) -> set[str]:
    return {token for token in tokens(value, helpers) if any(char.isdigit() for char in token)}


def identifier_like_tokens(value: object, helpers) -> set[str]:
    result: set[str] = set()
    for token in tokens(value, helpers):
        if len(token) >= 4 and any(char.isdigit() for char in token):
            result.add(token)
    return result


def token_supported_by_item2(token: str, evidence_tokens: set[str]) -> bool:
    if token in evidence_tokens:
        return True
    # Russian adjective/noun inflections often change only the ending:
    # ``красное`` -> ``красного``, ``проба`` -> ``пробы``.  A sufficiently
    # long common stem is evidence, while short codes/brands still require an
    # exact match.
    if not re.fullmatch(r"[а-я]+", token) or len(token) < 5:
        return False
    for evidence in evidence_tokens:
        if not re.fullmatch(r"[а-я]+", evidence) or len(evidence) < 5:
            continue
        common = 0
        for left, right in zip(token, evidence):
            if left != right:
                break
            common += 1
        if common >= 4 and common / min(len(token), len(evidence)) >= 0.70:
            return True
    return False


def build_messages(
    pack: dict[str, object],
    row: object,
    explicit_cache: bool,
) -> list[dict[str, object]]:
    system = (
        "Ты делаешь A2 hardening для product matching. Есть два разных товара A и B из "
        "проверенной human-negative пары. Перепиши только название B так, чтобы стиль и порядок "
        "слов стали ближе к A, но identity B не изменился. Attributes B неизменяемы. Сохрани все "
        "бренды, модели, артикулы, числа, размеры, цвета, количество, совместимость и единицы B. "
        "Не переноси ни одного факта из A, которого нет в B или attributes B. Не удаляй факты из "
        "исходного названия B. Ответь ровно JSON-объектом {\"rewritten_name2\": \"...\"}."
    )
    messages: list[dict[str, object]] = [{"role": "system", "content": system}]
    for example in pack["examples"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Категория: {example['category']}\n"
                    f"Название A: {example['name1']}\n"
                    f"Название B: {example['name2']}"
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"rewritten_name2": example["rewritten_name2"]},
                    ensure_ascii=False,
                ),
            }
        )
    if explicit_cache:
        # Alibaba explicit context cache uses a marker on the final content
        # block of the stable prefix.  The per-source user message stays after
        # this marker and therefore never becomes part of the shared cache.
        stable_content = str(messages[-1]["content"])
        messages[-1]["content"] = [
            {
                "type": "text",
                "text": stable_content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    messages.append(
        {
            "role": "user",
            "content": (
                f"Категория: {row.category}\n"
                f"Название A: {row.name1}\n"
                f"Название B: {row.name2}\n"
                f"Неизменяемые attributes B: {row.attributes2}"
            ),
        }
    )
    return messages


def parse_json_response(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("response_is_not_json")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("rewritten_name2"), str):
        raise ValueError("missing_rewritten_name2")
    return value["rewritten_name2"].strip()


def validate_rewrite(row: object, rewritten: str, helpers) -> tuple[bool, list[str], dict[str, object]]:
    reasons: list[str] = []
    if not rewritten:
        reasons.append("empty_rewrite")
    if len(rewritten) > 500:
        reasons.append("rewrite_too_long")
    original_normalized = helpers.normalize_text(row.name2)
    rewritten_normalized = helpers.normalize_text(rewritten)
    if rewritten_normalized == original_normalized:
        reasons.append("trivial_normalization")

    original_numbers = numeric_tokens(row.name2, helpers)
    rewritten_numbers = numeric_tokens(rewritten, helpers)
    missing_numbers = sorted(original_numbers - rewritten_numbers)
    if missing_numbers:
        reasons.append("lost_numeric_fact")
    copied_numbers = sorted(
        (numeric_tokens(row.name1, helpers) - original_numbers) & rewritten_numbers
    )
    if copied_numbers:
        reasons.append("copied_numeric_fact_from_item1")

    item1_tokens = tokens(row.name1, helpers)
    item2_tokens = tokens(row.name2, helpers)
    attrs2_tokens = tokens(row.attributes2, helpers)
    rewritten_tokens = tokens(rewritten, helpers)
    added_tokens = rewritten_tokens - item2_tokens
    copied_item1_only = sorted(
        token
        for token in added_tokens & (item1_tokens - item2_tokens)
        if not token_supported_by_item2(token, attrs2_tokens)
        and token not in COMMON_SURFACE_WORDS
        and len(token) >= 3
    )
    if copied_item1_only:
        reasons.append("copied_unsupported_fact_from_item1")

    missing_identifiers = sorted(
        identifier_like_tokens(row.name2, helpers) - identifier_like_tokens(rewritten, helpers)
    )
    if missing_identifiers and "lost_numeric_fact" not in reasons:
        reasons.append("lost_identifier")

    old_similarity = helpers.name_similarity(row.name1, row.name2)[0]
    new_similarity = helpers.name_similarity(row.name1, rewritten)[0] if rewritten else 0.0
    if new_similarity <= old_similarity and "trivial_normalization" not in reasons:
        reasons.append("not_harder_by_name_similarity")
    diagnostics = {
        "old_name_similarity": old_similarity,
        "new_name_similarity": new_similarity,
        "name_similarity_delta": new_similarity - old_similarity,
        "missing_numeric_tokens": missing_numbers,
        "copied_numeric_tokens": copied_numbers,
        "copied_item1_only_tokens": copied_item1_only,
        "added_tokens_present_in_item2_attributes": sorted(added_tokens & attrs2_tokens),
    }
    return not reasons, reasons, diagnostics


def request_completion(
    endpoint: str,
    api_key: str,
    messages: list[dict[str, object]],
    seed: int,
    timeout: int,
) -> tuple[str, dict[str, object], dict[str, object]]:
    body = json.dumps(
        {
            "model": MODEL_ID,
            "messages": messages,
            # This is a simple constrained rewrite task.  Reasoning previously
            # consumed 146,786 of 149,442 completion tokens in the 100-row
            # pilot and also amplified the cost of timed-out requests.
            "enable_thinking": False,
            "temperature": 0.25,
            "top_p": 0.8,
            "max_tokens": 180,
            "seed": seed,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            http_status = int(response.status)
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_headers = exc.headers or {}
        provider_request_id = (
            error_headers.get("x-request-id")
            or error_headers.get("x-dashscope-request-id")
            or error_headers.get("request-id")
        )
        detail = f"HTTP {exc.code}"
        if provider_request_id:
            detail += f" request_id={provider_request_id}"
        raise RuntimeError(detail) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"transport_error: {type(exc).__name__}") from exc

    content = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage", {})
    provider_request_id = (
        payload.get("id")
        or response_headers.get("x-request-id")
        or response_headers.get("x-dashscope-request-id")
        or response_headers.get("request-id")
    )
    transport = {
        "provider_request_id": str(provider_request_id) if provider_request_id else None,
        "http_status": http_status,
    }
    return str(content), usage if isinstance(usage, dict) else {}, transport


def append_jsonl(path: Path, record: dict[str, object], lock: Lock) -> None:
    encoded = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def load_uncertain_submissions(path: Path, terminal: set[str]) -> set[str]:
    if not path.is_file():
        return set()
    latest: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            latest[str(event["source_pair_sha256"])] = str(event["event"])
    return {
        source_hash
        for source_hash, event in latest.items()
        if event == "submitted" and source_hash not in terminal
    }


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(str(record["source_pair_sha256"]))
    return completed


def select_pilot(pool: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for category, group in pool.groupby("category", sort=True):
        ordered = group.sort_values("source_pair_sha256").copy()
        rng = random.Random(f"{seed}:{category}")
        indices = list(ordered.index)
        rng.shuffle(indices)
        parts.append(ordered.loc[indices[:per_category]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "source_pools" / "a2_source_pool.parquet",
    )
    parser.add_argument(
        "--few-shot",
        type=Path,
        default=REPO_ROOT / "synthetic_research" / "a2_few_shot_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_pilot_v2",
    )
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-explicit-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 16:
        raise RuntimeError("--workers must be between 1 and 16")

    if file_sha256(args.source_pool) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("A2 source-pool SHA-256 mismatch")
    if file_sha256(args.few_shot) != EXPECTED_FEW_SHOT_SHA256:
        raise RuntimeError("A2 few-shot SHA-256 mismatch")
    pack = json.loads(args.few_shot.read_text(encoding="utf-8"))
    if pack["canonical_split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Canonical split identity mismatch")
    if pack["validation_fingerprint"] != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Validation fingerprint mismatch")

    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme != "https" or not parsed_endpoint.hostname:
        raise RuntimeError("DASHSCOPE_BASE_URL must be a valid HTTPS URL")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not args.dry_run and not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    pool = pd.read_parquet(args.source_pool)
    pilot = select_pilot(pool, args.per_category, args.seed)
    if set(pilot["label"].unique()) != {0}:
        raise RuntimeError("Pilot contains a non-negative human source")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_responses.jsonl"
    journal_path = args.output_dir / "request_journal.jsonl"
    config_path = args.output_dir / "generation_config.json"
    explicit_cache = not args.no_explicit_cache
    config = {
        "generator_profile": GENERATOR_PROFILE,
        "model": MODEL_ID,
        "endpoint_host": parsed_endpoint.hostname,
        "source_pool_sha256": file_sha256(args.source_pool),
        "few_shot_sha256": file_sha256(args.few_shot),
        "canonical_split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "per_category": args.per_category,
        "seed": args.seed,
        "timeout_seconds": args.timeout,
        "workers": args.workers,
        "automatic_retries": 0,
        "enable_thinking": False,
        "explicit_cache": explicit_cache,
        "decoding": {"temperature": 0.25, "top_p": 0.8, "max_tokens": 180},
    }
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError("Generation config differs from the existing output identity")
    elif raw_path.is_file():
        raise RuntimeError("Refusing to mix the safe-v2 profile with a legacy output directory")

    completed = load_completed(raw_path)
    uncertain = load_uncertain_submissions(journal_path, completed)
    helpers = load_helpers()
    planned = [
        row
        for row in pilot.itertuples(index=False)
        if row.source_pair_sha256 not in completed and row.source_pair_sha256 not in uncertain
    ]
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "generator_profile": GENERATOR_PROFILE,
            "model": MODEL_ID,
            "endpoint_host": parsed_endpoint.hostname,
            "planned_rows": len(pilot),
            "already_completed": len(completed),
            "uncertain_not_retried": len(uncertain),
            "remaining": len(planned),
            "workers": args.workers,
            "automatic_retries": 0,
            "enable_thinking": False,
            "explicit_cache": explicit_cache,
            "counts_by_category": {
                str(key): int(value) for key, value in pilot["category"].value_counts().sort_index().items()
            },
        }, ensure_ascii=False, indent=2))
        return

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.time()
    write_lock = Lock()
    print_lock = Lock()

    def process_row(position: int, row: object) -> dict[str, object]:
        seed = args.seed + int(hashlib.sha256(row.source_pair_sha256.encode()).hexdigest()[:8], 16)
        submitted_at = time.time()
        append_jsonl(
            journal_path,
            {
                "event": "submitted",
                "source_pair_sha256": row.source_pair_sha256,
                "attempt_number": 1,
                "submitted_at_unix": submitted_at,
                "model": MODEL_ID,
                "enable_thinking": False,
            },
            write_lock,
        )
        record: dict[str, object] = {
            "source_pair_sha256": row.source_pair_sha256,
            "category": row.category,
            "id1": int(row.id1),
            "id2": int(row.id2),
            "model": MODEL_ID,
            "seed": seed,
            "attempt_number": 1,
            "attempted_at_unix": submitted_at,
            "enable_thinking": False,
            "explicit_cache": explicit_cache,
        }
        try:
            content, usage, transport = request_completion(
                endpoint,
                str(api_key),
                build_messages(pack, row, explicit_cache),
                seed,
                args.timeout,
            )
            rewritten = parse_json_response(content)
            accepted, reasons, diagnostics = validate_rewrite(row, rewritten, helpers)
            record.update({
                "request_status": "complete",
                "rewritten_name2": rewritten,
                "accepted_by_deterministic_validator": accepted,
                "reject_reasons": reasons,
                "diagnostics": diagnostics,
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
        append_jsonl(
            journal_path,
            {
                "event": str(record["request_status"]),
                "source_pair_sha256": row.source_pair_sha256,
                "attempt_number": 1,
                "finished_at_unix": record["finished_at_unix"],
                "provider_request_id": (record.get("transport") or {}).get("provider_request_id")
                if isinstance(record.get("transport"), dict)
                else None,
            },
            write_lock,
        )
        with print_lock:
            print(f"[{position}/{len(planned)}] {row.category}: {record['request_status']}", flush=True)
        return record

    indexed_planned = list(enumerate(planned, 1))
    parallel_start = 0
    if indexed_planned and explicit_cache:
        # Explicit caches are created only after the warm-up response returns.
        # Running this first request alone prevents all workers from racing on
        # the same cold prefix and paying cache-creation cost independently.
        position, row = indexed_planned[0]
        process_row(position, row)
        parallel_start = 1

    parallel_work = indexed_planned[parallel_start:]
    if parallel_work:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, position, row) for position, row in parallel_work]
            for future in as_completed(futures):
                future.result()

    raw_records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pool_by_hash = pool.set_index("source_pair_sha256")
    records: list[dict[str, object]] = []
    for raw_record in raw_records:
        record = dict(raw_record)
        if record.get("request_status") == "complete" and record.get("rewritten_name2"):
            source_hash = str(record["source_pair_sha256"])
            if source_hash not in pool_by_hash.index:
                raise RuntimeError(f"Raw response source is absent from the current pool: {source_hash}")
            source_row = pool_by_hash.loc[source_hash]
            accepted, reasons, diagnostics = validate_rewrite(
                source_row,
                str(record["rewritten_name2"]),
                helpers,
            )
            record["accepted_by_deterministic_validator"] = accepted
            record["reject_reasons"] = reasons
            record["diagnostics"] = diagnostics
        records.append(record)
    results = pd.DataFrame(records)
    results_path = args.output_dir / "a2_pilot_results.parquet"
    results.to_parquet(results_path, index=False)
    rejects = Counter(
        reason
        for record in records
        for reason in record.get("reject_reasons", [])
    )
    journal_events = []
    if journal_path.is_file():
        journal_events = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    submitted_attempts = sum(event.get("event") == "submitted" for event in journal_events)
    terminal_hashes = {str(record["source_pair_sha256"]) for record in records}
    uncertain_after_run = load_uncertain_submissions(journal_path, terminal_hashes)
    if uncertain_after_run:
        raise RuntimeError(f"Run ended with {len(uncertain_after_run)} uncertain paid submissions")
    prompt_tokens = sum(int((record.get("usage") or {}).get("prompt_tokens", 0)) for record in records)
    completion_tokens = sum(int((record.get("usage") or {}).get("completion_tokens", 0)) for record in records)
    reasoning_tokens = sum(
        int(((record.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        for record in records
    )
    cached_tokens = sum(
        int(((record.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        for record in records
    )
    manifest = {
        "status": "complete",
        "generator_profile": GENERATOR_PROFILE,
        "model": MODEL_ID,
        "endpoint_host": parsed_endpoint.hostname,
        "api_key_persisted": False,
        "source_pool_sha256": file_sha256(args.source_pool),
        "few_shot_sha256": file_sha256(args.few_shot),
        "canonical_split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "generation_seed": args.seed,
        "generation_config_sha256": file_sha256(config_path),
        "decoding": {
            "temperature": 0.25,
            "top_p": 0.8,
            "max_tokens": 180,
            "enable_thinking": False,
        },
        "transport": {
            "workers": args.workers,
            "automatic_retries": 0,
            "max_paid_attempts_per_source": 1,
            "explicit_cache": explicit_cache,
            "cache_warmup_requests": 1 if planned and explicit_cache else 0,
        },
        "planned_rows": len(pilot),
        "completed_records": len(records),
        "submitted_attempts": submitted_attempts,
        "uncertain_submissions": 0,
        "accepted": sum(bool(value.get("accepted_by_deterministic_validator")) for value in records),
        "request_errors": sum(value.get("request_status") == "error" for value in records),
        "reject_reasons": dict(rejects),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "visible_completion_tokens": completion_tokens - reasoning_tokens,
            "cached_prompt_tokens": cached_tokens,
        },
        "raw_responses_sha256": file_sha256(raw_path),
        "request_journal_sha256": file_sha256(journal_path) if journal_path.is_file() else None,
        "results_sha256": file_sha256(results_path),
        "elapsed_seconds_this_run": time.time() - started,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
