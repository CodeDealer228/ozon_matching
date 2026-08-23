"""Generate a checkpointed A2 pilot with Alibaba Singapore qwen3.8-27b.

Secrets are read only from ``DASHSCOPE_API_KEY``.  The script never prints or
persists the key.  Each human-train source is attempted once; transport errors
may retry, but a semantically rejected output is never regenerated.
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
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3.8-27b"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
EXPECTED_SOURCE_SHA256 = "2dee9a0b5b2ef54c4210b07b114346057bddc45effe728e2e43056f42505dd28"
EXPECTED_FEW_SHOT_SHA256 = "2aa9c1734fac177782c897a76780bd60b26d01e952b43c0a5a683af5257f4f2a"
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"

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


def build_messages(pack: dict[str, object], row: object) -> list[dict[str, str]]:
    system = (
        "Ты делаешь A2 hardening для product matching. Есть два разных товара A и B из "
        "проверенной human-negative пары. Перепиши только название B так, чтобы стиль и порядок "
        "слов стали ближе к A, но identity B не изменился. Attributes B неизменяемы. Сохрани все "
        "бренды, модели, артикулы, числа, размеры, цвета, количество, совместимость и единицы B. "
        "Не переноси ни одного факта из A, которого нет в B или attributes B. Не удаляй факты из "
        "исходного названия B. Ответь ровно JSON-объектом {\"rewritten_name2\": \"...\"}."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
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
    messages: list[dict[str, str]],
    seed: int,
    timeout: int,
    max_retries: int,
) -> tuple[str, dict[str, object]]:
    body = json.dumps(
        {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": 0.25,
            "top_p": 0.8,
            "max_tokens": 180,
            "seed": seed,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    for attempt in range(max_retries + 1):
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
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
            return str(content), usage if isinstance(usage, dict) else {}
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"transport_error: {type(exc).__name__}") from exc
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("unreachable")


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
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_pilot",
    )
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--timeout", type=int, default=120)
    # One transport retry is enough for the pilot.  A larger default can stall
    # the sequential run for many minutes on a single unhealthy connection;
    # the failed source is recorded explicitly and can be audited later.
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
    completed = load_completed(raw_path)
    helpers = load_helpers()
    planned = [row for row in pilot.itertuples(index=False) if row.source_pair_sha256 not in completed]
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "model": MODEL_ID,
            "endpoint_host": parsed_endpoint.hostname,
            "planned_rows": len(pilot),
            "already_completed": len(completed),
            "remaining": len(planned),
            "counts_by_category": {
                str(key): int(value) for key, value in pilot["category"].value_counts().sort_index().items()
            },
        }, ensure_ascii=False, indent=2))
        return

    started = time.time()
    for position, row in enumerate(planned, 1):
        seed = args.seed + int(hashlib.sha256(row.source_pair_sha256.encode()).hexdigest()[:8], 16)
        record: dict[str, object] = {
            "source_pair_sha256": row.source_pair_sha256,
            "category": row.category,
            "id1": int(row.id1),
            "id2": int(row.id2),
            "model": MODEL_ID,
            "seed": seed,
            "attempted_at_unix": time.time(),
        }
        try:
            content, usage = request_completion(
                endpoint,
                str(api_key),
                build_messages(pack, row),
                seed,
                args.timeout,
                args.max_retries,
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
                "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })
        except Exception as exc:
            record.update({
                "request_status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            })
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{position}/{len(planned)}] {row.category}: {record['request_status']}", flush=True)

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
    manifest = {
        "status": "complete",
        "model": MODEL_ID,
        "endpoint_host": parsed_endpoint.hostname,
        "api_key_persisted": False,
        "source_pool_sha256": file_sha256(args.source_pool),
        "few_shot_sha256": file_sha256(args.few_shot),
        "canonical_split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "generation_seed": args.seed,
        "decoding": {"temperature": 0.25, "top_p": 0.8, "max_tokens": 180},
        "planned_rows": len(pilot),
        "completed_records": len(records),
        "accepted": sum(bool(value.get("accepted_by_deterministic_validator")) for value in records),
        "request_errors": sum(value.get("request_status") == "error" for value in records),
        "reject_reasons": dict(rejects),
        "raw_responses_sha256": file_sha256(raw_path),
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
