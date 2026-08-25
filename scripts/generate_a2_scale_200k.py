"""Checkpointed A2 v4 generation for all 20 canonical categories.

The generator rotates through an explicit registry of Alibaba Singapore Qwen
models up to 32B, followed only by explicitly authorized bounded extensions.
It never retries an uncertain request and only changes model after an exact
allocation/free-quota rejection. IDs remain local and are not included in
provider messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
EXPECTED_CATEGORY_COUNT = 20
PROFILE = "a2_v4_all_categories_natural_two_candidate_rotating_models"
PROMPT_VERSION = 2
VALIDATOR_VERSION = 2
MAX_ATTRIBUTES_CHARS = 3000
MAX_COMPLETION_TOKENS = 512


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bounded_seed(*values: object) -> int:
    """Return a deterministic seed accepted by all registry models."""

    return int(stable_hash(*values)[:8], 16) % 2_147_483_647


def normalize(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    return " ".join(re.findall(r"[0-9a-zа-я]+", text))


def word_tokens(value: object) -> list[str]:
    return re.findall(r"[0-9a-zа-я]+", normalize(value))


def numeric_mentions(value: object) -> Counter[str]:
    result: Counter[str] = Counter()
    for token in re.findall(r"(?<![0-9])\d+(?:[.,]\d+)?(?![0-9])", str(value or "")):
        canonical = token.replace(",", ".")
        if "." in canonical:
            canonical = canonical.rstrip("0").rstrip(".")
        result[canonical] += 1
    return result


def identifier_tokens(value: object) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[0-9a-zа-я][0-9a-zа-я._/-]*", str(value or "").lower().replace("ё", "е")):
        compact = re.sub(r"[^0-9a-zа-я]", "", token)
        if len(compact) >= 3 and any(ch.isdigit() for ch in compact) and any(ch.isalpha() for ch in compact):
            result.add(compact)
    return result


def compact_attributes(value: object) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= MAX_ATTRIBUTES_CHARS:
        return text
    return text[:MAX_ATTRIBUTES_CHARS].rsplit(" ", 1)[0] + " …[локально сокращено]"


def build_messages(row: object, explicit_cache: bool) -> list[dict[str, object]]:
    system = (
        "Ты создаёшь A2 hard-negative для product matching. A и B — разные товары из "
        "human-negative пары. Перепиши только название B: сделай его естественным для "
        "маркетплейса и по стилю/порядку слов ближе к A, но сохрани identity B. Не удаляй "
        "факты из исходного B. Марка, модель, артикул, числа, размер, объём, количество, "
        "цвет, комплектация, совместимость и единицы B должны остаться верными. Можно "
        "использовать только факты B из title_b или attributes_b. Нельзя копировать факт "
        "только из A. Не объединяй отдельные ингредиенты, свойства или объекты в новое "
        "составное понятие: например, 'с курицей, с печенью' нельзя превращать в 'с куриной "
        "печенью'. Верни один или два действительно разных естественных варианта. Если "
        "безопасно изменить название нельзя, верни skip. Ответ — один JSON без markdown: "
        "{\"status\":\"ok\",\"candidates\":[{\"name2_prime\":\"...\"},"
        "{\"name2_prime\":\"...\"}]} или {\"status\":\"skip\",\"reason\":\"...\"}."
    )
    example_input = {
        "category": "Канцелярские товары",
        "title_a": "Тетрадь школьная 48 листов в клетку",
        "title_b": "24 листа альбом для рисования формат A4",
        "attributes_b": '{"Формат":"A4","Количество листов":"24"}',
    }
    example_output = {
        "status": "ok",
        "candidates": [
            {"name2_prime": "Альбом для рисования A4, 24 листа"},
            {"name2_prime": "Альбом A4 для рисования, 24 листа"},
        ],
    }
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(example_input, ensure_ascii=False, sort_keys=True)},
        {"role": "assistant", "content": json.dumps(example_output, ensure_ascii=False, sort_keys=True)},
    ]
    if explicit_cache:
        stable = str(messages[-1]["content"])
        messages[-1]["content"] = [
            {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}
        ]
    provider_input = {
        "category": str(row.category),
        "title_a": str(row.name1),
        "title_b": str(row.name2),
        "attributes_b": compact_attributes(row.attributes2),
    }
    messages.append(
        {"role": "user", "content": json.dumps(provider_input, ensure_ascii=False, sort_keys=True)}
    )
    return messages


def parse_response(content: str) -> dict[str, object]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("status") not in {"ok", "skip"}:
        raise ValueError("invalid_status_or_schema")
    if value["status"] == "skip":
        return {"status": "skip", "reason": str(value.get("reason") or "unspecified")[:500]}
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError("expected_one_or_two_candidates")
    parsed: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            title = candidate.strip()
        elif isinstance(candidate, dict) and isinstance(candidate.get("name2_prime"), str):
            title = candidate["name2_prime"].strip()
        else:
            raise ValueError("invalid_candidate_schema")
        if title:
            parsed.append(title)
    if not parsed:
        raise ValueError("no_nonempty_candidates")
    return {"status": "ok", "candidates": parsed}


def _counter_missing(required: Counter[str], actual: Counter[str]) -> list[str]:
    return sorted((required - actual).elements())


def _stem_supported(token: str, evidence: set[str]) -> bool:
    if token in evidence:
        return True
    if not token.isalpha() or len(token) < 5:
        return False
    return any(other.isalpha() and len(other) >= 5 and token[:4] == other[:4] for other in evidence)


def validate_candidate(row: object, candidate: str) -> tuple[bool, list[str], dict[str, object]]:
    reasons: list[str] = []
    original = str(row.name2)
    title_a = str(row.name1)
    attrs_b = str(row.attributes2)
    normalized = normalize(candidate)
    original_normalized = normalize(original)
    if not normalized:
        reasons.append("empty_candidate")
    if normalized == original_normalized:
        reasons.append("trivial_candidate")
    if len(candidate) > 500 or len(candidate) > max(120, int(len(original) * 2.5 + 40)):
        reasons.append("unnatural_length")
    if re.search(r"(?:\bstatus\b|name2_prime|rewritten|```|\{\s*\")", candidate, flags=re.IGNORECASE):
        reasons.append("model_metadata_in_title")

    candidate_numbers = numeric_mentions(candidate)
    original_numbers = numeric_mentions(original)
    attrs_numbers = numeric_mentions(attrs_b)
    a_numbers = numeric_mentions(title_a)
    missing_numbers = _counter_missing(original_numbers, candidate_numbers)
    copied_a_numbers = sorted(set((a_numbers - original_numbers - attrs_numbers).elements()) & set(candidate_numbers))
    if missing_numbers:
        reasons.append("lost_numeric_fact")
    if copied_a_numbers:
        reasons.append("copied_a_only_numeric_fact")

    candidate_ids = identifier_tokens(candidate)
    original_ids = identifier_tokens(original)
    attrs_ids = identifier_tokens(attrs_b)
    a_ids = identifier_tokens(title_a)
    missing_ids = sorted(original_ids - candidate_ids)
    copied_a_ids = sorted((a_ids - original_ids - attrs_ids) & candidate_ids)
    if missing_ids:
        reasons.append("lost_identifier")
    if copied_a_ids:
        reasons.append("copied_a_only_identifier")

    original_words = {token for token in word_tokens(original) if token.isalpha() and len(token) >= 4}
    candidate_words = {token for token in word_tokens(candidate) if token.isalpha() and len(token) >= 4}
    retained = sum(_stem_supported(token, candidate_words) for token in original_words)
    retention = retained / len(original_words) if original_words else 1.0
    if len(original_words) >= 3 and retention < 0.35:
        reasons.append("too_little_b_content_retained")

    # Reject only a meaningful word visibly copied from A when neither B's
    # title nor B's attributes support it.  This catches identity drift such
    # as adding "oversize" from A, without a handcrafted stopword list and
    # without rejecting ordinary inflection changes.
    a_content = {token for token in word_tokens(title_a) if token.isalpha() and len(token) >= 5}
    b_title_content = {token for token in word_tokens(original) if token.isalpha() and len(token) >= 5}
    b_evidence = b_title_content | {
        token for token in word_tokens(attrs_b) if token.isalpha() and len(token) >= 5
    }
    copied_a_only_words = sorted(
        token
        for token in candidate_words
        if len(token) >= 5
        and not _stem_supported(token, b_title_content)
        and _stem_supported(token, a_content)
        and not _stem_supported(token, b_evidence)
    )
    if copied_a_only_words:
        reasons.append("copied_a_only_unsupported_word")

    candidate_tokens = word_tokens(candidate)
    repetition = 0.0
    if candidate_tokens:
        repetition = max(Counter(candidate_tokens).values()) / len(candidate_tokens)
    if len(candidate_tokens) >= 6 and repetition > 0.35:
        reasons.append("excessive_token_repetition")

    diagnostics = {
        "missing_numeric_mentions": missing_numbers,
        "copied_a_only_numeric_mentions": copied_a_numbers,
        "missing_identifiers": missing_ids,
        "copied_a_only_identifiers": copied_a_ids,
        "copied_a_only_unsupported_words": copied_a_only_words,
        "b_content_retention": retention,
        "max_token_repetition": repetition,
        "old_similarity_to_a": SequenceMatcher(None, normalize(title_a), original_normalized).ratio(),
        "new_similarity_to_a": SequenceMatcher(None, normalize(title_a), normalized).ratio(),
        "similarity_to_b": SequenceMatcher(None, original_normalized, normalized).ratio(),
    }
    diagnostics["similarity_delta_to_a"] = (
        diagnostics["new_similarity_to_a"] - diagnostics["old_similarity_to_a"]
    )
    return not reasons, sorted(set(reasons)), diagnostics


def append_jsonl(path: Path, value: dict[str, object], lock: Lock) -> None:
    encoded = json.dumps(value, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def safe_error(exc: Exception) -> dict[str, object]:
    body = getattr(exc, "body", None)
    code = getattr(exc, "code", None)
    if not code and isinstance(body, dict):
        nested = body.get("error")
        code = body.get("code") or (nested.get("code") if isinstance(nested, dict) else None)
    return {
        "type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "code": str(code or ""),
        "message": str(exc)[:1000],
    }


def is_exact_quota_error(error: dict[str, object]) -> bool:
    code = str(error.get("code") or "").casefold()
    message = str(error.get("message") or "").casefold()
    exact_codes = {"allocationquota.freetieronly", "allocationquota.exceeded", "quotaexceeded"}
    return code in exact_codes or "allocationquota.freetieronly" in message


def load_state(raw_path: Path, journal_path: Path) -> tuple[list[dict[str, object]], set[str], set[str]]:
    records = read_jsonl(raw_path)
    completed = {str(record["source_orientation_sha256"]) for record in records}
    last_event: dict[str, str] = {}
    for event in read_jsonl(journal_path):
        last_event[str(event["source_orientation_sha256"])] = str(event["event"])
    uncertain = {
        source
        for source, event in last_event.items()
        if event in {"submitted", "uncertain"} and source not in completed
    }
    return records, completed, uncertain


def accepted_from_record(record: dict[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, candidate in enumerate(record.get("candidate_results") or []):
        if candidate.get("accepted"):
            result.append(
                {
                    "source_orientation_sha256": record["source_orientation_sha256"],
                    "source_pair_sha256": record["source_pair_sha256"],
                    "category": record["category"],
                    "model": record["model"],
                    "candidate_index": index,
                    "name2_prime": candidate["name2_prime"],
                    "diagnostics": json.dumps(candidate.get("diagnostics") or {}, ensure_ascii=False),
                }
            )
    return result


def choose_balanced_categories(
    categories: list[str],
    accepted_counts: Counter[str],
    attempted_counts: Counter[str],
    run_sources: Counter[str],
    target_per_category: int,
    max_new_sources_per_category: int | None,
    limit: int,
) -> list[str]:
    """Choose the least-attempted active categories for the next request window."""

    active = [
        category
        for category in categories
        if accepted_counts[category] < target_per_category
        and (
            max_new_sources_per_category is None
            or run_sources[category] < max_new_sources_per_category
        )
    ]
    return sorted(
        active,
        key=lambda category: (
            attempted_counts[category],
            accepted_counts[category],
            category,
        ),
    )[:limit]


def create_snapshot(
    output_dir: Path,
    pool: pd.DataFrame,
    records: list[dict[str, object]],
    target_per_category: int,
    registry_sha: str,
    raw_path: Path,
    journal_path: Path,
    model_events_path: Path,
    registry_extensions: list[dict[str, object]],
) -> dict[str, object]:
    accepted_rows = [candidate for record in records for candidate in accepted_from_record(record)]
    accepted = pd.DataFrame(accepted_rows)
    if accepted.empty:
        accepted = pd.DataFrame(
            columns=["source_orientation_sha256", "source_pair_sha256", "category", "model", "candidate_index", "name2_prime", "diagnostics"]
        )
    accepted = accepted.sort_values(["category", "source_orientation_sha256", "candidate_index"])
    accepted = accepted.groupby("category", sort=True, group_keys=False).head(target_per_category).reset_index(drop=True)
    local_columns = pool[["source_orientation_sha256", "id1", "id2", "name1", "name2"]]
    selected = accepted.merge(local_columns, on="source_orientation_sha256", how="left", validate="many_to_one")
    selected_path = output_dir / "accepted_a2.parquet"
    selected.to_parquet(selected_path, index=False)

    request_status = Counter(str(record.get("request_status")) for record in records)
    model_counts = Counter(str(record.get("model")) for record in records)
    token_totals: Counter[str] = Counter()
    for record in records:
        usage = record.get("usage") or {}
        token_totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        token_totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        token_totals["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
        token_totals["cached_prompt_tokens"] += int(prompt_details.get("cached_tokens") or 0)
    accepted_counts = {str(k): int(v) for k, v in selected.category.value_counts().sort_index().items()}
    categories = sorted(map(str, pool.category.unique()))
    for category in categories:
        accepted_counts.setdefault(category, 0)
    manifest = {
        "status": "target_complete" if all(accepted_counts[c] >= target_per_category for c in categories) else "in_progress",
        "profile": PROFILE,
        "prompt_version": PROMPT_VERSION,
        "target_per_category": target_per_category,
        "target_total": target_per_category * len(categories),
        "category_count": len(categories),
        "terminal_records": len(records),
        "request_status_counts": dict(sorted(request_status.items())),
        "records_by_model": dict(sorted(model_counts.items())),
        "accepted_total": len(selected),
        "accepted_by_category": accepted_counts,
        "token_usage": dict(token_totals),
        "model_registry_sha256": registry_sha,
        "model_registry_extensions": registry_extensions,
        "source_pool_sha256": file_sha256(output_dir.parent / "source_pool/source_orientations.parquet"),
        "accepted_a2_sha256": file_sha256(selected_path),
        "raw_responses_sha256": file_sha256(raw_path) if raw_path.is_file() else None,
        "request_journal_sha256": file_sha256(journal_path) if journal_path.is_file() else None,
        "model_events_sha256": file_sha256(model_events_path) if model_events_path.is_file() else None,
        "validation_used_as_generation_source": False,
        "ids_sent_to_provider": False,
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    preview_lines = ["# A2 v4 generation preview", "", f"Принято: **{len(selected)}** / {manifest['target_total']}.", ""]
    for category in categories:
        preview_lines.extend([f"## {category}", "", f"Принято: {accepted_counts[category]}", ""])
        sample = selected[selected.category.eq(category)].head(3)
        for row in sample.itertuples(index=False):
            clean_a = str(row.name1).replace("|", "¦")
            clean_b = str(row.name2).replace("|", "¦")
            clean_new = str(row.name2_prime).replace("|", "¦")
            preview_lines.extend([f"- A: {clean_a}", f"  - B: {clean_b}", f"  - B′: {clean_new}"])
        if sample.empty:
            preview_lines.append("- Пока нет принятых примеров.")
        preview_lines.append("")
    (output_dir / "generation_preview.md").write_text("\n".join(preview_lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/a2_scale_200k/source_pool/source_orientations.parquet",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "synthetic_research/a2_scale_200k/model_rotation_v1.json",
    )
    parser.add_argument("--registry-extension", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/a2_scale_200k/generation_v3",
    )
    parser.add_argument("--exclude-output-dir", type=Path, action="append", default=[])
    parser.add_argument("--target-per-category", type=int, default=10_000)
    parser.add_argument("--max-new-sources-per-category", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--no-explicit-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.workers <= 12:
        raise RuntimeError("--workers must be between 1 and 12")
    source_path = args.source_pool.resolve()
    registry_path = args.registry.resolve()
    pool = pd.read_parquet(source_path)
    required = {"source_orientation_sha256", "source_pair_sha256", "category", "label", "id1", "id2", "name1", "name2", "attributes2"}
    if not required.issubset(pool.columns):
        raise RuntimeError(f"Source pool missing columns: {sorted(required - set(pool.columns))}")
    if pool.source_orientation_sha256.duplicated().any() or set(pool.label.unique()) != {0}:
        raise RuntimeError("Invalid source-pool identity or label")
    categories = sorted(map(str, pool.category.unique()))
    if len(categories) != EXPECTED_CATEGORY_COUNT:
        raise RuntimeError(f"Expected 20 categories, found {len(categories)}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    models = [str(value["id"]) for value in sorted(registry["models"], key=lambda value: int(value["rank"]))]
    if not models or any(int(value["parameter_billions"]) > 32 for value in registry["models"]):
        raise RuntimeError("Invalid <=32B model registry")
    registry_sha = file_sha256(registry_path)
    registry_extensions: list[dict[str, object]] = []
    for extension_path_value in args.registry_extension:
        extension_path = extension_path_value.resolve()
        extension = json.loads(extension_path.read_text(encoding="utf-8"))
        extension_models = sorted(
            extension.get("models") or [], key=lambda value: int(value["rank"])
        )
        if not extension_models:
            raise RuntimeError(f"Empty model registry extension: {extension_path}")
        if any(
            not 32 < int(value["parameter_billions"]) <= 35
            for value in extension_models
        ):
            raise RuntimeError(
                "Registry extension may contain only explicitly authorized 33B-35B models"
            )
        extension_ids = [str(value["id"]) for value in extension_models]
        duplicate_ids = set(models) & set(extension_ids)
        if duplicate_ids:
            raise RuntimeError(f"Duplicate models across registries: {sorted(duplicate_ids)}")
        models.extend(extension_ids)
        registry_extensions.append(
            {
                "path_name": extension_path.name,
                "sha256": file_sha256(extension_path),
                "models": extension_ids,
            }
        )
    source_sha = file_sha256(source_path)
    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError("DASHSCOPE_BASE_URL must be HTTPS")

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "profile": PROFILE,
            "source_rows": len(pool),
            "source_pool_sha256": source_sha,
            "categories": categories,
            "target_per_category": args.target_per_category,
            "models": models,
            "workers": args.workers,
            "enable_thinking": False,
            "temperature": 0.2,
            "top_p_set": False,
            "automatic_retries": 0,
            "ids_sent_to_provider": False,
        }, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependencies from scripts/requirements-a2.txt") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_responses.jsonl"
    journal_path = args.output_dir / "request_journal.jsonl"
    model_events_path = args.output_dir / "model_events.jsonl"
    config_path = args.output_dir / "generation_config.json"
    extension_state_path = args.output_dir / "model_registry_extensions.json"
    if extension_state_path.is_file():
        existing_extensions = json.loads(extension_state_path.read_text(encoding="utf-8"))
        if not registry_extensions:
            raise RuntimeError("Existing registry extensions must be supplied again on resume")
        if existing_extensions != registry_extensions:
            raise RuntimeError("Registry extension identity differs from existing output state")
    elif registry_extensions:
        extension_state_path.write_text(
            json.dumps(registry_extensions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    config = {
        "profile": PROFILE,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "source_pool_sha256": source_sha,
        "model_registry_sha256": registry_sha,
        "canonical_split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "target_per_category": args.target_per_category,
        "endpoint_host": parsed.hostname,
        "max_candidates_per_source": 2,
        "enable_thinking": False,
        "temperature": 0.2,
        "top_p_set": False,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "automatic_retries": 0,
        "explicit_cache": not args.no_explicit_cache,
        "excluded_output_identities": [],
    }
    externally_excluded: set[str] = set()
    for exclude_value in args.exclude_output_dir:
        exclude_dir = exclude_value.resolve()
        exclude_raw = exclude_dir / "raw_responses.jsonl"
        exclude_journal = exclude_dir / "request_journal.jsonl"
        for record in read_jsonl(exclude_raw):
            externally_excluded.add(str(record["source_orientation_sha256"]))
        for event in read_jsonl(exclude_journal):
            externally_excluded.add(str(event["source_orientation_sha256"]))
        config["excluded_output_identities"].append({
            "path_name": exclude_dir.name,
            "excluded_sources": len(externally_excluded),
            "raw_responses_sha256": file_sha256(exclude_raw) if exclude_raw.is_file() else None,
            "request_journal_sha256": file_sha256(exclude_journal) if exclude_journal.is_file() else None,
        })
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Generation config differs from existing output identity")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    records, completed, uncertain = load_state(raw_path, journal_path)
    exhausted = {
        str(event["model"])
        for event in read_jsonl(model_events_path)
        if event.get("event") == "quota_exhausted"
    }
    write_lock = Lock()
    accepted_counts: Counter[str] = Counter()
    attempted_counts: Counter[str] = Counter()
    for record in records:
        accepted_counts[str(record["category"])] += len(accepted_from_record(record))
        attempted_counts[str(record["category"])] += 1
    run_sources: Counter[str] = Counter()
    terminal_since_snapshot = 0
    explicit_cache = not args.no_explicit_cache

    ordered_by_category = {
        category: list(pool[pool.category.eq(category)].sort_values("source_orientation_sha256").itertuples(index=False))
        for category in categories
    }
    cursor = {category: 0 for category in categories}

    def next_row(category: str):
        rows = ordered_by_category[category]
        while cursor[category] < len(rows):
            row = rows[cursor[category]]
            cursor[category] += 1
            source = str(row.source_orientation_sha256)
            if source not in completed and source not in uncertain and source not in externally_excluded:
                return row
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=args.timeout)

    def request_one(row: object, model: str) -> dict[str, object]:
        source = str(row.source_orientation_sha256)
        seed = bounded_seed(source, model)
        append_jsonl(journal_path, {
            "event": "submitted", "source_orientation_sha256": source,
            "model": model, "submitted_at_unix": time.time(), "attempt_for_model": 1,
        }, write_lock)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=build_messages(row, explicit_cache and not model.startswith("qwen2")),
                temperature=0.2,
                max_tokens=MAX_COMPLETION_TOKENS,
                seed=seed,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
                timeout=args.timeout,
            )
            content = completion.choices[0].message.content or ""
            usage = completion.usage.model_dump() if completion.usage else {}
            try:
                parsed_response = parse_response(content)
            except Exception as exc:
                return {
                    "kind": "terminal",
                    "record": {
                        "source_orientation_sha256": source,
                        "source_pair_sha256": str(row.source_pair_sha256),
                        "category": str(row.category), "model": model,
                        "request_status": "parse_error", "error": safe_error(exc),
                        "candidate_results": [], "usage": usage,
                        "finish_reason": completion.choices[0].finish_reason,
                        "provider_request_id": getattr(completion, "_request_id", None) or completion.id,
                        "raw_content": content,
                        "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "completed_at_unix": time.time(),
                    },
                }
            candidates = []
            if parsed_response["status"] == "ok":
                seen: set[str] = set()
                for title in parsed_response["candidates"]:
                    key = normalize(title)
                    accepted, reasons, diagnostics = validate_candidate(row, title)
                    if key in seen:
                        accepted = False
                        reasons = sorted(set(reasons + ["duplicate_candidate_in_response"]))
                    seen.add(key)
                    candidates.append({
                        "name2_prime": title, "accepted": accepted,
                        "reject_reasons": reasons, "diagnostics": diagnostics,
                    })
            return {
                "kind": "terminal",
                "record": {
                    "source_orientation_sha256": source,
                    "source_pair_sha256": str(row.source_pair_sha256),
                    "category": str(row.category), "model": model,
                    "request_status": "complete", "provider_status": parsed_response["status"],
                    "skip_reason": parsed_response.get("reason"), "candidate_results": candidates,
                    "usage": usage, "finish_reason": completion.choices[0].finish_reason,
                    "provider_request_id": getattr(completion, "_request_id", None) or completion.id,
                    "raw_content": content, "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "completed_at_unix": time.time(),
                },
            }
        except Exception as exc:
            error = safe_error(exc)
            if is_exact_quota_error(error):
                return {"kind": "quota", "source": source, "model": model, "error": error}
            if error.get("status_code") is None:
                return {"kind": "uncertain", "source": source, "model": model, "error": error}
            return {
                "kind": "terminal",
                "record": {
                    "source_orientation_sha256": source,
                    "source_pair_sha256": str(row.source_pair_sha256),
                    "category": str(row.category), "model": model,
                    "request_status": "provider_error", "error": error,
                    "candidate_results": [], "completed_at_unix": time.time(),
                },
            }

    stop_reason = "no_remaining_sources"
    while True:
        active_categories = choose_balanced_categories(
            categories,
            accepted_counts,
            attempted_counts,
            run_sources,
            args.target_per_category,
            args.max_new_sources_per_category,
            args.workers,
        )
        if not active_categories:
            stop_reason = "target_or_run_cap_reached"
            break
        available_models = [model for model in models if model not in exhausted]
        if not available_models:
            stop_reason = "all_registered_model_quotas_exhausted"
            break
        model = available_models[0]

        batch: list[object] = []
        for category in active_categories:
            row = next_row(category)
            if row is not None:
                batch.append(row)
        if not batch:
            stop_reason = "source_pool_exhausted_before_target"
            break

        quota_hit = False
        with ThreadPoolExecutor(max_workers=min(args.workers, len(batch))) as executor:
            futures = {executor.submit(request_one, row, model): row for row in batch}
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                source = str(row.source_orientation_sha256)
                if result["kind"] == "quota":
                    append_jsonl(journal_path, {
                        "event": "quota_rejected", "source_orientation_sha256": source,
                        "model": model, "at_unix": time.time(), "error": result["error"],
                    }, write_lock)
                    quota_hit = True
                    continue
                if result["kind"] == "uncertain":
                    append_jsonl(journal_path, {
                        "event": "uncertain", "source_orientation_sha256": source,
                        "model": model, "at_unix": time.time(), "error": result["error"],
                    }, write_lock)
                    uncertain.add(source)
                    run_sources[str(row.category)] += 1
                    attempted_counts[str(row.category)] += 1
                    continue
                record = result["record"]
                append_jsonl(raw_path, record, write_lock)
                append_jsonl(journal_path, {
                    "event": "terminal", "source_orientation_sha256": source,
                    "model": model, "at_unix": time.time(),
                }, write_lock)
                records.append(record)
                completed.add(source)
                run_sources[str(row.category)] += 1
                attempted_counts[str(row.category)] += 1
                accepted_counts[str(row.category)] += len(accepted_from_record(record))
                terminal_since_snapshot += 1

        if quota_hit:
            exhausted.add(model)
            append_jsonl(model_events_path, {
                "event": "quota_exhausted", "model": model, "at_unix": time.time(),
                "exact_reason": "AllocationQuota.FreeTierOnly_or_equivalent",
            }, write_lock)
            print(json.dumps({"event": "model_rotated", "exhausted_model": model, "next_model": next((m for m in models if m not in exhausted), None)}, ensure_ascii=False), flush=True)

        if terminal_since_snapshot >= args.snapshot_every:
            manifest = create_snapshot(args.output_dir, pool, records, args.target_per_category, registry_sha, raw_path, journal_path, model_events_path, registry_extensions)
            terminal_since_snapshot = 0
            print(json.dumps({"event": "checkpoint", "terminal_records": len(records), "accepted_total": manifest["accepted_total"], "accepted_by_category": manifest["accepted_by_category"]}, ensure_ascii=False), flush=True)

    manifest = create_snapshot(args.output_dir, pool, records, args.target_per_category, registry_sha, raw_path, journal_path, model_events_path, registry_extensions)
    manifest["stop_reason"] = stop_reason
    manifest["uncertain_sources"] = len(uncertain)
    manifest["exhausted_models"] = [model for model in models if model in exhausted]
    manifest["externally_excluded_sources"] = len(externally_excluded)
    (args.output_dir / "generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
