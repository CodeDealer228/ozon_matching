"""Generate paired counterfactual examples from human positive product pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from sample_name_attribute_overlap import candidate_intersections, flatten_values


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
TOOL_NAME = "submit_replacement_decision"
ASCII_SIGNAL_RE = re.compile(r"[A-Za-z0-9]")
WHITESPACE_RE = re.compile(r"\s")
OFFER_STRIP_CHARS = "\t\r\n .,;:!?\"'«»"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_sha256(id1: object, id2: object) -> str:
    ordered = sorted((str(id1), str(id2)))
    return hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()


def parse_attributes(value: object) -> object:
    if not isinstance(value, str):
        raise ValueError("attributes_not_string")
    parsed = json.loads(value)
    if not isinstance(parsed, (dict, list)):
        raise ValueError("attributes_not_object_or_list")
    return parsed


def offered_tokens(left: pd.Series, right: pd.Series, max_tokens: int) -> list[str]:
    left_attributes = parse_attributes(left["attributes"])
    right_attributes = parse_attributes(right["attributes"])
    left_text = f"{left['name']} {' '.join(flatten_values(left_attributes))}"
    right_text = f"{right['name']} {' '.join(flatten_values(right_attributes))}"
    intersections = candidate_intersections(left_text, right_text)
    candidates = {
        token.strip(OFFER_STRIP_CHARS)
        for tokens in intersections.values()
        for token in tokens
        if token.strip(OFFER_STRIP_CHARS)
        and ASCII_SIGNAL_RE.search(token.strip(OFFER_STRIP_CHARS))
    }

    def rank(token: str) -> tuple[int, int, int, str]:
        has_letter = bool(re.search(r"[A-Za-z]", token))
        has_digit = bool(re.search(r"\d", token))
        return (int(has_letter and has_digit), int(has_letter), len(token), token)

    return sorted(candidates, key=rank, reverse=True)[:max_tokens]


def decision_tool(allowed_tokens: list[str], max_replacements: int) -> dict[str, object]:
    if not allowed_tokens:
        raise ValueError("allowed_tokens_empty")
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Return safe independent token replacements or skip this product pair. "
                "source_token must be selected from the supplied enum."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["replace", "skip"]},
                    "replacements": {
                        "type": "array",
                        "maxItems": max_replacements,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source_token": {"type": "string", "enum": allowed_tokens},
                                "replacement_token": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 64,
                                },
                            },
                            "required": ["source_token", "replacement_token"],
                        },
                    },
                    "skip_reason": {"type": "string", "maxLength": 200},
                },
                "required": ["status", "replacements", "skip_reason"],
            },
        },
    }


def validate_decision(
    value: object, allowed_tokens: list[str], max_replacements: int
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("decision_not_object")
    if set(value) != {"status", "replacements", "skip_reason"}:
        raise ValueError("decision_fields_mismatch")
    status = value["status"]
    replacements = value["replacements"]
    skip_reason = value["skip_reason"]
    if status not in {"replace", "skip"}:
        raise ValueError("invalid_status")
    if not isinstance(replacements, list) or not isinstance(skip_reason, str):
        raise ValueError("invalid_decision_types")
    if status == "skip":
        if replacements or not skip_reason.strip():
            raise ValueError("invalid_skip")
        return value
    if not 1 <= len(replacements) <= max_replacements:
        raise ValueError("invalid_replacement_count")
    allowed = set(allowed_tokens)
    seen: set[str] = set()
    for replacement in replacements:
        if not isinstance(replacement, dict) or set(replacement) != {
            "source_token",
            "replacement_token",
        }:
            raise ValueError("invalid_replacement_object")
        source = replacement["source_token"]
        target = replacement["replacement_token"]
        if source not in allowed:
            raise ValueError("source_token_not_offered")
        if source in seen:
            raise ValueError("duplicate_source_token")
        if not isinstance(target, str) or not target or len(target) > 64:
            raise ValueError("invalid_replacement_token")
        if WHITESPACE_RE.search(target):
            raise ValueError("replacement_token_contains_whitespace")
        if source.casefold() == target.casefold():
            raise ValueError("replacement_does_not_change_token")
        seen.add(source)
    return value


def parse_tool_decision(
    message: object, allowed_tokens: list[str], max_replacements: int
) -> dict[str, object]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls or len(tool_calls) != 1:
        raise ValueError("expected_one_tool_call")
    call = tool_calls[0]
    if call.function.name != TOOL_NAME:
        raise ValueError("unexpected_tool_name")
    return validate_decision(
        json.loads(call.function.arguments), allowed_tokens, max_replacements
    )


def build_messages(
    left: pd.Series,
    right: pd.Series,
    tokens: list[str],
    few_shots: list[dict[str, object]],
) -> list[dict[str, str]]:
    system = (
        "Ты выбираешь безопасные контрфактуальные замены для пары одинаковых товаров. "
        "Можно заменить только source_token из offered_tokens. replacement_token должен "
        "быть одним естественным токеном того же смыслового типа и формата: бренд на бренд, "
        "артикул на артикул, число на правдоподобное число. Каждая замена независима. "
        "Не выбирай служебные слова, единицы без значения и изменения, которые не меняют "
        "идентичность товара. Если уверенной замены нет, обязательно верни status=skip. "
        "Ответ верни только принудительным вызовом функции."
    )
    payload: dict[str, object] = {
        "category": str(left["category"]),
        "item_1": {"name": str(left["name"]), "attributes": parse_attributes(left["attributes"])},
        "item_2": {"name": str(right["name"]), "attributes": parse_attributes(right["attributes"])},
        "offered_tokens": tokens,
    }
    if few_shots:
        payload["category_examples"] = few_shots
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def replace_text(text: str, source: str, target: str) -> tuple[str, int]:
    escaped = re.escape(source)
    pattern = re.compile(rf"(?<![\w]){escaped}(?![\w])", flags=re.IGNORECASE)
    return pattern.subn(lambda _: target, text)


def replace_nested(value: object, source: str, target: str) -> tuple[object, int]:
    if isinstance(value, dict):
        output: dict[str, object] = {}
        count = 0
        for key, nested in value.items():
            output[key], changed = replace_nested(nested, source, target)
            count += changed
        return output, count
    if isinstance(value, list):
        output_list: list[object] = []
        count = 0
        for nested in value:
            replaced, changed = replace_nested(nested, source, target)
            output_list.append(replaced)
            count += changed
        return output_list, count
    if isinstance(value, str):
        return replace_text(value, source, target)
    return value, 0


def stable_synthetic_id(operation_key: str, side: str) -> int:
    digest = hashlib.sha256(f"{operation_key}\0{side}".encode("utf-8")).digest()
    return -(int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)) - 1


def mutate_item(
    item: pd.Series, source: str, target: str, synthetic_id: int
) -> tuple[dict[str, object], int]:
    name, name_changes = replace_text(str(item["name"]), source, target)
    attributes, attribute_changes = replace_nested(
        parse_attributes(item["attributes"]), source, target
    )
    total = name_changes + attribute_changes
    if total == 0:
        raise ValueError("replacement_not_applied")
    return {
        "id": synthetic_id,
        "name": name,
        "category": str(item["category"]),
        "attributes": json.dumps(attributes, ensure_ascii=False, separators=(",", ":")),
    }, total


def materialize(
    records: list[dict[str, object]], items: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic_items: list[dict[str, object]] = []
    synthetic_pairs: list[dict[str, object]] = []
    seen_ids: dict[int, str] = {}
    for record in records:
        decision = record.get("decision")
        if record.get("request_status") != "complete" or not isinstance(decision, dict):
            continue
        if decision.get("status") != "replace":
            continue
        id1, id2 = record["id1"], record["id2"]
        left, right = items.loc[id1], items.loc[id2]
        for replacement in decision["replacements"]:
            source = str(replacement["source_token"])
            target = str(replacement["replacement_token"])
            operation_key = hashlib.sha256(
                f"{record['source_pair_sha256']}\0{source}\0{target}".encode("utf-8")
            ).hexdigest()
            left_id = stable_synthetic_id(operation_key, "left")
            right_id = stable_synthetic_id(operation_key, "right")
            for synthetic_id in (left_id, right_id):
                previous = seen_ids.setdefault(synthetic_id, operation_key)
                if previous != operation_key:
                    raise RuntimeError("synthetic_id_collision")
            left_prime, left_changes = mutate_item(left, source, target, left_id)
            right_prime, right_changes = mutate_item(right, source, target, right_id)
            for row, side, changes in (
                (left_prime, "left", left_changes),
                (right_prime, "right", right_changes),
            ):
                row.update({
                    "source_item_id": id1 if side == "left" else id2,
                    "source_pair_sha256": record["source_pair_sha256"],
                    "source_token": source,
                    "replacement_token": target,
                    "replacement_count": changes,
                })
                synthetic_items.append(row)
            base = {
                "source_id1": id1,
                "source_id2": id2,
                "source_pair_sha256": record["source_pair_sha256"],
                "source_token": source,
                "replacement_token": target,
                "operation_sha256": operation_key,
            }
            synthetic_pairs.append({**base, "id1": left_id, "id2": right_id, "target": 1, "variant": "both_replaced"})
            replace_left = int(operation_key[-1], 16) % 2 == 0
            synthetic_pairs.append({
                **base,
                "id1": left_id if replace_left else id1,
                "id2": id2 if replace_left else right_id,
                "target": 0,
                "variant": "left_only" if replace_left else "right_only",
            })
    return pd.DataFrame(synthetic_items), pd.DataFrame(synthetic_pairs)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_few_shots(path: Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(rows, list) for rows in value.values()):
        raise ValueError("few_shots_must_map_category_to_list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/splits/matches_item_disjoint_70_30_seed42.parquet"),
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--validation-pairs", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/token_replacement_counterfactual/pilot"))
    parser.add_argument("--few-shots", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-offered-tokens", type=int, default=12)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument("--max-pair-chars", type=int, default=1200)
    parser.add_argument("--model")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.max_offered_tokens <= 0 or args.max_replacements <= 0:
        raise ValueError("count_and_limits_must_be_positive")
    if args.execute and not args.model:
        raise ValueError("--model is required with --execute")
    if args.execute and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = args.output_dir / "request_journal.jsonl"
    items = pd.read_parquet(args.items, columns=["id", "category", "name", "attributes"]).set_index("id", drop=False)
    pairs = pd.read_parquet(args.pairs)
    source_has_split = "split" in pairs.columns
    if "target" not in pairs.columns:
        if "label" not in pairs.columns:
            raise ValueError("pairs_need_target_or_label")
        pairs["target"] = pairs["label"]
    validation_ids: set[object] = set()
    if source_has_split:
        validation_rows = pairs[~pairs["split"].eq(args.source_split)]
        validation_ids.update(validation_rows["id1"])
        validation_ids.update(validation_rows["id2"])
        pairs = pairs[pairs["split"].eq(args.source_split)].copy()
    if args.validation_pairs:
        validation = pd.read_parquet(args.validation_pairs, columns=["id1", "id2"])
        validation_ids.update(validation["id1"])
        validation_ids.update(validation["id2"])
    few_shots = load_few_shots(args.few_shots)
    existing = read_jsonl(journal_path)
    completed = {str(row["source_pair_sha256"]) for row in existing}

    candidates_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    positives = pairs[pairs["target"].eq(1)].sample(frac=1, random_state=args.seed)
    for pair in positives.itertuples(index=False):
        source_hash = pair_sha256(pair.id1, pair.id2)
        if source_hash in completed or pair.id1 in validation_ids or pair.id2 in validation_ids:
            continue
        if pair.id1 not in items.index or pair.id2 not in items.index:
            continue
        left, right = items.loc[pair.id1], items.loc[pair.id2]
        if str(left["category"]) != str(right["category"]):
            continue
        if sum(len(str(row[key])) for row in (left, right) for key in ("name", "attributes")) > args.max_pair_chars:
            continue
        try:
            tokens = offered_tokens(left, right, args.max_offered_tokens)
        except (json.JSONDecodeError, ValueError):
            continue
        if not tokens:
            continue
        category = str(left["category"])
        if len(candidates_by_category[category]) >= args.count:
            continue
        candidates_by_category[category].append({
            "id1": pair.id1,
            "id2": pair.id2,
            "source_pair_sha256": source_hash,
            "category": category,
            "offered_tokens": tokens,
        })
        if sum(map(len, candidates_by_category.values())) >= args.count * 20:
            break
    candidates: list[dict[str, object]] = []
    category_offsets = defaultdict(int)
    categories = sorted(candidates_by_category)
    while len(candidates) < args.count:
        added = False
        for category in categories:
            offset = category_offsets[category]
            rows = candidates_by_category[category]
            if offset >= len(rows):
                continue
            candidates.append(rows[offset])
            category_offsets[category] += 1
            added = True
            if len(candidates) == args.count:
                break
        if not added:
            break
    if len(candidates) < args.count:
        raise RuntimeError(f"Found only {len(candidates)} eligible positive pairs")

    preview_path = args.output_dir / "request_preview.jsonl"
    preview_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    if args.execute:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=args.base_url.rstrip("/"),
            max_retries=0,
            timeout=args.timeout,
        )
        for candidate in candidates:
            left, right = items.loc[candidate["id1"]], items.loc[candidate["id2"]]
            try:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(left, right, candidate["offered_tokens"], few_shots.get(candidate["category"], [])),
                    tools=[decision_tool(candidate["offered_tokens"], args.max_replacements)],
                    tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                    temperature=0.1,
                    max_tokens=256,
                    extra_body={"enable_thinking": False},
                )
                decision = parse_tool_decision(
                    completion.choices[0].message,
                    candidate["offered_tokens"],
                    args.max_replacements,
                )
                record = {
                    **candidate,
                    "request_status": "complete",
                    "decision": decision,
                    "model": args.model,
                    "usage": completion.usage.model_dump() if completion.usage else {},
                    "provider_request_id": getattr(completion, "_request_id", None) or completion.id,
                }
            except Exception as exc:
                record = {
                    **candidate,
                    "request_status": "terminal_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "model": args.model,
                }
            append_jsonl(journal_path, record)

    records = read_jsonl(journal_path)
    synthetic_items, synthetic_pairs = materialize(records, items)
    output_files: dict[str, dict[str, object]] = {}
    if not synthetic_items.empty:
        items_path = args.output_dir / "synthetic_items.parquet"
        pairs_path = args.output_dir / "synthetic_pairs.parquet"
        synthetic_items.to_parquet(items_path, index=False)
        synthetic_pairs.to_parquet(pairs_path, index=False)
        for path in (items_path, pairs_path):
            output_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "status": "complete" if args.execute else "dry_run_complete",
        "execute": args.execute,
        "source": "human_positive_pairs_only",
        "source_pairs_sha256": sha256_file(args.pairs),
        "source_split": args.source_split if source_has_split else None,
        "validation_item_denylist_size": len(validation_ids),
        "provider_payload_fields": ["category", "name_1", "attributes_1", "name_2", "attributes_2", "offered_tokens"],
        "ids_sent_to_provider": False,
        "labels_sent_to_provider": False,
        "validation_sent_to_provider": False,
        "eligible_candidates": len(candidates),
        "journal_records": len(records),
        "synthetic_items": len(synthetic_items),
        "synthetic_pairs": len(synthetic_pairs),
        "positive_synthetic_pairs": int(synthetic_pairs["target"].eq(1).sum()) if not synthetic_pairs.empty else 0,
        "negative_synthetic_pairs": int(synthetic_pairs["target"].eq(0).sum()) if not synthetic_pairs.empty else 0,
        "request_preview_sha256": sha256_file(preview_path),
        "outputs": output_files,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
