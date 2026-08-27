"""Judge deterministic identity bundles grounded in complete human-train cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from alibaba_model_quota_router import ModelQuotaRouter
from generate_token_replacement_aggressive_rag import call_chat_rotating
from generate_token_replacement_counterfactuals import (
    DEFAULT_BASE_URL,
    append_jsonl,
    parse_attributes,
    read_jsonl,
    replace_text,
    sha256_file,
)
from generate_token_replacement_fast_v2 import (
    attribute_leaves,
    context_tokens,
    public_item,
)
from repair_token_replacement_identity_bundles import balanced_sample, risky_operations


TOOL_NAME = "select_grounded_identity_bundles"
PIPELINE_VERSION = 1
IDENTITY_VALUE_LIMIT = 96
IDENTITY_KINDS = ("brand", "model", "article")
MODEL_KEY_RE = re.compile(r"(?:^|\s)(?:модель|model|серия|линейка)(?:$|\s)")
ARTICLE_KEY_RE = re.compile(
    r"артик|партномер|part number|oem|oe-код|код товар|код продукт|номер производ"
)
FORBIDDEN_MODEL_KEY_PARTS = (
    "автомобил", "совместим", "поддерживаем", "назначение", "тип",
)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+._/-]*")
SAFE_RESIDUAL_LATIN = {
    "black", "white", "red", "blue", "green", "gray", "grey", "silver",
    "gold", "pink", "beige", "brown", "orange", "yellow", "purple",
    "transparent", "dark", "light", "matte", "glossy", "for", "with",
}
PROTECTED_SPEC_KEY_PARTS = (
    "цвет", "размер", "памят", "объем", "объём", "емкость", "ёмкость",
    "мощност", "напряж", "комплектац", "количество", "вес", "масса",
    "длина", "ширина", "высота", "диаметр", "разрешение", "процессор",
)
CYRILLIC_TO_LATIN = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
})


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def normalize_key(value: str) -> str:
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", value.casefold()).split())


def identity_kind(key: str) -> str | None:
    normalized = normalize_key(key)
    if any(value in normalized for value in ("бренд", "brand", "торговая марка")):
        return "brand"
    if MODEL_KEY_RE.search(normalized) and not any(
        value in normalized for value in FORBIDDEN_MODEL_KEY_PARTS
    ):
        return "model"
    if ARTICLE_KEY_RE.search(normalized):
        return "article"
    return None


def clean_identity_value(value: object) -> str | None:
    text = str(value).strip()
    if not 1 < len(text) <= IDENTITY_VALUE_LIMIT or "\n" in text:
        return None
    if text.casefold() in {"нет", "не указан", "без бренда", "noname", "no name"}:
        return None
    if len(re.findall(r"[,;|]", text)) > 1:
        return None
    return text


def identity_fields(item: pd.Series | dict[str, object]) -> dict[str, list[tuple[str, str]]]:
    attributes = parse_attributes(item["attributes"])
    output: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for key, value in attribute_leaves(attributes):
        kind = identity_kind(key)
        cleaned = clean_identity_value(value)
        if kind is None or cleaned is None:
            continue
        marker = (kind, cleaned.casefold())
        if marker in seen:
            continue
        seen.add(marker)
        output[kind].append((normalize_key(key), cleaned))
    return dict(output)


def contains(text: str, span: str) -> bool:
    return replace_text(text, span, "__probe__")[1] > 0


def item_text(item: pd.Series | dict[str, object]) -> str:
    return f"{item['name']} {item['attributes']}"


def pair_text(left: pd.Series, right: pd.Series) -> str:
    return f"{item_text(left)} {item_text(right)}"


def brand_values(fields: dict[str, list[tuple[str, str]]]) -> set[str]:
    return {value.casefold() for _, value in fields.get("brand", [])}


def name_context(item: pd.Series | dict[str, object]) -> set[str]:
    identity = {
        token
        for values in identity_fields(item).values()
        for _, value in values
        for token in context_tokens(value)
    }
    return context_tokens(str(item["name"])) - identity


def target_card_candidates(
    row: dict[str, object], items: pd.DataFrame, name_index: dict[tuple[str, str], list[int]]
) -> list[pd.Series]:
    key = (str(row["category"]), str(row["evidence_name"]).casefold())
    candidates: list[tuple[int, int, pd.Series]] = []
    left = items.loc[int(row["id1"])]
    right = items.loc[int(row["id2"])]
    source_context = name_context(left) | name_context(right)
    source_brands = brand_values(identity_fields(left)) | brand_values(identity_fields(right))
    for item_id in name_index.get(key, []):
        target = items.loc[item_id]
        fields = identity_fields(target)
        role_values = fields.get(str(row["source_role"]), [])
        if not any(contains(value, str(row["replacement_token"])) for _, value in role_values):
            continue
        if row["source_role"] == "article" and source_brands:
            if not (source_brands & brand_values(fields)):
                continue
        overlap = len(source_context & name_context(target))
        if overlap == 0:
            continue
        candidates.append((-overlap, int(item_id), target))
    return [target for _, _, target in sorted(candidates, key=lambda value: value[:2])]


def comparable_target(
    source_values: list[tuple[str, str]], target_values: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    target_by_key: dict[str, list[str]] = defaultdict(list)
    for key, value in target_values:
        target_by_key[key].append(value)
    output: list[tuple[str, str]] = []
    for source_key, source_value in source_values:
        exact = target_by_key.get(source_key, [])
        if len(exact) == 1:
            output.append((source_value, exact[0]))
        elif len(source_values) == 1 and len(target_values) == 1:
            output.append((source_value, target_values[0][1]))
    return output


def build_bundle(
    row: dict[str, object], left: pd.Series, right: pd.Series, target: pd.Series
) -> dict[str, object] | None:
    source_fields = {
        kind: [
            *identity_fields(left).get(kind, []),
            *identity_fields(right).get(kind, []),
        ]
        for kind in IDENTITY_KINDS
    }
    target_fields = identity_fields(target)
    mappings: list[dict[str, str]] = [{
        "kind": str(row["source_role"]),
        "source_span": str(row["source_token"]),
        "target_span": str(row["replacement_token"]),
    }]
    if row["source_role"] == "brand":
        dependent_kinds = ("model", "article")
    else:
        dependent_kinds = ("model", "article")
    for kind in dependent_kinds:
        for source, replacement in comparable_target(
            source_fields.get(kind, []), target_fields.get(kind, [])
        ):
            if source.casefold() == replacement.casefold():
                continue
            mappings.append({"kind": kind, "source_span": source, "target_span": replacement})
    clean: list[dict[str, str]] = []
    seen: set[str] = set()
    source_all_text = pair_text(left, right)
    target_all_text = item_text(target)
    for mapping in sorted(mappings, key=lambda value: -len(value["source_span"])):
        source = mapping["source_span"]
        replacement = mapping["target_span"]
        if source.casefold() in seen or not contains(source_all_text, source):
            continue
        if not contains(target_all_text, replacement):
            continue
        seen.add(source.casefold())
        clean.append(mapping)
    required = (
        str(row["source_token"]).casefold(),
        str(row["replacement_token"]).casefold(),
    )
    if not any(
        (mapping["source_span"].casefold(), mapping["target_span"].casefold()) == required
        for mapping in clean
    ):
        return None
    source_dependents = [
        value
        for kind in ("model", "article")
        for _, value in source_fields.get(kind, [])
        if value.casefold() != str(row["source_token"]).casefold()
    ]
    if row["source_role"] == "brand" and source_dependents and len(clean) == 1:
        return None
    bundle_id = "bundle_" + stable_hash({
        "operation": row["operation_sha256"],
        "target_name": str(target["name"]),
        "mappings": clean,
    })[:16]
    return {
        "bundle_id": bundle_id,
        "target_card": public_item(target, max_attribute_chars=1100),
        "mappings": clean,
    }


def replace_identity_attributes(value: object, mappings: list[dict[str, str]], path: tuple[str, ...] = ()) -> object:
    if isinstance(value, dict):
        return {
            key: replace_identity_attributes(nested, mappings, (*path, str(key)))
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [replace_identity_attributes(nested, mappings, path) for nested in value]
    if identity_kind(" ".join(path)) is None:
        return value
    text = str(value)
    for mapping in mappings:
        text = replace_text(text, mapping["source_span"], mapping["target_span"])[0]
    return text


def apply_bundle(item: pd.Series, mappings: list[dict[str, str]]) -> dict[str, object]:
    name = str(item["name"])
    for mapping in mappings:
        name = replace_text(name, mapping["source_span"], mapping["target_span"])[0]
    attributes = parse_attributes(item["attributes"])
    changed_attributes = replace_identity_attributes(attributes, mappings)
    if name == str(item["name"]) and changed_attributes == attributes:
        raise ValueError("grounded_bundle_not_applied")
    return {"name": name, "attributes": changed_attributes}


def latinize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold().translate(CYRILLIC_TO_LATIN))


def local_bundle_rejection(
    row: dict[str, object], bundle: dict[str, object], left: pd.Series, right: pd.Series
) -> str | None:
    replacement = str(row["replacement_token"])
    target_name = str(bundle["target_card"]["name"])
    if not contains(target_name, replacement):
        return "replacement_missing_from_target_name"
    source_specs: dict[str, set[str]] = defaultdict(set)
    target_specs: dict[str, set[str]] = defaultdict(set)
    for item, destination in ((left, source_specs), (right, source_specs), (bundle["target_card"], target_specs)):
        attributes = item["attributes"]
        if isinstance(attributes, str):
            attributes = parse_attributes(attributes)
        for key, value in attribute_leaves(attributes):
            normalized_key = normalize_key(key)
            if any(fragment in normalized_key for fragment in PROTECTED_SPEC_KEY_PARTS):
                destination[normalized_key].add(normalize_key(str(value)))
    for key in source_specs.keys() & target_specs.keys():
        if source_specs[key].isdisjoint(target_specs[key]):
            return "grounded_target_spec_conflict:" + key
    if row["source_role"] != "brand":
        return None
    source_brand = latinize(str(row["source_token"]))
    preview_text = f"{bundle['preview_name_1']} {bundle['preview_name_2']}"
    for token in re.findall(r"[A-Za-zА-Яа-я0-9+._/-]+", preview_text):
        if source_brand and latinize(token) == source_brand:
            return "old_brand_alias_remains"
    source_latin = {
        token.casefold()
        for token in LATIN_TOKEN_RE.findall(f"{left['name']} {right['name']}")
    }
    target_latin = {token.casefold() for token in LATIN_TOKEN_RE.findall(target_name)}
    mapped_source_latin = {
        token.casefold()
        for mapping in bundle["mappings"]
        for token in LATIN_TOKEN_RE.findall(mapping["source_span"])
    }
    unsupported = source_latin - target_latin - mapped_source_latin - SAFE_RESIDUAL_LATIN
    if unsupported:
        return "unsupported_source_identity_tokens:" + ",".join(sorted(unsupported)[:5])
    return None


def prepare_rows(
    generation_dir: Path, source_pack: Path, items: pd.DataFrame
) -> tuple[list[dict[str, object]], Counter[str]]:
    risky = risky_operations(generation_dir, source_pack, items)
    name_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in items.itertuples(index=False):
        name_index[(str(item.category), str(item.name).casefold())].append(int(item.id))
    output: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    for row in risky:
        left = items.loc[int(row["id1"])]
        right = items.loc[int(row["id2"])]
        cards = target_card_candidates(row, items, name_index)
        if not cards:
            rejected["no_complete_compatible_target_card"] += 1
            continue
        bundles: list[dict[str, object]] = []
        seen: set[str] = set()
        for target in cards[:4]:
            bundle = build_bundle(row, left, right, target)
            if bundle is None or bundle["bundle_id"] in seen:
                continue
            seen.add(str(bundle["bundle_id"]))
            try:
                preview_1 = apply_bundle(left, bundle["mappings"])
                preview_2 = apply_bundle(right, bundle["mappings"])
            except ValueError:
                continue
            bundle["preview_name_1"] = preview_1["name"]
            bundle["preview_name_2"] = preview_2["name"]
            rejection = local_bundle_rejection(row, bundle, left, right)
            if rejection is not None:
                rejected[rejection.split(":", 1)[0]] += 1
                continue
            bundles.append(bundle)
        if not bundles:
            rejected["no_deterministic_grounded_bundle"] += 1
            continue
        output.append({
            **row,
            "item_1": public_item(left, max_attribute_chars=1100),
            "item_2": public_item(right, max_attribute_chars=1100),
            "bundles": bundles,
        })
    return output, rejected


def decision_schema(slot: int, row: dict[str, object]) -> dict[str, object]:
    bundle_ids = [str(bundle["bundle_id"]) for bundle in row["bundles"]]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot": {"type": "integer", "const": slot},
            "status": {"type": "string", "enum": ["choose", "skip"]},
            "bundle_id": {"type": "string", "enum": ["", *bundle_ids]},
            "reason": {"type": "string", "maxLength": 200},
        },
        "required": ["slot", "status", "bundle_id", "reason"],
    }


def selection_tool(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Choose one grounded identity bundle or skip each row.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": len(rows),
                        "maxItems": len(rows),
                        "items": {"oneOf": [
                            decision_schema(slot, row) for slot, row in enumerate(rows)
                        ]},
                    }
                },
                "required": ["decisions"],
            },
        },
    }


def messages(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    system = (
        "Ты проверяешь готовые пакеты замены идентичности marketplace-товаров. Код уже "
        "построил каждый пакет только из полной реальной human-train карточки и запретил "
        "изменение характеристик. Выбери bundle_id, только если новая пара названий остаётся "
        "естественной, тип товара совпадает, а бренд, модель и артикул не противоречат друг "
        "другу. Нельзя предлагать свои строки или mappings. Если есть сомнение — skip. "
        "Ответ только вызовом функции, ровно по одному решению на slot."
    )
    payload = []
    for slot, row in enumerate(rows):
        payload.append({
            "slot": slot,
            "category": row["category"],
            "source_role": row["source_role"],
            "item_1": row["item_1"],
            "item_2": row["item_2"],
            "bundles": row["bundles"],
        })
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"rows": payload}, ensure_ascii=False)},
    ]


def parse_decisions(message: object, rows: list[dict[str, object]]) -> list[dict[str, str]]:
    calls = getattr(message, "tool_calls", None)
    if not calls or len(calls) != 1 or calls[0].function.name != TOOL_NAME:
        raise ValueError("expected_one_bundle_selection_call")
    value = json.loads(calls[0].function.arguments)
    decisions = value.get("decisions") if isinstance(value, dict) else None
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise ValueError("bundle_decision_count_mismatch")
    output: list[dict[str, str] | None] = [None] * len(rows)
    for decision in decisions:
        slot = decision.get("slot") if isinstance(decision, dict) else None
        if isinstance(slot, str) and slot.isdigit():
            slot = int(slot)
        if not isinstance(slot, int) or not 0 <= slot < len(rows) or output[slot] is not None:
            raise ValueError("invalid_bundle_slot")
        status = decision.get("status")
        bundle_id = str(decision.get("bundle_id", ""))
        allowed = {str(bundle["bundle_id"]) for bundle in rows[slot]["bundles"]}
        if status == "choose" and bundle_id not in allowed:
            raise ValueError("bundle_not_offered")
        if status == "skip":
            bundle_id = ""
        if status not in {"choose", "skip"}:
            raise ValueError("invalid_bundle_status")
        output[slot] = {
            "status": str(status),
            "bundle_id": bundle_id,
            "reason": str(decision.get("reason", ""))[:200],
        }
    if any(decision is None for decision in output):
        raise ValueError("missing_bundle_slot")
    return [decision for decision in output if decision is not None]


def process_batch(
    batch: list[dict[str, object]], client: Any, router: ModelQuotaRouter, offset: int
) -> dict[str, object]:
    started = time.perf_counter()

    def request(model: str) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages(batch),
            tools=[selection_tool(batch)],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            temperature=0.0,
            max_tokens=900,
            extra_body={"enable_thinking": False},
        )

    completion, model, error = call_chat_rotating(
        client, router, offset=offset, excluded=set(), request=request
    )
    batch_hash = stable_hash([row["operation_sha256"] for row in batch])
    if error:
        return {
            "request_status": "terminal_error",
            "batch_sha256": batch_hash,
            "operation_sha256s": [row["operation_sha256"] for row in batch],
            "model": model,
            "error": error,
            "elapsed_seconds": time.perf_counter() - started,
        }
    assert completion is not None
    try:
        decisions = parse_decisions(completion.choices[0].message, batch)
    except Exception as exc:
        return {
            "request_status": "terminal_error",
            "batch_sha256": batch_hash,
            "operation_sha256s": [row["operation_sha256"] for row in batch],
            "model": model,
            "error": {"classification": "invalid_structured_output", "error": str(exc)},
            "elapsed_seconds": time.perf_counter() - started,
        }
    results = []
    for row, decision in zip(batch, decisions):
        selected = next(
            (bundle for bundle in row["bundles"] if bundle["bundle_id"] == decision["bundle_id"]),
            None,
        )
        results.append({
            "operation_sha256": row["operation_sha256"],
            "category": row["category"],
            "source_role": row["source_role"],
            "decision": decision,
            "selected_bundle": selected,
        })
    return {
        "request_status": "complete",
        "batch_sha256": batch_hash,
        "operation_sha256s": [row["operation_sha256"] for row in batch],
        "model": model,
        "usage": completion.usage.model_dump() if completion.usage else {},
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--model-registry", type=Path, action="append", required=True)
    parser.add_argument("--exclude-journal", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.workers) <= 0 or args.count < 0:
        raise ValueError("invalid_limits")
    if args.execute and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id", drop=False)
    prepared, deterministic_rejections = prepare_rows(
        args.generation_dir, args.source_pack, items
    )
    attempted: set[str] = set()
    journal = args.output_dir / "selection_journal.jsonl"
    for path in [journal, *args.exclude_journal]:
        for record in read_jsonl(path):
            attempted.update(str(value) for value in record.get("operation_sha256s", []))
    selected = [] if args.count == 0 else balanced_sample(
        [row for row in prepared if row["operation_sha256"] not in attempted], args.count
    )
    preview = [{
        "operation_sha256": row["operation_sha256"],
        "category": row["category"],
        "source_role": row["source_role"],
        "item_1": row["item_1"],
        "item_2": row["item_2"],
        "bundles": row["bundles"],
    } for row in selected]
    (args.output_dir / "request_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.execute and selected:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=args.base_url.rstrip("/"),
            max_retries=0,
            timeout=args.timeout,
        )
        router = ModelQuotaRouter(args.model_registry, args.output_dir / "model_events.jsonl")
        batches = [selected[index:index + args.batch_size] for index in range(0, len(selected), args.batch_size)]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for index, batch in enumerate(batches):
                submitted = {
                    "request_status": "submitted",
                    "batch_sha256": stable_hash([row["operation_sha256"] for row in batch]),
                    "operation_sha256s": [row["operation_sha256"] for row in batch],
                }
                append_jsonl(journal, submitted)
                futures[executor.submit(process_batch, batch, client, router, index)] = submitted
            for future in as_completed(futures):
                append_jsonl(journal, future.result())
    records = read_jsonl(journal)
    results = [
        result
        for record in records if record.get("request_status") == "complete"
        for result in record.get("results", [])
    ]
    prepared_by_operation = {
        str(row["operation_sha256"]): {
            str(bundle["bundle_id"]): bundle for bundle in row["bundles"]
        }
        for row in prepared
    }
    chosen = []
    stale_chosen = 0
    for row in results:
        if row["decision"]["status"] != "choose":
            continue
        current_bundle = prepared_by_operation.get(str(row["operation_sha256"]), {}).get(
            str(row["decision"]["bundle_id"])
        )
        if current_bundle is None:
            stale_chosen += 1
            continue
        row["selected_bundle"] = current_bundle
        chosen.append(row)
    with (args.output_dir / "chosen_bundles.jsonl").open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "source_generation_pairs_sha256": sha256_file(args.generation_dir / "synthetic_pairs.parquet"),
        "source_pack_sha256": sha256_file(args.source_pack),
        "grounded_candidates_total": len(prepared),
        "deterministic_rejections": dict(sorted(deterministic_rejections.items())),
        "selected_this_run": len(selected),
        "completed_results": len(results),
        "chosen": len(chosen),
        "chosen_removed_by_stricter_local_audit": stale_chosen,
        "skipped": len(results) - len(chosen),
        "terminal_batches": sum(record.get("request_status") == "terminal_error" for record in records),
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
