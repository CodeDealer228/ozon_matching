"""Repair accepted brand/article counterfactuals with minimal identity bundles."""

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
from typing import Any

import pandas as pd

from alibaba_model_quota_router import ModelQuotaRouter
from generate_token_replacement_aggressive_rag import call_chat_rotating
from generate_token_replacement_counterfactuals import (
    DEFAULT_BASE_URL,
    append_jsonl,
    parse_attributes,
    read_jsonl,
    replace_nested,
    replace_text,
    sha256_file,
)
from generate_token_replacement_fast_v2 import (
    ReplacementBank,
    build_operations,
    flatten_complete,
    infer_source_role,
    public_item,
)


TOOL_NAME = "submit_identity_bundle_repairs"
PROMPT_VERSION = 1
WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9+./_-]*")


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def span_candidates(text: str, *, max_words: int = 3, limit: int = 80) -> list[str]:
    words = WORD_RE.findall(text)
    spans: list[str] = []
    seen: set[str] = set()
    for size in range(1, max_words + 1):
        for index in range(len(words) - size + 1):
            span = " ".join(words[index:index + size]).strip()
            folded = span.casefold()
            if len(span) < 2 or folded in seen:
                continue
            seen.add(folded)
            spans.append(span)
            if len(spans) >= limit:
                return spans
    return spans


def offered_source_spans(
    source_row: dict[str, object], left: pd.Series, right: pd.Series, source_token: str
) -> list[str]:
    values = [source_token]
    for token in source_row.get("offered_tokens", []):
        token = str(token)
        if infer_source_role(left, right, token) in {"brand", "article"}:
            values.append(token)
    for item in (left, right):
        attributes = parse_attributes(item["attributes"])
        for key, value in attributes.items():
            folded_key = str(key).casefold()
            if any(part in folded_key for part in ("бренд", "модель", "артик", "партномер")):
                values.extend(span_candidates(str(value), max_words=3, limit=12))
    output: list[str] = []
    seen: set[str] = set()
    combined = f"{left['name']} {left['attributes']} {right['name']} {right['attributes']}"
    for value in values:
        value = str(value).strip()
        folded = value.casefold()
        if not value or folded in seen:
            continue
        if replace_text(combined, value, "__probe__")[1] == 0:
            continue
        seen.add(folded)
        output.append(value)
    return output[:30]


def risky_operations(
    generation_dir: Path,
    source_pack: Path,
    items: pd.DataFrame,
) -> list[dict[str, object]]:
    source_rows = read_jsonl(source_pack)
    source_by_hash = {str(row["source_pair_sha256"]): row for row in source_rows}
    bank = ReplacementBank(source_rows, items)
    pair_ops = pd.read_parquet(generation_dir / "synthetic_pairs.parquet").drop_duplicates(
        "operation_sha256"
    )
    output: list[dict[str, object]] = []
    for row in pair_ops.itertuples(index=False):
        source_row = source_by_hash[str(row.source_pair_sha256)]
        left, right = items.loc[int(row.source_id1)], items.loc[int(row.source_id2)]
        candidates = build_operations(source_row, items, bank, 4, 6)
        matched = next(
            (
                operation
                for operation in candidates
                if str(operation["source_token"]).casefold() == str(row.source_token).casefold()
                and str(operation["replacement_token"]).casefold()
                == str(row.replacement_token).casefold()
            ),
            None,
        )
        if matched is None:
            raise RuntimeError(f"operation_metadata_not_reconstructed:{row.operation_sha256}")
        role = str(matched["source_role"])
        if role not in {"brand", "article"}:
            continue
        evidence_name = str(matched.get("evidence_name", "")).strip()
        target_spans = span_candidates(evidence_name, max_words=3, limit=80)
        if str(row.replacement_token).casefold() not in {value.casefold() for value in target_spans}:
            target_spans.insert(0, str(row.replacement_token))
        output.append({
            "operation_sha256": str(row.operation_sha256),
            "source_pair_sha256": str(row.source_pair_sha256),
            "id1": int(row.source_id1),
            "id2": int(row.source_id2),
            "category": str(left["category"]),
            "source_role": role,
            "source_token": str(row.source_token),
            "replacement_token": str(row.replacement_token),
            "evidence_name": evidence_name,
            "source_spans": offered_source_spans(
                source_row, left, right, str(row.source_token)
            ),
            "target_spans": target_spans,
            "item_1": public_item(left, max_attribute_chars=1100),
            "item_2": public_item(right, max_attribute_chars=1100),
        })
    return output


def balanced_sample(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0 or count >= len(rows):
        return rows
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["category"]), str(row["source_role"]))].append(row)
    keys = sorted(groups)
    selected: list[dict[str, object]] = []
    offsets: Counter[tuple[str, str]] = Counter()
    while len(selected) < count:
        added = False
        for key in keys:
            index = offsets[key]
            if index >= len(groups[key]):
                continue
            selected.append(groups[key][index])
            offsets[key] += 1
            added = True
            if len(selected) == count:
                break
        if not added:
            break
    return selected


def slot_schema(slot: int, row: dict[str, object]) -> dict[str, object]:
    mapping = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_span": {"type": "string", "enum": row["source_spans"]},
            "target_span": {"type": "string", "enum": row["target_spans"]},
        },
        "required": ["source_span", "target_span"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot": {"type": "integer", "const": slot},
            "status": {"type": "string", "enum": ["keep", "repair", "skip"]},
            "mappings": {"type": "array", "maxItems": 4, "items": mapping},
            "reason": {"type": "string", "maxLength": 240},
        },
        "required": ["slot", "status", "mappings", "reason"],
    }


def repair_tool(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Repair identity-only replacements without changing specifications.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": len(rows),
                        "maxItems": len(rows),
                        "items": {"oneOf": [slot_schema(i, row) for i, row in enumerate(rows)]},
                    }
                },
                "required": ["decisions"],
            },
        },
    }


def messages(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    system = (
        "You repair marketplace counterfactual identity replacements. For every slot choose keep, "
        "repair, or skip. Change only lexical identity: brand, product family, model, or article. "
        "Never change memory, color, size, quantity, region, processor, screen, battery, camera, or "
        "other specifications. Every mapping must use exactly one offered source_span and one offered "
        "target_span grounded in the human-train evidence name. For a brand replacement, also replace "
        "dependent family/model spans when they would otherwise belong to the old brand. Preserve "
        "missing fields: do not add a model to a card that did not mention it. Choose skip when the "
        "evidence is insufficient or a coherent minimal identity bundle cannot be formed. Choose keep "
        "only when the existing one-token operation is already coherent. Return only the tool call."
    )
    payload = []
    for slot, row in enumerate(rows):
        payload.append({
            "slot": slot,
            "category": row["category"],
            "source_role": row["source_role"],
            "existing_operation": {
                "source_token": row["source_token"],
                "replacement_token": row["replacement_token"],
            },
            "item_1": row["item_1"],
            "item_2": row["item_2"],
            "human_train_target_evidence_name": row["evidence_name"],
            "offered_source_spans": row["source_spans"],
            "offered_target_spans": row["target_spans"],
        })
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"rows": payload}, ensure_ascii=False)},
    ]


def parse_decisions(message: object, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    calls = getattr(message, "tool_calls", None)
    if not calls or len(calls) != 1 or calls[0].function.name != TOOL_NAME:
        raise ValueError("expected_one_repair_tool_call")
    value = json.loads(calls[0].function.arguments)
    decisions = value.get("decisions") if isinstance(value, dict) else None
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise ValueError("repair_decision_count_mismatch")
    output: list[dict[str, object] | None] = [None] * len(rows)
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("invalid_repair_decision")
        slot = decision.get("slot")
        if isinstance(slot, str) and slot.isdigit():
            slot = int(slot)
        if not isinstance(slot, int) or not 0 <= slot < len(rows) or output[slot] is not None:
            raise ValueError("invalid_repair_slot")
        status = decision.get("status")
        mappings = decision.get("mappings")
        if status not in {"keep", "repair", "skip"} or not isinstance(mappings, list):
            raise ValueError("invalid_repair_fields")
        if status == "keep":
            mappings = [{
                "source_span": rows[slot]["source_token"],
                "target_span": rows[slot]["replacement_token"],
            }]
        if status == "skip":
            mappings = []
        if status == "repair" and not mappings:
            status = "skip"
        allowed_source = set(rows[slot]["source_spans"])
        allowed_target = set(rows[slot]["target_spans"])
        clean: list[dict[str, str]] = []
        seen: set[str] = set()
        for mapping in mappings:
            if not isinstance(mapping, dict):
                raise ValueError("invalid_mapping")
            source = mapping.get("source_span")
            target = mapping.get("target_span")
            if source not in allowed_source or target not in allowed_target:
                raise ValueError("mapping_not_offered")
            if source.casefold() in seen or source.casefold() == target.casefold():
                continue
            seen.add(source.casefold())
            clean.append({"source_span": source, "target_span": target})
        output[slot] = {
            "status": status if clean or status == "skip" else "skip",
            "mappings": clean,
            "reason": str(decision.get("reason", ""))[:240],
        }
    if any(value is None for value in output):
        raise ValueError("missing_repair_slot")
    return [value for value in output if value is not None]


def apply_mappings(item: pd.Series, mappings: list[dict[str, str]]) -> dict[str, object]:
    name = str(item["name"])
    attributes: object = parse_attributes(item["attributes"])
    changes = 0
    for mapping in sorted(mappings, key=lambda value: -len(value["source_span"])):
        name, count = replace_text(name, mapping["source_span"], mapping["target_span"])
        changes += count
        attributes, count = replace_nested(
            attributes, mapping["source_span"], mapping["target_span"]
        )
        changes += count
    if changes == 0:
        raise ValueError("repair_not_applied")
    return {"name": name, "attributes": attributes}


def process_batch(
    batch: list[dict[str, object]],
    items: pd.DataFrame,
    client: Any,
    router: ModelQuotaRouter,
    offset: int,
) -> dict[str, object]:
    started = time.perf_counter()

    def request(model: str) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages(batch),
            tools=[repair_tool(batch)],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            temperature=0.0,
            max_tokens=1400,
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
    results: list[dict[str, object]] = []
    for row, decision in zip(batch, decisions):
        if decision["status"] != "skip":
            try:
                left = apply_mappings(items.loc[row["id1"]], decision["mappings"])
                right = apply_mappings(items.loc[row["id2"]], decision["mappings"])
                decision["preview_name_1"] = left["name"]
                decision["preview_name_2"] = right["name"]
            except ValueError as exc:
                decision = {
                    "status": "skip",
                    "mappings": [],
                    "reason": str(exc),
                }
        results.append({
            "operation_sha256": row["operation_sha256"],
            "category": row["category"],
            "source_role": row["source_role"],
            "existing_operation": {
                "source_token": row["source_token"],
                "replacement_token": row["replacement_token"],
            },
            "evidence_name": row["evidence_name"],
            "decision": decision,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
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
    all_risky_rows = risky_operations(args.generation_dir, args.source_pack, items)
    journal = args.output_dir / "repair_journal.jsonl"
    existing = read_jsonl(journal)
    attempted = {
        str(value)
        for record in existing
        for value in record.get("operation_sha256s", [])
    }
    rows = [row for row in all_risky_rows if row["operation_sha256"] not in attempted]
    selected = balanced_sample(rows, args.count)
    preview = [{key: row[key] for key in (
        "operation_sha256", "category", "source_role", "source_token",
        "replacement_token", "evidence_name", "source_spans", "target_spans",
    )} for row in selected[:20]]
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
                batch_hash = stable_hash([row["operation_sha256"] for row in batch])
                append_jsonl(journal, {
                    "request_status": "submitted",
                    "batch_sha256": batch_hash,
                    "operation_sha256s": [row["operation_sha256"] for row in batch],
                })
                futures[executor.submit(process_batch, batch, items, client, router, index)] = batch_hash
            for future in as_completed(futures):
                append_jsonl(journal, future.result())
    records = read_jsonl(journal)
    complete = [record for record in records if record.get("request_status") == "complete"]
    results = [result for record in complete for result in record.get("results", [])]
    counts = Counter(str(row["decision"]["status"]) for row in results)
    manifest = {
        "status": "complete",
        "prompt_version": PROMPT_VERSION,
        "source_generation_pairs_sha256": sha256_file(args.generation_dir / "synthetic_pairs.parquet"),
        "source_pack_sha256": sha256_file(args.source_pack),
        "risky_operations_total": len(all_risky_rows),
        "selected_this_run": len(selected),
        "terminal_batches": sum(record.get("request_status") == "terminal_error" for record in records),
        "completed_results": len(results),
        "decision_counts": dict(sorted(counts.items())),
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
    }
    (args.output_dir / "repair_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
