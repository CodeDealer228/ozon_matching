"""Fast quality-first token replacement with batched generator and validator calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from alibaba_model_quota_router import ModelQuotaRouter
from generate_token_replacement_counterfactuals import (
    DEFAULT_BASE_URL,
    append_jsonl,
    load_few_shots,
    materialize,
    read_jsonl,
    sha256_file,
)
from generate_token_replacement_aggressive_rag import compact_rag_item, semantic_type


TOOL_NAME = "submit_batched_replacement_decisions"
CONTEXT_RE = re.compile(r"[A-Za-zА-Яа-я0-9-]{3,}")
PROMPT_VERSION = 1
CANDIDATE_BANK_VERSION = 1

GENERATOR_PROMPT = (
    "Ты выбираешь контрфактуальные замены сразу для пакета положительных пар одинаковых "
    "товаров. Для каждого slot можно выбрать до двух operation_id только из его списка. "
    "Валидная операция обязана менять идентичность товара: бренд, модель, артикул, размер, "
    "объём или другую существенную характеристику. Односторонняя замена создаст negative "
    "разных товаров, а двусторонняя — новый internally-consistent positive. Кандидаты взяты "
    "из реальных human-train карточек той же категории, но не каждый подходит данному типу "
    "товара. Выбирай только естественную замену того же смыслового типа и формата; иначе skip. "
    "Не пытайся исправлять исходную пару и не требуй сохранять исходный SKU. Ответ — только "
    "вызов функции. Верни решение ровно для каждого slot."
)

VALIDATOR_PROMPT = (
    "Ты независимо проверяешь выбранные контрфактуальные операции для положительных пар. "
    "Для каждого slot оставь до двух operation_id только из candidate_operations. Смена SKU, "
    "бренда, модели, артикула, размера или объёма является желаемым эффектом: одна изменённая "
    "карточка даёт negative, обе изменённые карточки дают новый positive. Отклоняй операцию, "
    "если replacement неправдоподобен для типа товара, другого смыслового типа, искусственен "
    "или создаёт внутреннее противоречие внутри изменённой карточки. Не отклоняй только потому, "
    "что товар становится другим. Ответ — только вызов функции для каждого slot."
)


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def context_tokens(value: str) -> set[str]:
    return {token.casefold() for token in CONTEXT_RE.findall(value)}


def public_item(item: pd.Series, max_attribute_chars: int = 700) -> dict[str, object]:
    return compact_rag_item(
        {"name": str(item["name"]), "attributes": json.loads(str(item["attributes"]))},
        max_attribute_chars=max_attribute_chars,
    )


def compact_rag(row: dict[str, object]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for relation in ("same_product", "different_product"):
        match = next(
            (example for example in row["rag_examples"] if example["relation"] == relation),
            None,
        )
        if match is None:
            continue
        selected.append({
            "relation": relation,
            "item_1": compact_rag_item(match["item_1"], max_attribute_chars=500),
            "item_2": compact_rag_item(match["item_2"], max_attribute_chars=500),
        })
    return selected


@dataclass(frozen=True)
class BankEntry:
    token: str
    token_type: str
    context: frozenset[str]
    source_pair_sha256: str


class ReplacementBank:
    def __init__(self, source_rows: list[dict[str, object]], items: pd.DataFrame):
        self.entries: dict[tuple[str, str], list[BankEntry]] = defaultdict(list)
        self.postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self.frequencies: Counter[tuple[str, str, str]] = Counter()
        for row in source_rows:
            category = str(row["category"])
            left = items.loc[row["id1"]]
            right = items.loc[row["id2"]]
            context = frozenset(
                context_tokens(f"{left['name']} {right['name']}")
                - {str(token).casefold() for token in row["offered_tokens"]}
            )
            for token_value in row["offered_tokens"]:
                token = str(token_value)
                token_type = semantic_type(token)
                key = (category, token_type)
                index = len(self.entries[key])
                self.entries[key].append(
                    BankEntry(token, token_type, context, str(row["source_pair_sha256"]))
                )
                self.frequencies[(category, token_type, token.casefold())] += 1
                for word in sorted(context)[:40]:
                    self.postings[(category, token_type, word)].append(index)

    @staticmethod
    def _format_score(source: str, target: str) -> float:
        score = -0.08 * abs(len(source) - len(target))
        source_digits = sum(char.isdigit() for char in source)
        target_digits = sum(char.isdigit() for char in target)
        score -= 0.15 * abs(source_digits - target_digits)
        if source.isupper() == target.isupper():
            score += 0.2
        if source.islower() == target.islower():
            score += 0.2
        if semantic_type(source) == "size_or_quantity":
            try:
                left = float(source.replace(",", "."))
                right = float(target.replace(",", "."))
                if left > 0 and right > 0:
                    score -= abs(math.log(right / left))
            except ValueError:
                score -= 2.0
        return score

    def candidates(
        self,
        category: str,
        source: str,
        context: set[str],
        excluded: set[str],
        limit: int,
    ) -> list[dict[str, object]]:
        token_type = semantic_type(source)
        key = (category, token_type)
        entries = self.entries.get(key, [])
        candidate_indices: set[int] = set()
        for word in sorted(context, key=lambda value: (len(value), value), reverse=True)[:12]:
            candidate_indices.update(self.postings.get((category, token_type, word), [])[:1000])
        if len(candidate_indices) < limit * 4:
            candidate_indices.update(range(min(len(entries), 4000)))
        best: dict[str, tuple[float, BankEntry]] = {}
        excluded_folded = {value.casefold() for value in excluded}
        for index in candidate_indices:
            entry = entries[index]
            folded = entry.token.casefold()
            if folded in excluded_folded or folded == source.casefold():
                continue
            overlap = len(context & set(entry.context))
            frequency = self.frequencies[(category, token_type, folded)]
            score = 1.5 * overlap + math.log1p(frequency) + self._format_score(source, entry.token)
            previous = best.get(folded)
            if previous is None or score > previous[0]:
                best[folded] = (score, entry)
        ranked = sorted(best.values(), key=lambda value: (-value[0], value[1].token.casefold()))
        return [
            {
                "replacement_token": entry.token,
                "bank_frequency": self.frequencies[(category, token_type, entry.token.casefold())],
                "context_overlap": len(context & set(entry.context)),
            }
            for _, entry in ranked[:limit]
        ]


def build_operations(
    row: dict[str, object],
    items: pd.DataFrame,
    bank: ReplacementBank,
    source_token_limit: int,
    candidates_per_token: int,
) -> list[dict[str, object]]:
    left = items.loc[row["id1"]]
    right = items.loc[row["id2"]]
    context = context_tokens(f"{left['name']} {right['name']}")
    offered = [str(token) for token in row["offered_tokens"][:source_token_limit]]
    operations: list[dict[str, object]] = []
    for source in offered:
        for candidate in bank.candidates(
            str(row["category"]), source, context, set(offered), candidates_per_token
        ):
            operations.append({
                "operation_id": f"op{len(operations)}",
                "source_token": source,
                **candidate,
            })
    return operations


def decision_schema(slot: int, operation_ids: list[str], max_replacements: int) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot": {"type": "integer", "const": slot},
            "status": {"type": "string", "enum": ["replace", "skip"]},
            "operation_ids": {
                "type": "array",
                "maxItems": max_replacements,
                "items": {"type": "string", "enum": operation_ids},
            },
            "skip_reason": {"type": "string", "maxLength": 200},
        },
        "required": ["slot", "status", "operation_ids", "skip_reason"],
    }


def batch_tool(rows: list[dict[str, object]], max_replacements: int) -> dict[str, object]:
    alternatives = [
        decision_schema(index, [str(op["operation_id"]) for op in row["candidate_operations"]], max_replacements)
        for index, row in enumerate(rows)
    ]
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return one structured replacement decision for every batch slot.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": len(rows),
                        "maxItems": len(rows),
                        "items": {"oneOf": alternatives},
                    }
                },
                "required": ["decisions"],
            },
        },
    }


def validate_batch_decisions(
    value: object, rows: list[dict[str, object]], max_replacements: int
) -> list[dict[str, object]]:
    if not isinstance(value, dict) or set(value) != {"decisions"}:
        raise ValueError("batch_decision_fields_mismatch")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise ValueError("batch_decision_count_mismatch")
    output: list[dict[str, object] | None] = [None] * len(rows)
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
            "slot", "status", "operation_ids", "skip_reason"
        }:
            raise ValueError("invalid_batch_decision")
        slot = decision["slot"]
        if not isinstance(slot, int) or not 0 <= slot < len(rows) or output[slot] is not None:
            raise ValueError("invalid_or_duplicate_slot")
        status = decision["status"]
        operation_ids = decision["operation_ids"]
        skip_reason = decision["skip_reason"]
        if status not in {"replace", "skip"} or not isinstance(operation_ids, list) or not isinstance(skip_reason, str):
            raise ValueError("invalid_batch_types")
        allowed = {str(op["operation_id"]): op for op in rows[slot]["candidate_operations"]}
        if status == "skip":
            if operation_ids or not skip_reason.strip():
                raise ValueError("invalid_batch_skip")
            output[slot] = {"status": "skip", "replacements": [], "skip_reason": skip_reason}
            continue
        if not 1 <= len(operation_ids) <= max_replacements or len(set(operation_ids)) != len(operation_ids):
            raise ValueError("invalid_batch_replacement_count")
        if any(not isinstance(operation_id, str) or operation_id not in allowed for operation_id in operation_ids):
            raise ValueError("operation_not_allowed")
        replacements = [
            {
                "source_token": str(allowed[operation_id]["source_token"]),
                "replacement_token": str(allowed[operation_id]["replacement_token"]),
            }
            for operation_id in operation_ids
        ]
        if len({row["source_token"] for row in replacements}) != len(replacements):
            raise ValueError("duplicate_source_token")
        output[slot] = {"status": "replace", "replacements": replacements, "skip_reason": skip_reason}
    if any(decision is None for decision in output):
        raise ValueError("missing_slot")
    return [decision for decision in output if decision is not None]


def parse_batch_message(
    message: object, rows: list[dict[str, object]], max_replacements: int
) -> list[dict[str, object]]:
    calls = getattr(message, "tool_calls", None)
    if not calls or len(calls) != 1 or calls[0].function.name != TOOL_NAME:
        raise ValueError("expected_one_batch_tool_call")
    return validate_batch_decisions(
        json.loads(calls[0].function.arguments), rows, max_replacements
    )


def category_few_shots(few_shots: dict[str, list[dict[str, object]]], category: str) -> list[dict[str, object]]:
    return [
        {"input": row["input"], "decision": row["decision"]}
        for row in few_shots.get(category, [])[:3]
    ]


def provider_rows(batch: list[dict[str, object]], items: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for slot, row in enumerate(batch):
        output.append({
            "slot": slot,
            "category": str(row["category"]),
            "item_1": public_item(items.loc[row["id1"]]),
            "item_2": public_item(items.loc[row["id2"]]),
            "retrieved_human_train_examples": compact_rag(row),
            "candidate_operations": row["candidate_operations"],
        })
    return output


def call_rotating(
    client: Any,
    router: ModelQuotaRouter,
    *,
    offset: int,
    excluded: Iterable[str],
    request: Any,
) -> tuple[Any | None, str | None, dict[str, object] | None]:
    attempted = set(excluded)
    while True:
        model = router.choose_model(offset, exclude=attempted)
        if model is None:
            return None, None, {"classification": "no_available_model"}
        try:
            return request(model), model, None
        except Exception as exc:
            classification = router.register_error(model, exc)
            if classification in {"quota", "model_unavailable"}:
                attempted.add(model)
                continue
            return None, model, {
                "classification": classification,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }


def process_batch(
    batch: list[dict[str, object]],
    items: pd.DataFrame,
    few_shots: dict[str, list[dict[str, object]]],
    client: Any,
    router: ModelQuotaRouter,
    max_replacements: int,
    batch_index: int,
    model_pool_width: int,
) -> dict[str, object]:
    started = time.perf_counter()
    category = str(batch[0]["category"])
    public_rows = provider_rows(batch, items)
    examples = category_few_shots(few_shots, category)
    messages = [
        {"role": "system", "content": GENERATOR_PROMPT},
        {"role": "system", "content": json.dumps({"category_few_shots": examples}, ensure_ascii=False, sort_keys=True)},
        {"role": "user", "content": json.dumps({"rows": public_rows}, ensure_ascii=False, sort_keys=True)},
    ]

    def generator_request(model: str) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[batch_tool(public_rows, max_replacements)],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            temperature=0.2,
            max_tokens=max(1024, 192 * len(batch)),
            extra_body={"enable_thinking": False},
        )

    generation, generator_model, error = call_rotating(
        client,
        router,
        offset=batch_index % model_pool_width,
        excluded=set(),
        request=generator_request,
    )
    if error:
        return {
            "request_status": "terminal_error",
            "batch_sha256": stable_hash([row["source_pair_sha256"] for row in batch]),
            "source_pair_sha256s": [row["source_pair_sha256"] for row in batch],
            "category": category,
            "stage": "generator",
            "model": generator_model,
            "error": error,
            "elapsed_seconds": time.perf_counter() - started,
        }
    try:
        generated = parse_batch_message(generation.choices[0].message, public_rows, max_replacements)
    except Exception as exc:
        return {
            "request_status": "terminal_error",
            "batch_sha256": stable_hash([row["source_pair_sha256"] for row in batch]),
            "source_pair_sha256s": [row["source_pair_sha256"] for row in batch],
            "category": category,
            "stage": "generator_parse",
            "model": generator_model,
            "error": {"classification": "invalid_structured_output", "error": str(exc)},
            "elapsed_seconds": time.perf_counter() - started,
        }

    validation_rows: list[dict[str, object]] = []
    validation_slots: list[int] = []
    for slot, decision in enumerate(generated):
        if decision["status"] != "replace":
            continue
        allowed = {
            (str(op["source_token"]), str(op["replacement_token"])): op
            for op in public_rows[slot]["candidate_operations"]
        }
        chosen = [allowed[(op["source_token"], op["replacement_token"])] for op in decision["replacements"]]
        validation_rows.append({
            **{key: public_rows[slot][key] for key in ("category", "item_1", "item_2", "retrieved_human_train_examples")},
            "candidate_operations": chosen,
        })
        validation_slots.append(slot)

    validated = generated
    validator_model = None
    validator_usage: dict[str, object] = {}
    validator_request_id = None
    if validation_rows:
        validator_messages = [
            {"role": "system", "content": VALIDATOR_PROMPT},
            {"role": "user", "content": json.dumps({"rows": validation_rows}, ensure_ascii=False, sort_keys=True)},
        ]

        def validator_request(model: str) -> Any:
            return client.chat.completions.create(
                model=model,
                messages=validator_messages,
                tools=[batch_tool(validation_rows, max_replacements)],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                temperature=0.0,
                max_tokens=max(512, 192 * len(validation_rows)),
                extra_body={"enable_thinking": False},
            )

        validation, validator_model, error = call_rotating(
            client,
            router,
            offset=(batch_index + 1) % model_pool_width,
            excluded={str(generator_model)},
            request=validator_request,
        )
        if error:
            return {
                "request_status": "terminal_error",
                "batch_sha256": stable_hash([row["source_pair_sha256"] for row in batch]),
                "source_pair_sha256s": [row["source_pair_sha256"] for row in batch],
                "category": category,
                "stage": "validator",
                "generator_model": generator_model,
                "model": validator_model,
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
            }
        try:
            checked = parse_batch_message(validation.choices[0].message, validation_rows, max_replacements)
        except Exception as exc:
            return {
                "request_status": "terminal_error",
                "batch_sha256": stable_hash([row["source_pair_sha256"] for row in batch]),
                "source_pair_sha256s": [row["source_pair_sha256"] for row in batch],
                "category": category,
                "stage": "validator_parse",
                "generator_model": generator_model,
                "model": validator_model,
                "error": {"classification": "invalid_structured_output", "error": str(exc)},
                "elapsed_seconds": time.perf_counter() - started,
            }
        validated = [
            {"status": "skip", "replacements": [], "skip_reason": "generator_skip"}
            for _ in batch
        ]
        for slot, decision in enumerate(generated):
            if decision["status"] == "skip":
                validated[slot] = decision
        for local_slot, original_slot in enumerate(validation_slots):
            validated[original_slot] = checked[local_slot]
        validator_usage = validation.usage.model_dump() if validation.usage else {}
        validator_request_id = getattr(validation, "_request_id", None) or validation.id

    results: list[dict[str, object]] = []
    for row, generator_decision, final_decision in zip(batch, generated, validated):
        results.append({
            **row,
            "request_status": "complete",
            "generator_decision": generator_decision,
            "decision": final_decision,
        })
    return {
        "request_status": "complete",
        "batch_sha256": stable_hash([row["source_pair_sha256"] for row in batch]),
        "source_pair_sha256s": [row["source_pair_sha256"] for row in batch],
        "category": category,
        "generator_model": generator_model,
        "validator_model": validator_model,
        "generator_usage": generation.usage.model_dump() if generation.usage else {},
        "validator_usage": validator_usage,
        "generator_request_id": getattr(generation, "_request_id", None) or generation.id,
        "validator_request_id": validator_request_id,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def balanced_sources(
    source_rows: list[dict[str, object]],
    excluded: set[str],
    source_count: int,
) -> list[dict[str, object]]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in source_rows:
        if str(row["source_pair_sha256"]) not in excluded:
            by_category[str(row["category"])].append(row)
    categories = sorted(by_category)
    if source_count <= 0:
        return [row for category in categories for row in by_category[category]]
    base, remainder = divmod(source_count, len(categories))
    selected: list[dict[str, object]] = []
    for index, category in enumerate(categories):
        count = base + int(index < remainder)
        selected.extend(by_category[category][:count])
    return selected


def make_batches(rows: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    category_batches: dict[str, list[list[dict[str, object]]]] = {}
    for category, category_rows in by_category.items():
        category_batches[category] = [
            category_rows[index:index + batch_size]
            for index in range(0, len(category_rows), batch_size)
        ]
    batches: list[list[dict[str, object]]] = []
    while any(category_batches.values()):
        for category in sorted(category_batches):
            if category_batches[category]:
                batches.append(category_batches[category].pop(0))
    return batches


def flatten_complete(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        result
        for record in records
        if record.get("request_status") == "complete"
        for result in record.get("results", [])
    ]


def usage_sum(records: list[dict[str, object]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        for key in ("generator_usage", "validator_usage"):
            for name, value in record.get(key, {}).items():
                if isinstance(value, int):
                    totals[name] += value
                elif isinstance(value, dict):
                    for nested_name, nested_value in value.items():
                        if isinstance(nested_value, int):
                            totals[f"{name}.{nested_name}"] += nested_value
    return dict(sorted(totals.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--few-shots", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, action="append", required=True)
    parser.add_argument("--exclude-journal", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--source-token-limit", type=int, default=4)
    parser.add_argument("--candidates-per-token", type=int, default=6)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument("--model-pool-width", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.workers,
        args.source_token_limit,
        args.candidates_per_token,
        args.max_replacements,
        args.model_pool_width,
    ) <= 0:
        raise ValueError("limits_must_be_positive")
    if args.execute and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / "batch_journal.jsonl"
    existing = read_jsonl(journal)
    excluded_sources = {
        str(source_hash)
        for record in existing
        for source_hash in record.get("source_pair_sha256s", [])
    }
    for path in args.exclude_journal:
        for record in read_jsonl(path):
            if "source_pair_sha256s" in record:
                excluded_sources.update(str(value) for value in record["source_pair_sha256s"])
            elif "source_pair_sha256" in record:
                excluded_sources.add(str(record["source_pair_sha256"]))

    source_rows = read_jsonl(args.source_pack)
    items = pd.read_parquet(args.items, columns=["id", "category", "name", "attributes"]).set_index("id", drop=False)
    few_shots = load_few_shots(args.few_shots, args.max_replacements)
    bank_started = time.perf_counter()
    bank = ReplacementBank(source_rows, items)
    bank_seconds = time.perf_counter() - bank_started
    selected = balanced_sources(source_rows, excluded_sources, args.source_count)
    prepared: list[dict[str, object]] = []
    for row in selected:
        operations = build_operations(
            row, items, bank, args.source_token_limit, args.candidates_per_token
        )
        if operations:
            prepared.append({**row, "candidate_operations": operations})
    batches = make_batches(prepared, args.batch_size)
    provider_contract = {
        "prompt_version": PROMPT_VERSION,
        "candidate_bank_version": CANDIDATE_BANK_VERSION,
        "source_pack_sha256": sha256_file(args.source_pack),
        "few_shots_sha256": sha256_file(args.few_shots),
        "requested_sources": args.source_count,
        "prepared_sources": len(prepared),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "model_pool_width": args.model_pool_width,
        "generator_calls_per_batch": 1,
        "validator_calls_per_nonempty_batch": 1,
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
        "provider_fields": ["category", "names", "attributes", "rag_relations", "candidate_operations"],
    }
    (args.output_dir / "provider_contract.json").write_text(
        json.dumps(provider_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preview_rows = provider_rows(prepared[: min(2, len(prepared))], items)
    (args.output_dir / "provider_payload_preview.json").write_text(
        json.dumps({"rows": preview_rows}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_started = time.perf_counter()
    if args.execute and batches:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=args.base_url.rstrip("/"),
            max_retries=0,
            timeout=args.timeout,
        )
        router = ModelQuotaRouter(args.model_registry, args.output_dir / "model_events.jsonl")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for batch_index, batch in enumerate(batches):
                source_hashes = [row["source_pair_sha256"] for row in batch]
                append_jsonl(journal, {
                    "request_status": "submitted",
                    "batch_sha256": stable_hash(source_hashes),
                    "source_pair_sha256s": source_hashes,
                    "category": batch[0]["category"],
                })
                future = executor.submit(
                    process_batch,
                    batch,
                    items,
                    few_shots,
                    client,
                    router,
                    args.max_replacements,
                    batch_index,
                    args.model_pool_width,
                )
                futures[future] = batch_index
            completed_batches = 0
            for future in as_completed(futures):
                append_jsonl(journal, future.result())
                completed_batches += 1
                print(json.dumps({"completed_batches": completed_batches, "total_batches": len(batches)}), flush=True)

    elapsed = time.perf_counter() - run_started
    records = read_jsonl(journal)
    complete_rows = flatten_complete(records)
    synthetic_items, synthetic_pairs = materialize(complete_rows, items)
    outputs: dict[str, dict[str, object]] = {}
    for name, frame in (("synthetic_items.parquet", synthetic_items), ("synthetic_pairs.parquet", synthetic_pairs)):
        if not frame.empty:
            path = args.output_dir / name
            frame.to_parquet(path, index=False)
            outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    accepted_operations = sum(
        len(row["decision"]["replacements"])
        for row in complete_rows
        if row["decision"]["status"] == "replace"
    )
    processed_sources = len(complete_rows)
    terminal_sources = sum(
        len(record.get("source_pair_sha256s", []))
        for record in records
        if record.get("request_status") == "terminal_error"
    )
    attempted_sources = processed_sources + terminal_sources
    pair_yield = accepted_operations * 2
    sources_per_hour = attempted_sources / elapsed * 3600 if elapsed > 0 else 0.0
    operations_per_source = accepted_operations / attempted_sources if attempted_sources else 0.0
    remaining_operations = max(0, 6000 - accepted_operations)
    additional_sources = remaining_operations / operations_per_source if operations_per_source > 0 else None
    required_sources = attempted_sources + additional_sources if additional_sources is not None else None
    eta_hours = additional_sources / sources_per_hour if additional_sources is not None and sources_per_hour else None
    eligible_remaining = len(source_rows) - len(excluded_sources)
    complete_batches = [record for record in records if record.get("request_status") == "complete"]
    model_counts = {
        "generator": dict(sorted(Counter(str(record.get("generator_model")) for record in complete_batches).items())),
        "validator": dict(sorted(Counter(str(record.get("validator_model")) for record in complete_batches).items())),
    }
    decision_counts = Counter(str(row["decision"]["status"]) for row in complete_rows)
    operations_by_category: Counter[str] = Counter()
    for row in complete_rows:
        if row["decision"]["status"] == "replace":
            operations_by_category[str(row["category"])] += len(row["decision"]["replacements"])
    submitted_batches = {
        str(record["batch_sha256"])
        for record in records
        if record.get("request_status") == "submitted"
    }
    terminal_batch_hashes = {
        str(record["batch_sha256"])
        for record in records
        if record.get("request_status") in {"complete", "terminal_error"}
    }
    unresolved_batches = sorted(submitted_batches - terminal_batch_hashes)
    manifest = {
        "status": "complete" if not unresolved_batches else "stopped_with_uncertain_batches",
        "execute": args.execute,
        "prompt_version": PROMPT_VERSION,
        "candidate_bank_version": CANDIDATE_BANK_VERSION,
        "source_pack_sha256": sha256_file(args.source_pack),
        "few_shots_sha256": sha256_file(args.few_shots),
        "bank_build_seconds": bank_seconds,
        "api_elapsed_seconds": elapsed,
        "prepared_sources": len(prepared),
        "processed_sources": processed_sources,
        "terminal_sources": terminal_sources,
        "attempted_sources": attempted_sources,
        "terminal_batches": sum(record.get("request_status") == "terminal_error" for record in records),
        "accepted_operations": accepted_operations,
        "operations_by_category": dict(sorted(operations_by_category.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "synthetic_pairs": pair_yield,
        "positive_pairs": int(synthetic_pairs["target"].eq(1).sum()) if not synthetic_pairs.empty else 0,
        "negative_pairs": int(synthetic_pairs["target"].eq(0).sum()) if not synthetic_pairs.empty else 0,
        "operations_per_source": operations_per_source,
        "sources_per_hour": sources_per_hour,
        "estimated_sources_for_12000_pairs": required_sources,
        "estimated_hours_for_12000_pairs": eta_hours,
        "eligible_sources_remaining": eligible_remaining,
        "projected_max_total_pairs": pair_yield + eligible_remaining * operations_per_source * 2,
        "unresolved_batches": unresolved_batches,
        "model_counts": model_counts,
        "usage": usage_sum(records),
        "outputs": outputs,
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
    }
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
