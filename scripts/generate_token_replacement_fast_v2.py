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
    mutate_item,
    read_jsonl,
    replace_text,
    sha256_file,
)
from generate_token_replacement_aggressive_rag import compact_rag_item, semantic_type


TOOL_NAME = "submit_batched_replacement_decisions"
CONTEXT_RE = re.compile(r"[A-Za-zА-Яа-я0-9-]{3,}")
VALUE_RE = re.compile(r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9./_-]{0,63}")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
PROMPT_VERSION = 2
CANDIDATE_BANK_VERSION = 4
CONTEXT_STOPWORDS = {
    "для", "или", "при", "без", "под", "над", "товар", "товара", "набор",
    "цвет", "размер", "штук", "шт", "арт", "модель", "бренд", "комплект",
    "the", "and", "with", "from", "black", "white",
}
ROLE_KEYWORDS = {
    "brand": ("бренд", "brand", "производитель", "торговая марка"),
    "article": (
        "артик", "партномер", "part number", "модель", "model", "oem", "oe-код",
        "код товара", "код продукт", "серия", "номер производ",
    ),
    "volume": ("объем", "объём", "емкость", "ёмкость", "литраж"),
    "weight": ("вес", "масса", "фасов"),
    "count": ("количество", "число штук", "деталей", "предметов"),
    "length": ("длина",),
    "width": ("ширина",),
    "height": ("высота",),
    "diameter": ("диаметр",),
    "power": ("мощност",),
    "resolution": ("разрешение", "dpi"),
    "format": ("формат",),
    "size": ("размер", "габарит"),
}
NUMERIC_ROLES = {
    "volume", "weight", "count", "length", "width", "height", "diameter",
    "power", "resolution", "size",
}
UNIT_SUFFIX_ROLES = {
    "dpi": "resolution",
    "см": "length",
    "мм": "length",
    "мл": "volume",
    "л": "volume",
    "г": "weight",
    "кг": "weight",
    "шт": "count",
    "штук": "count",
    "вт": "power",
}
UNIT_TOKEN_RE = re.compile(r"^(\d+(?:[.,]\d+)?)(dpi|см|мм|мл|л|г|кг|шт|штук|вт)$", re.IGNORECASE)
BAD_BRAND_VALUES = {
    "нет", "бренда", "без бренда", "noname", "no name", "китай", "россия",
    "германия", "италия", "франция", "сша", "япония", "корея", "таиланд",
}

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
    return {
        token.casefold()
        for token in CONTEXT_RE.findall(value)
        if token.casefold() not in CONTEXT_STOPWORDS and not token.isdigit()
    }


def attribute_leaves(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[str, object]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from attribute_leaves(nested, (*path, str(key)))
    elif isinstance(value, list):
        for nested in value:
            yield from attribute_leaves(nested, path)
    else:
        yield " ".join(path).casefold(), value


def key_role(key: str) -> str | None:
    if "страна" in key:
        return None
    for role, fragments in ROLE_KEYWORDS.items():
        if any(fragment in key for fragment in fragments):
            return role
    return None


def token_roles(attributes: object, token: str) -> set[str]:
    roles: set[str] = set()
    for key, value in attribute_leaves(attributes):
        role = key_role(key)
        if role is None:
            continue
        value_text = str(value).strip()
        if role == "brand":
            if value_text.casefold() == token.casefold():
                roles.add(role)
        elif replace_text(value_text, token, "__probe__")[1] > 0:
            roles.add(role)
    return roles


def infer_source_role(left: pd.Series, right: pd.Series, token: str) -> str | None:
    left_attributes = json.loads(str(left["attributes"]))
    right_attributes = json.loads(str(right["attributes"]))
    left_roles = token_roles(left_attributes, token)
    right_roles = token_roles(right_attributes, token)
    left_name_count = replace_text(str(left["name"]), token, "__probe__")[1]
    right_name_count = replace_text(str(right["name"]), token, "__probe__")[1]
    unit_match = UNIT_TOKEN_RE.fullmatch(token)
    if unit_match and left_name_count and right_name_count:
        return UNIT_SUFFIX_ROLES[unit_match.group(2).casefold()]
    common = left_roles & right_roles
    for role in (
        "article", "brand", "volume", "weight", "count", "length", "width",
        "height", "diameter", "power", "resolution", "format", "size",
    ):
        if role in common:
            return role
    for role in (
        "article", "brand", "volume", "weight", "count", "length", "width",
        "height", "diameter", "power", "resolution", "format", "size",
    ):
        if role in left_roles | right_roles and left_name_count and right_name_count:
            return role
    if semantic_type(token) == "model" and left_name_count and right_name_count:
        return "article"
    if semantic_type(token) == "model" and left_name_count and right_name_count:
        return "article"
    return None


def role_values(item: dict[str, object], role: str, source_token: str) -> list[str]:
    values: list[str] = []
    token_type = semantic_type(source_token)
    source_unit_match = UNIT_TOKEN_RE.fullmatch(source_token)
    for key, value in attribute_leaves(item.get("attributes", {})):
        if key_role(key) != role:
            continue
        value_text = str(value).strip()
        if role == "brand":
            candidates = [value_text] if len(VALUE_RE.findall(value_text)) == 1 else []
        else:
            pattern = NUMBER_RE if role in NUMERIC_ROLES else VALUE_RE
            candidates = pattern.findall(value_text)
        for candidate in candidates:
            candidate = str(candidate).strip(".,;:!?()[]{}\"'")
            if source_unit_match and candidate and UNIT_TOKEN_RE.fullmatch(candidate) is None:
                candidate = candidate + source_unit_match.group(2)
            if (
                candidate
                and candidate.casefold() not in BAD_BRAND_VALUES
                and semantic_type(candidate) == token_type
            ):
                values.append(candidate)
    return values


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
    role: str
    context: frozenset[str]
    source_pair_sha256: str
    evidence_name: str


class ReplacementBank:
    def __init__(self, source_rows: list[dict[str, object]], items: pd.DataFrame):
        self.entries: dict[tuple[str, str, str], list[BankEntry]] = defaultdict(list)
        self.postings: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        self.frequencies: Counter[tuple[str, str, str, str]] = Counter()
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
                role = infer_source_role(left, right, token)
                if role is None:
                    continue
                token_type = semantic_type(token)
                key = (category, role, token_type)
                index = len(self.entries[key])
                self.entries[key].append(
                    BankEntry(
                        token,
                        token_type,
                        role,
                        context,
                        str(row["source_pair_sha256"]),
                        str(left["name"]),
                    )
                )
                self.frequencies[(category, role, token_type, token.casefold())] += 1
                for word in sorted(context)[:40]:
                    self.postings[(category, role, token_type, word)].append(index)

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
        source_role: str,
        context: set[str],
        excluded: set[str],
        limit: int,
    ) -> list[dict[str, object]]:
        token_type = semantic_type(source)
        key = (category, source_role, token_type)
        entries = self.entries.get(key, [])
        candidate_indices: set[int] = set()
        for word in sorted(context, key=lambda value: (len(value), value), reverse=True)[:12]:
            candidate_indices.update(
                self.postings.get((category, source_role, token_type, word), [])[:1000]
            )
        best: dict[str, tuple[float, BankEntry]] = {}
        excluded_folded = {value.casefold() for value in excluded}
        for index in candidate_indices:
            entry = entries[index]
            folded = entry.token.casefold()
            if folded in excluded_folded or folded == source.casefold():
                continue
            overlap = len(context & set(entry.context))
            if overlap == 0:
                continue
            frequency = self.frequencies[(category, source_role, token_type, folded)]
            score = 1.5 * overlap + math.log1p(frequency) + self._format_score(source, entry.token)
            previous = best.get(folded)
            if previous is None or score > previous[0]:
                best[folded] = (score, entry)
        ranked = sorted(best.values(), key=lambda value: (-value[0], value[1].token.casefold()))
        return [
            {
                "replacement_token": entry.token,
                "source_role": source_role,
                "candidate_source": "contextual_human_train_bank",
                "evidence_name": entry.evidence_name[:240],
                "bank_frequency": self.frequencies[
                    (category, source_role, token_type, entry.token.casefold())
                ],
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
        source_role = infer_source_role(left, right, source)
        if source_role is None:
            continue
        left_name_changes = replace_text(str(left["name"]), source, "__probe__")[1]
        right_name_changes = replace_text(str(right["name"]), source, "__probe__")[1]
        if semantic_type(source) in {"model", "size_or_quantity"} and not (
            left_name_changes and right_name_changes
        ):
            continue
        try:
            mutate_item(left, source, "__probe__", -1)
            mutate_item(right, source, "__probe__", -2)
        except ValueError:
            continue
        candidates: list[dict[str, object]] = []
        seen_targets = {value.casefold() for value in offered}
        for example in row["rag_examples"]:
            if example["relation"] != "different_product":
                continue
            for item_key in ("item_1", "item_2"):
                evidence_item = example[item_key]
                for target in role_values(evidence_item, source_role, source):
                    folded = target.casefold()
                    if folded in seen_targets:
                        continue
                    seen_targets.add(folded)
                    candidates.append({
                        "replacement_token": target,
                        "source_role": source_role,
                        "candidate_source": "retrieved_hard_negative",
                        "evidence_name": str(evidence_item.get("name", ""))[:240],
                        "bank_frequency": 1,
                        "context_overlap": len(context & context_tokens(str(evidence_item.get("name", "")))),
                    })
        for candidate in bank.candidates(
            str(row["category"]),
            source,
            source_role,
            context,
            set(offered) | seen_targets,
            candidates_per_token * 2,
        ):
            folded = str(candidate["replacement_token"]).casefold()
            if folded in seen_targets:
                continue
            seen_targets.add(folded)
            candidates.append(candidate)
        for candidate in candidates[:candidates_per_token]:
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
        if not isinstance(decision, dict) or not {"slot", "status", "operation_ids"}.issubset(decision):
            raise ValueError("invalid_batch_decision")
        slot = decision["slot"]
        if isinstance(slot, str) and slot.isdigit():
            slot = int(slot)
        if not isinstance(slot, int) or not 0 <= slot < len(rows) or output[slot] is not None:
            raise ValueError("invalid_or_duplicate_slot")
        status = decision["status"]
        operation_ids = decision["operation_ids"]
        skip_reason = str(decision.get("skip_reason", ""))
        if status not in {"replace", "skip"} or not isinstance(operation_ids, list) or not isinstance(skip_reason, str):
            raise ValueError("invalid_batch_types")
        allowed = {str(op["operation_id"]): op for op in rows[slot]["candidate_operations"]}
        if status == "skip":
            if operation_ids:
                raise ValueError("invalid_batch_skip")
            output[slot] = {
                "status": "skip",
                "replacements": [],
                "skip_reason": skip_reason.strip() or "model_skip",
            }
            continue
        if not 1 <= len(operation_ids) <= max_replacements or len(set(operation_ids)) != len(operation_ids):
            raise ValueError("invalid_batch_replacement_count")
        if any(not isinstance(operation_id, str) or operation_id not in allowed for operation_id in operation_ids):
            raise ValueError("operation_not_allowed")
        replacements: list[dict[str, str]] = []
        seen_sources: set[str] = set()
        for operation_id in operation_ids:
            operation = allowed[operation_id]
            source_token = str(operation["source_token"])
            if source_token.casefold() in seen_sources:
                continue
            seen_sources.add(source_token.casefold())
            replacements.append({
                "source_token": source_token,
                "replacement_token": str(operation["replacement_token"]),
            })
        if not replacements:
            output[slot] = {
                "status": "skip",
                "replacements": [],
                "skip_reason": "duplicate_source_candidates",
            }
            continue
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


def filter_locally_applicable(
    rows: list[dict[str, object]], items: pd.DataFrame
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    filtered: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for original in rows:
        row = {**original, "decision": {**original["decision"]}}
        decision = row["decision"]
        if decision["status"] != "replace":
            filtered.append(row)
            continue
        kept: list[dict[str, str]] = []
        for operation in decision["replacements"]:
            source = str(operation["source_token"])
            target = str(operation["replacement_token"])
            try:
                mutate_item(items.loc[row["id1"]], source, target, -1)
                mutate_item(items.loc[row["id2"]], source, target, -2)
            except ValueError as exc:
                rejected.append({
                    "source_pair_sha256": row["source_pair_sha256"],
                    "category": row["category"],
                    "source_token": source,
                    "replacement_token": target,
                    "reason": str(exc),
                })
                continue
            kept.append({"source_token": source, "replacement_token": target})
        if kept:
            decision["replacements"] = kept
        else:
            row["decision"] = {
                "status": "skip",
                "replacements": [],
                "skip_reason": "local_replacement_not_applicable",
            }
        filtered.append(row)
    return filtered, rejected


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
    parser.add_argument("--finalize-only", action="store_true")
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
    bank_seconds = 0.0
    prepared: list[dict[str, object]] = []
    batches: list[list[dict[str, object]]] = []
    if not args.finalize_only:
        bank_started = time.perf_counter()
        bank = ReplacementBank(source_rows, items)
        bank_seconds = time.perf_counter() - bank_started
        selection_count = args.source_count * 4 if args.source_count > 0 else 0
        selected = balanced_sources(source_rows, excluded_sources, selection_count)
        for row in selected:
            operations = build_operations(
                row, items, bank, args.source_token_limit, args.candidates_per_token
            )
            if operations:
                prepared.append({**row, "candidate_operations": operations})
        if args.source_count > 0:
            prepared = balanced_sources(prepared, set(), args.source_count)
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
    if not args.finalize_only:
        (args.output_dir / "provider_contract.json").write_text(
            json.dumps(provider_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preview_rows = provider_rows(prepared[: min(2, len(prepared))], items)
        (args.output_dir / "provider_payload_preview.json").write_text(
            json.dumps({"rows": preview_rows}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        seen_preview_categories: set[str] = set()
        audit_rows: list[dict[str, object]] = []
        for row in prepared:
            category = str(row["category"])
            if category in seen_preview_categories:
                continue
            seen_preview_categories.add(category)
            audit_rows.append({
                "category": category,
                "source_pair_sha256": row["source_pair_sha256"],
                "name_1": str(items.at[row["id1"], "name"]),
                "name_2": str(items.at[row["id2"], "name"]),
                "candidate_operations": row["candidate_operations"],
            })
        (args.output_dir / "candidate_bank_audit_preview.json").write_text(
            json.dumps(audit_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    if args.finalize_only and journal.exists():
        elapsed = max(journal.stat().st_mtime - journal.stat().st_ctime, 1e-9)
    records = read_jsonl(journal)
    complete_rows, local_rejections = filter_locally_applicable(
        flatten_complete(records), items
    )
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
        "local_rejected_operations": len(local_rejections),
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
    (args.output_dir / "local_rejections.json").write_text(
        json.dumps(local_rejections, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
