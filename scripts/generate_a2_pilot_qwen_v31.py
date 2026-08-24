"""Run the disjoint, two-candidate A2 v3.1 pilot using the safe v3 engine."""

from __future__ import annotations

import json
import sys

import generate_a2_pilot_qwen_v3 as engine


PROFILE = "a2_qwen_contract_v31_two_candidate"
EXPECTED_SOURCE_SHA256 = "4cd7635b24933a4ab9989517177c5102ce1a3e3fbb4ec020a2c81d63a1c5cb6d"
EXPECTED_PLAN_SHA256 = "4f7bccefec1f7090cf2884e7061101db974d6e581c705922ddce4739c45dd348"
EXPECTED_FEW_SHOT_SHA256 = "1f6ed4386bdbb9b0b3fd789036993778f7d21fdeeb372abc3334b8634e705057"
EXPECTED_PILOT_ROWS = 32
VARIANT_KINDS = {"minimal", "alternate_placement"}


def build_messages(
    pack: dict[str, object], contract: dict[str, object], explicit_cache: bool
) -> list[dict[str, object]]:
    system = (
        "Ты выполняешь A2 hardening для product matching. A и B — разные товары из "
        "проверенной human-negative пары. Перепиши только name2, сохраняя identity B. "
        "Используй исключительно ALLOWED_OPERATIONS; placeholders вида <..._A_ONLY> "
        "запрещено копировать или раскрывать. Все protected numbers/identifiers должны "
        "остаться. Не добавляй никаких иных фактов. Если безопасного содержательного "
        "изменения нет, верни status=skip. Для status=ok верни РОВНО ДВА различных "
        "кандидата с одинаковым набором подтвержденных фактов: variant_kind=minimal "
        "делает минимальную вставку без перефразирования B; "
        "variant_kind=alternate_placement вставляет те же факты в другой естественной "
        "позиции. Каждый кандидат обязан перечислить использованные fact_id и "
        "operation_id. Ответь одним JSON-объектом без markdown. Схема: "
        "{status:'ok',candidates:[{variant_kind:'minimal',name2_prime:'...',"
        "used_fact_ids:[...],applied_operation_ids:[...]},{variant_kind:"
        "'alternate_placement',name2_prime:'...',used_fact_ids:[...],"
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
        "content": json.dumps(
            engine.compact_contract(contract), ensure_ascii=False, sort_keys=True
        ),
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
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("expected_exactly_two_candidates")
    if {candidate.get("variant_kind") for candidate in candidates} != VARIANT_KINDS:
        raise ValueError("expected_both_variant_kinds")
    names = []
    fact_sets = []
    operation_sets = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name2_prime"), str):
            raise ValueError("invalid_candidate")
        if not isinstance(candidate.get("used_fact_ids"), list):
            raise ValueError("missing_used_fact_ids")
        if not isinstance(candidate.get("applied_operation_ids"), list):
            raise ValueError("missing_applied_operation_ids")
        names.append(candidate["name2_prime"].strip().casefold())
        fact_sets.append(tuple(sorted(map(str, candidate["used_fact_ids"]))))
        operation_sets.append(tuple(sorted(map(str, candidate["applied_operation_ids"]))))
    if len(set(names)) != 2:
        raise ValueError("candidates_not_distinct")
    if fact_sets[0] != fact_sets[1] or operation_sets[0] != operation_sets[1]:
        raise ValueError("candidate_fact_sets_differ")
    return value


def add_default_argument(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend((flag, value))


def main() -> None:
    repo = engine.REPO_ROOT
    add_default_argument(
        "--source-pool",
        str(repo / "artifacts/synthetic_v2/a2_v31_pilot_plan/pilot_source_pool.parquet"),
    )
    add_default_argument(
        "--plan",
        str(repo / "artifacts/synthetic_v2/a2_v31_pilot_plan/pilot_generation_plan.jsonl"),
    )
    add_default_argument("--few-shot", str(repo / "synthetic_research/a2_few_shot_v31.json"))
    add_default_argument(
        "--output-dir", str(repo / "artifacts/synthetic_v2/a2_pilot_v31")
    )

    engine.PROFILE = PROFILE
    engine.EXPECTED_SOURCE_SHA256 = EXPECTED_SOURCE_SHA256
    engine.EXPECTED_PLAN_SHA256 = EXPECTED_PLAN_SHA256
    engine.EXPECTED_FEW_SHOT_SHA256 = EXPECTED_FEW_SHOT_SHA256
    engine.EXPECTED_PILOT_ROWS = EXPECTED_PILOT_ROWS
    engine.CANDIDATE_COUNT_LABEL = "exactly_2_same_facts_different_placement"
    engine.build_messages = build_messages
    engine.parse_response = parse_response
    engine.main()


if __name__ == "__main__":
    main()
