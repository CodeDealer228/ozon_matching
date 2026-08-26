"""Quality-first multi-generator, RAG, web-research, validator pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from alibaba_model_quota_router import ModelQuotaRouter
from generate_token_replacement_counterfactuals import (
    DEFAULT_BASE_URL,
    TOOL_NAME,
    append_jsonl,
    build_messages,
    decision_tool,
    load_few_shots,
    materialize,
    parse_tool_decision,
    read_jsonl,
    sha256_file,
    validate_decision,
)
from token_replacement_web_search_tool import (
    build_research_payload,
    run_alibaba_web_research,
)


ROLE_PROMPTS = {
    "catalog": (
        "Проверь смысловой тип каждого токена и внутреннюю согласованность карточек. "
        "Предлагай замену только если она правдоподобна для этого типа товара."
    ),
    "hard_negative": (
        "Ищи трудный контрфактуальный пример: внешне очень похожий товар, который после "
        "односторонней замены действительно становится другим товаром."
    ),
    "marketplace": (
        "Предпочитай естественные значения и написание из реальных marketplace-карточек; "
        "избегай учебных, искусственных и слишком очевидных замен."
    ),
}
ROLE_TEMPERATURES = {"catalog": 0.1, "hard_negative": 0.35, "marketplace": 0.55}
WORD_RE = re.compile(r"[\w-]{3,}", flags=re.UNICODE)


def public_source(candidate: dict[str, object], items: pd.DataFrame) -> dict[str, object]:
    left, right = items.loc[candidate["id1"]], items.loc[candidate["id2"]]
    return {
        "category": str(candidate["category"]),
        "item_1": {"name": str(left["name"]), "attributes": json.loads(left["attributes"])},
        "item_2": {"name": str(right["name"]), "attributes": json.loads(right["attributes"])},
        "offered_tokens": candidate["offered_tokens"],
    }


def compact_rag_item(value: dict[str, object], max_attribute_chars: int = 900) -> dict[str, object]:
    attributes = value.get("attributes", {})
    compact: dict[str, object] = {}
    if isinstance(attributes, dict):
        for key, nested in attributes.items():
            candidate = {**compact, str(key): nested}
            if len(json.dumps(candidate, ensure_ascii=False)) > max_attribute_chars:
                break
            compact[str(key)] = nested
    return {"name": str(value.get("name", "")), "attributes": compact}


def strip_rag(row: dict[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for example in row["rag_examples"]:
        output.append({
            "relation": example["relation"],
            "category": example["category"],
            "item_1": compact_rag_item(example["item_1"]),
            "item_2": compact_rag_item(example["item_2"]),
        })
    return output


def generator_messages(
    candidate: dict[str, object],
    items: pd.DataFrame,
    few_shots: list[dict[str, object]],
    role: str,
) -> list[dict[str, str]]:
    left, right = items.loc[candidate["id1"]], items.loc[candidate["id2"]]
    messages = build_messages(left, right, candidate["offered_tokens"], few_shots)
    messages[0]["content"] += " " + ROLE_PROMPTS[role]
    payload = json.loads(messages[1]["content"])
    payload["retrieved_human_train_examples"] = strip_rag(candidate)
    payload["generator_role"] = role
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return messages


def operation_key(value: dict[str, object]) -> tuple[str, str]:
    return (
        str(value["source_token"]).casefold(),
        str(value["replacement_token"]).casefold(),
    )


def candidate_tool(operations: list[dict[str, str]], max_replacements: int) -> dict[str, object]:
    alternatives = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_token": {"type": "string", "const": operation["source_token"]},
                "replacement_token": {
                    "type": "string",
                    "const": operation["replacement_token"],
                },
            },
            "required": ["source_token", "replacement_token"],
        }
        for operation in operations
    ]
    tool = decision_tool(
        sorted({operation["source_token"] for operation in operations}), max_replacements
    )
    tool["function"]["parameters"]["properties"]["replacements"]["items"] = {
        "oneOf": alternatives
    }
    return tool


def call_chat_rotating(
    client: Any,
    router: ModelQuotaRouter,
    *,
    offset: int,
    excluded: set[str],
    request: Callable[[str], Any],
) -> tuple[Any | None, str | None, dict[str, object] | None]:
    attempted: set[str] = set(excluded)
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


def infer_product_type_hint(candidate: dict[str, object], items: pd.DataFrame) -> str:
    left = {word.casefold() for word in WORD_RE.findall(str(items.at[candidate["id1"], "name"]))}
    right = {word.casefold() for word in WORD_RE.findall(str(items.at[candidate["id2"], "name"]))}
    common = sorted(left & right, key=lambda value: (-len(value), value))[:5]
    return " ".join(common) if common else str(candidate["category"])


def semantic_type(token: str) -> str:
    has_letters = bool(re.search(r"[A-Za-zА-Яа-я]", token))
    has_digits = bool(re.search(r"\d", token))
    if has_letters and has_digits:
        return "model"
    if has_letters:
        return "brand"
    if has_digits:
        return "size_or_quantity"
    return "other"


def process_source(
    candidate: dict[str, object],
    items: pd.DataFrame,
    few_shots: dict[str, list[dict[str, object]]],
    client: Any,
    router: ModelQuotaRouter,
    web_router: ModelQuotaRouter,
    max_replacements: int,
    source_index: int,
) -> dict[str, object]:
    category_examples = [
        *few_shots.get("__global__", []),
        *few_shots.get(str(candidate["category"]), []),
    ]
    proposals: list[dict[str, object]] = []
    used_models: set[str] = set()
    for role_index, role in enumerate(ROLE_PROMPTS):
        def request(model: str, role: str = role) -> Any:
            return client.chat.completions.create(
                model=model,
                messages=generator_messages(candidate, items, category_examples, role),
                tools=[decision_tool(candidate["offered_tokens"], max_replacements)],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                temperature=ROLE_TEMPERATURES[role],
                max_tokens=256,
                extra_body={"enable_thinking": False},
            )

        completion, model, error = call_chat_rotating(
            client,
            router,
            offset=source_index + role_index,
            excluded=used_models,
            request=request,
        )
        if error:
            proposals.append({"role": role, "model": model, "error": error})
            continue
        assert completion is not None and model is not None
        used_models.add(model)
        try:
            decision = parse_tool_decision(
                completion.choices[0].message,
                candidate["offered_tokens"],
                max_replacements,
            )
        except Exception as exc:
            proposals.append({
                "role": role,
                "model": model,
                "error": {"classification": "invalid_structured_output", "error": str(exc)},
            })
            continue
        proposals.append({
            "role": role,
            "model": model,
            "decision": decision,
            "usage": completion.usage.model_dump() if completion.usage else {},
            "provider_request_id": getattr(completion, "_request_id", None) or completion.id,
        })

    valid = [proposal for proposal in proposals if isinstance(proposal.get("decision"), dict)]
    if not valid:
        return {**candidate, "request_status": "terminal_error", "proposals": proposals}
    if all(proposal["decision"]["status"] == "skip" for proposal in valid):
        return {
            **candidate,
            "request_status": "complete",
            "decision": {"status": "skip", "replacements": [], "skip_reason": "generator_consensus_skip"},
            "proposals": proposals,
            "web_briefs": [],
        }

    operation_values: dict[tuple[str, str], dict[str, str]] = {}
    votes: Counter[tuple[str, str]] = Counter()
    source_votes: Counter[str] = Counter()
    for proposal in valid:
        decision = proposal["decision"]
        if decision["status"] != "replace":
            continue
        for operation in decision["replacements"]:
            key = operation_key(operation)
            operation_values[key] = {
                "source_token": str(operation["source_token"]),
                "replacement_token": str(operation["replacement_token"]),
            }
            votes[key] += 1
            source_votes[str(operation["source_token"])] += 1
    shortlist = [operation_values[key] for key, count in votes.items() if count >= 2]
    candidate_values = {operation_key(operation): operation for operation in shortlist}

    web_briefs: list[dict[str, object]] = []
    web_sources = [
        source
        for source, _ in source_votes.most_common(2)
        if not shortlist or semantic_type(source) in {"brand", "model", "article", "compatibility_code"}
    ]
    for web_index, source in enumerate(web_sources):
        payload = build_research_payload(
            category=str(candidate["category"]),
            product_type_hint=infer_product_type_hint(candidate, items),
            source_token=source,
            semantic_type_hint=semantic_type(source),
            max_candidates=3,
        )

        def web_request(model: str) -> Any:
            return run_alibaba_web_research(client, model=model, payload=payload)

        result, web_model, error = call_chat_rotating(
            client,
            web_router,
            offset=source_index + web_index,
            excluded=set(),
            request=web_request,
        )
        if error:
            web_briefs.append({"source_token": source, "model": web_model, "error": error})
            continue
        brief, response = result
        web_briefs.append({
            "source_token": source,
            "model": web_model,
            "brief": brief,
            "provider_request_id": getattr(response, "_request_id", None) or response.id,
            "usage": response.usage.model_dump() if response.usage else {},
        })
        if brief["status"] == "found":
            for web_candidate in brief["candidates"]:
                operation = {
                    "source_token": source,
                    "replacement_token": str(web_candidate["replacement_token"]),
                }
                candidate_values[operation_key(operation)] = operation

    candidates = list(candidate_values.values())
    if not candidates:
        return {
            **candidate,
            "request_status": "complete",
            "decision": {"status": "skip", "replacements": [], "skip_reason": "no_candidate_operation"},
            "proposals": proposals,
            "web_briefs": web_briefs,
        }
    validator_payload = {
        **public_source(candidate, items),
        "retrieved_human_train_examples": strip_rag(candidate),
        "generator_proposals": proposals,
        "web_research": web_briefs,
        "candidate_operations": candidates,
    }
    validator_messages = [
        {
            "role": "system",
            "content": (
                "Ты финальный валидатор контрфактуальной операции. Выбери до двух только "
                "из candidate_operations, если они сохраняют внутреннюю согласованность обеих "
                "карточек и дают естественный hard negative при односторонней замене. RAG train "
                "examples показывают реальную границу same/different. При сомнении верни skip."
            ),
        },
        {"role": "user", "content": json.dumps(validator_payload, ensure_ascii=False, sort_keys=True)},
    ]

    def validator_request(model: str) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=validator_messages,
            tools=[candidate_tool(candidates, max_replacements)],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            temperature=0.0,
            max_tokens=256,
            extra_body={"enable_thinking": False},
        )

    validation, validator_model, error = call_chat_rotating(
        client,
        router,
        offset=0,
        excluded=set(),
        request=validator_request,
    )
    if error:
        return {
            **candidate,
            "request_status": "terminal_error",
            "proposals": proposals,
            "web_briefs": web_briefs,
            "validator_error": error,
        }
    try:
        decision = parse_tool_decision(
            validation.choices[0].message,
            sorted({operation["source_token"] for operation in candidates}),
            max_replacements,
        )
        decision = validate_decision(decision, candidate["offered_tokens"], max_replacements)
        allowed = {operation_key(operation) for operation in candidates}
        if decision["status"] == "replace" and any(
            operation_key(operation) not in allowed for operation in decision["replacements"]
        ):
            raise ValueError("validator_operation_not_shortlisted")
    except Exception as exc:
        return {
            **candidate,
            "request_status": "terminal_error",
            "proposals": proposals,
            "web_briefs": web_briefs,
            "validator_error": {"classification": "invalid_structured_output", "error": str(exc)},
        }
    return {
        **candidate,
        "request_status": "complete",
        "decision": decision,
        "proposals": proposals,
        "web_briefs": web_briefs,
        "validator_model": validator_model,
        "validator_usage": validation.usage.model_dump() if validation.usage else {},
        "validator_request_id": getattr(validation, "_request_id", None) or validation.id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--few-shots", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, action="append", required=True)
    parser.add_argument("--web-model-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-operations-per-category", type=int, default=300)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(args.source_pack)
    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id", drop=False)
    few_shots = load_few_shots(args.few_shots, args.max_replacements)
    journal = args.output_dir / "request_journal.jsonl"
    records = read_jsonl(journal)
    completed = {str(record["source_pair_sha256"]) for record in records}
    operations_by_category: Counter[str] = Counter()
    for record in records:
        if record.get("request_status") == "complete" and record.get("decision", {}).get("status") == "replace":
            operations_by_category[str(record["category"])] += len(record["decision"]["replacements"])

    queues: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    for row in source_rows:
        if str(row["source_pair_sha256"]) not in completed:
            queues[str(row["category"])].append(row)
    preview = args.output_dir / "provider_contract.json"
    preview.write_text(json.dumps({
        "source_rows": len(source_rows),
        "target_operations_per_category": args.target_operations_per_category,
        "generator_roles": list(ROLE_PROMPTS),
        "rag_examples_per_source": 5,
        "ids_sent": False,
        "validation_sent": False,
        "web_receives_full_cards": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.execute:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=args.base_url.rstrip("/"),
            max_retries=0,
            timeout=args.timeout,
        )
        router = ModelQuotaRouter(args.model_registry, args.output_dir / "model_events.jsonl")
        web_router = ModelQuotaRouter(
            [args.web_model_registry], args.output_dir / "web_model_events.jsonl"
        )
        source_index = len(completed)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while True:
                active = [
                    category
                    for category in sorted(queues)
                    if queues[category]
                    and operations_by_category[category] < args.target_operations_per_category
                ]
                if not active or not router.available_models:
                    break
                batch_categories = active[: args.workers]
                futures = {}
                for category in batch_categories:
                    candidate = queues[category].popleft()
                    append_jsonl(journal, {
                        **{key: candidate[key] for key in ("id1", "id2", "source_pair_sha256", "category", "offered_tokens")},
                        "request_status": "submitted",
                    })
                    future = executor.submit(
                        process_source,
                        candidate,
                        items,
                        few_shots,
                        client,
                        router,
                        web_router,
                        args.max_replacements,
                        source_index,
                    )
                    source_index += 1
                    futures[future] = category
                for future in as_completed(futures):
                    category = futures[future]
                    record = future.result()
                    if record.get("request_status") == "complete" and record.get("decision", {}).get("status") == "replace":
                        remaining = args.target_operations_per_category - operations_by_category[category]
                        record["decision"]["replacements"] = record["decision"]["replacements"][:remaining]
                        operations_by_category[category] += len(record["decision"]["replacements"])
                    append_jsonl(journal, record)
                total = sum(operations_by_category.values())
                if total and total % 50 < len(futures) * args.max_replacements:
                    print(json.dumps({
                        "accepted_operations": total,
                        "synthetic_pairs": total * 2,
                        "categories_at_target": sum(
                            count >= args.target_operations_per_category
                            for count in operations_by_category.values()
                        ),
                    }, sort_keys=True), flush=True)

    records = read_jsonl(journal)
    synthetic_items, synthetic_pairs = materialize(records, items)
    outputs: dict[str, dict[str, object]] = {}
    if not synthetic_pairs.empty:
        items_path = args.output_dir / "synthetic_items.parquet"
        pairs_path = args.output_dir / "synthetic_pairs.parquet"
        synthetic_items.to_parquet(items_path, index=False)
        synthetic_pairs.to_parquet(pairs_path, index=False)
        for path in (items_path, pairs_path):
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    final_operations = Counter()
    for record in records:
        if record.get("request_status") == "complete" and record.get("decision", {}).get("status") == "replace":
            final_operations[str(record["category"])] += len(record["decision"]["replacements"])
    target_reached = bool(final_operations) and all(
        final_operations[category] >= args.target_operations_per_category
        for category in {str(row["category"]) for row in source_rows}
    )
    manifest = {
        "status": "target_complete" if target_reached else "dry_run_complete" if not args.execute else "stopped",
        "execute": args.execute,
        "source_pack_sha256": sha256_file(args.source_pack),
        "few_shots_sha256": sha256_file(args.few_shots),
        "target_operations_per_category": args.target_operations_per_category,
        "accepted_operations": sum(final_operations.values()),
        "operations_by_category": dict(sorted(final_operations.items())),
        "synthetic_pairs": len(synthetic_pairs),
        "positive_pairs": int(synthetic_pairs["target"].eq(1).sum()) if not synthetic_pairs.empty else 0,
        "negative_pairs": int(synthetic_pairs["target"].eq(0).sum()) if not synthetic_pairs.empty else 0,
        "ids_sent": False,
        "validation_sent": False,
        "outputs": outputs,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
