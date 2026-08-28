"""Require two independent structured validators for grounded bundles."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from alibaba_model_quota_router import ModelQuotaRouter
from generate_token_replacement_aggressive_rag import call_chat_rotating
from generate_token_replacement_counterfactuals import (
    DEFAULT_BASE_URL,
    append_jsonl,
    read_jsonl,
    sha256_file,
)
from judge_token_replacement_grounded_bundles import stable_hash


TOOL_NAME = "validate_grounded_bundles"
VALIDATOR_VERSION = 1
PROMPTS = (
    (
        "Ты строгий аудитор синтетических marketplace-карточек. Для каждого slot выбери "
        "keep только если пакет явно согласован. Reject/skip при смене типа товара, article "
        "другого бренда, остатке старого бренда/линейки/SKU, неподтверждённой фирменной "
        "формуле, противоречии target-card, искусственном повторе слов или превращении "
        "обычного описания в модель. Не предполагай, что бренд выпускает невидимую в "
        "evidence линейку. При сомнении skip. Ответ только tool call."
    ),
    (
        "Ты независимый red-team валидатор counterfactual product matching. Проверь, что "
        "после применения bundle две новые карточки всё ещё описывают один правдоподобный "
        "товар, а односторонняя замена действительно меняет идентичность. Полная реальная "
        "target-card должна подтверждать изменённый бренд/model/article; оставшаяся source "
        "линейка или формула обязана быть совместима. Не оценивай стиль мягко: keep только "
        "при высокой уверенности, иначе skip. Никаких собственных исправлений."
    ),
)


def decision_schema(slot: int) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot": {"type": "integer", "const": slot},
            "status": {"type": "string", "enum": ["keep", "skip"]},
            "reason": {"type": "string", "maxLength": 220},
        },
        "required": ["slot", "status", "reason"],
    }


def tool(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return one strict keep/skip decision for every slot.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": len(rows),
                        "maxItems": len(rows),
                        "items": {
                            "oneOf": [decision_schema(slot) for slot in range(len(rows))]
                        },
                    }
                },
                "required": ["decisions"],
            },
        },
    }


def messages(rows: list[dict[str, object]], prompt: str) -> list[dict[str, str]]:
    payload = []
    for slot, row in enumerate(rows):
        payload.append({
            "slot": slot,
            "category": row["category"],
            "source_role": row["source_role"],
            "source_item_1": row["item_1"],
            "source_item_2": row["item_2"],
            "mappings": row["bundle"]["mappings"],
            "new_name_1": row["bundle"]["preview_name_1"],
            "new_name_2": row["bundle"]["preview_name_2"],
            "grounded_target_card": row["bundle"]["target_card"],
        })
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"rows": payload}, ensure_ascii=False)},
    ]


def parse(message: object, rows: list[dict[str, object]]) -> list[dict[str, str]]:
    calls = getattr(message, "tool_calls", None)
    if not calls or len(calls) != 1 or calls[0].function.name != TOOL_NAME:
        raise ValueError("expected_one_validator_tool_call")
    value = json.loads(calls[0].function.arguments)
    decisions = value.get("decisions") if isinstance(value, dict) else None
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise ValueError("validator_decision_count_mismatch")
    output: list[dict[str, str] | None] = [None] * len(rows)
    for decision in decisions:
        slot = decision.get("slot") if isinstance(decision, dict) else None
        if isinstance(slot, str) and slot.isdigit():
            slot = int(slot)
        if not isinstance(slot, int) or not 0 <= slot < len(rows) or output[slot] is not None:
            raise ValueError("invalid_validator_slot")
        status = decision.get("status")
        if status not in {"keep", "skip"}:
            raise ValueError("invalid_validator_status")
        output[slot] = {
            "status": str(status),
            "reason": str(decision.get("reason", ""))[:220],
        }
    if any(value is None for value in output):
        raise ValueError("missing_validator_slot")
    return [value for value in output if value is not None]


def one_vote(
    batch: list[dict[str, object]],
    prompt_index: int,
    client: Any,
    router: ModelQuotaRouter,
    offset: int,
) -> dict[str, object]:
    def request(model: str) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages(batch, PROMPTS[prompt_index]),
            tools=[tool(batch)],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            temperature=0.0,
            max_tokens=800,
            extra_body={"enable_thinking": False},
        )

    completion, model, error = call_chat_rotating(
        client, router, offset=offset, excluded=set(), request=request
    )
    if error:
        return {"status": "terminal_error", "model": model, "error": error}
    assert completion is not None
    try:
        decisions = parse(completion.choices[0].message, batch)
    except Exception as exc:
        return {
            "status": "terminal_error",
            "model": model,
            "error": {"classification": "invalid_structured_output", "error": str(exc)},
        }
    return {
        "status": "complete",
        "model": model,
        "usage": completion.usage.model_dump() if completion.usage else {},
        "decisions": decisions,
    }


def validate_batch(
    batch: list[dict[str, object]],
    client: Any,
    router: ModelQuotaRouter,
    offset: int,
) -> dict[str, object]:
    started = time.perf_counter()
    votes = [
        one_vote(batch, prompt_index, client, router, offset * 2 + prompt_index)
        for prompt_index in range(2)
    ]
    results = []
    if all(vote["status"] == "complete" for vote in votes):
        for slot, row in enumerate(batch):
            decisions = [vote["decisions"][slot] for vote in votes]
            results.append({
                "operation_sha256": row["operation_sha256"],
                "bundle_id": row["bundle_id"],
                "category": row["category"],
                "source_role": row["source_role"],
                "accepted": all(value["status"] == "keep" for value in decisions),
                "votes": decisions,
                "bundle": row["bundle"],
            })
    return {
        "request_status": "complete" if results else "terminal_error",
        "batch_sha256": stable_hash([row["bundle_id"] for row in batch]),
        "operation_sha256s": [row["operation_sha256"] for row in batch],
        "bundle_ids": [row["bundle_id"] for row in batch],
        "validator_calls": votes,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def load_candidates(
    preview_path: Path,
    first_journal: Path,
    synthetic_pairs_path: Path,
    items_path: Path,
) -> list[dict[str, object]]:
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    preview_by_operation = {
        str(row["operation_sha256"]): row for row in preview
    }
    pair_rows = pd.read_parquet(
        synthetic_pairs_path,
        columns=[
            "bundle_id",
            "operation_sha256",
            "source_id1",
            "source_id2",
            "source_role",
        ],
    ).drop_duplicates("bundle_id")
    allowed_by_bundle = {
        str(row.bundle_id): row for row in pair_rows.itertuples(index=False)
    }
    source_ids = set(pair_rows["source_id1"].astype(int)) | set(
        pair_rows["source_id2"].astype(int)
    )
    items = pd.read_parquet(
        items_path, columns=["id", "name", "attributes"]
    ).set_index("id")
    items = items.loc[list(source_ids)]
    candidates = []
    seen: set[str] = set()
    for record in read_jsonl(first_journal):
        if record.get("request_status") != "complete":
            continue
        for result in record.get("results", []):
            decision = result.get("decision", {})
            bundle_id = str(decision.get("bundle_id", ""))
            if decision.get("status") != "choose" or bundle_id not in allowed_by_bundle:
                continue
            if bundle_id in seen:
                continue
            pair = allowed_by_bundle[bundle_id]
            source = preview_by_operation.get(str(result["operation_sha256"]))
            bundle = result.get("selected_bundle")
            if bundle is None and source is not None:
                bundle = next(
                    value for value in source["bundles"] if value["bundle_id"] == bundle_id
                )
            if not isinstance(bundle, dict):
                raise ValueError(f"missing_selected_bundle:{bundle_id}")
            left = items.loc[int(pair.source_id1)]
            right = items.loc[int(pair.source_id2)]
            seen.add(bundle_id)
            candidates.append({
                "operation_sha256": str(pair.operation_sha256),
                "bundle_id": bundle_id,
                "category": str(result["category"]),
                "source_role": str(pair.source_role),
                "item_1": {
                    "name": str(left["name"]),
                    "attributes": left["attributes"],
                },
                "item_2": {
                    "name": str(right["name"]),
                    "attributes": right["attributes"],
                },
                "bundle": bundle,
            })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-preview", type=Path, required=True)
    parser.add_argument("--first-journal", type=Path, required=True)
    parser.add_argument("--synthetic-pairs", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--model-registry", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.workers) <= 0:
        raise ValueError("invalid_limits")
    if args.execute and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(
        args.request_preview, args.first_journal, args.synthetic_pairs, args.items
    )
    (args.output_dir / "validator_preview.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    journal = args.output_dir / "dual_validator_journal.jsonl"
    attempted = {
        str(value)
        for record in read_jsonl(journal)
        for value in record.get("bundle_ids", [])
    }
    selected = [row for row in candidates if row["bundle_id"] not in attempted]
    if args.execute and selected:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=args.base_url.rstrip("/"),
            max_retries=0,
            timeout=args.timeout,
        )
        router = ModelQuotaRouter(args.model_registry, args.output_dir / "model_events.jsonl")
        batches = [
            selected[index:index + args.batch_size]
            for index in range(0, len(selected), args.batch_size)
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for index, batch in enumerate(batches):
                append_jsonl(journal, {
                    "request_status": "submitted",
                    "batch_sha256": stable_hash([row["bundle_id"] for row in batch]),
                    "operation_sha256s": [row["operation_sha256"] for row in batch],
                    "bundle_ids": [row["bundle_id"] for row in batch],
                })
                futures[executor.submit(validate_batch, batch, client, router, index)] = batch
            for future in as_completed(futures):
                append_jsonl(journal, future.result())
    records = read_jsonl(journal)
    results = [
        result
        for record in records if record.get("request_status") == "complete"
        for result in record.get("results", [])
    ]
    accepted = [row for row in results if row["accepted"]]
    with (args.output_dir / "accepted_bundles.jsonl").open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "validator_version": VALIDATOR_VERSION,
        "input_synthetic_pairs_sha256": sha256_file(args.synthetic_pairs),
        "candidates": len(candidates),
        "selected_this_run": len(selected),
        "completed_results": len(results),
        "accepted_by_both": len(accepted),
        "rejected": len(results) - len(accepted),
        "terminal_batches": sum(record.get("request_status") == "terminal_error" for record in records),
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
    }
    (args.output_dir / "dual_validator_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
