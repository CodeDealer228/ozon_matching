"""Generate counterfactuals from prebuilt identity bundles grounded in real cards."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from alibaba_model_quota_router import ModelQuotaRouter
from generate_token_replacement_aggressive_rag import semantic_type
from generate_token_replacement_counterfactuals import (
    DEFAULT_BASE_URL,
    append_jsonl,
    read_jsonl,
    sha256_file,
    stable_synthetic_id,
)
from generate_token_replacement_fast_v2 import public_item
from judge_token_replacement_grounded_bundles import (
    apply_bundle,
    build_bundle,
    identity_fields,
    local_bundle_rejection,
    process_batch,
    stable_hash,
)
from repair_token_replacement_identity_bundles import (
    balanced_sample,
    explicit_identity_role,
)


PIPELINE_VERSION = 1


def full_target_cards(
    source_row: dict[str, object],
    items: pd.DataFrame,
    name_index: dict[tuple[str, str], list[int]],
) -> list[pd.Series]:
    excluded = {int(source_row["id1"]), int(source_row["id2"])}
    output: list[pd.Series] = []
    seen: set[int] = set()
    for example in source_row["rag_examples"]:
        if example.get("relation") != "different_product":
            continue
        for key in ("item_1", "item_2"):
            evidence = example[key]
            name_key = (
                str(source_row["category"]),
                str(evidence.get("name", "")).casefold(),
            )
            for item_id in name_index.get(name_key, [])[:3]:
                if item_id in excluded or item_id in seen:
                    continue
                seen.add(item_id)
                output.append(items.loc[item_id])
    return output


def target_tokens(target: pd.Series, role: str, source_token: str) -> list[str]:
    fields = identity_fields(target)
    kinds = ("brand",) if role == "brand" else ("model", "article")
    output: list[str] = []
    seen: set[str] = set()
    for kind in kinds:
        for _, value in fields.get(kind, []):
            folded = value.casefold()
            if folded in seen or folded == source_token.casefold():
                continue
            if semantic_type(value) != semantic_type(source_token):
                continue
            seen.add(folded)
            output.append(value)
    return output


def prepare_rows(
    source_pack: Path,
    items: pd.DataFrame,
    excluded_source_pairs: set[str],
) -> tuple[list[dict[str, object]], Counter[str]]:
    source_rows = read_jsonl(source_pack)
    name_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in items.itertuples(index=False):
        name_index[(str(item.category), str(item.name).casefold())].append(int(item.id))
    output: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    for source_row in source_rows:
        source_hash = str(source_row["source_pair_sha256"])
        if source_hash in excluded_source_pairs:
            rejected["source_used_by_old_generation"] += 1
            continue
        left = items.loc[int(source_row["id1"])]
        right = items.loc[int(source_row["id2"])]
        cards = full_target_cards(source_row, items, name_index)
        if not cards:
            rejected["no_full_rag_target_card"] += 1
            continue
        for source_token in [str(value) for value in source_row["offered_tokens"]]:
            role = explicit_identity_role(left, right, source_token)
            if role not in {"brand", "article"}:
                continue
            bundles: list[dict[str, object]] = []
            seen_bundles: set[str] = set()
            operation_sha = stable_hash({
                "pipeline": PIPELINE_VERSION,
                "source_pair_sha256": source_hash,
                "source_role": role,
                "source_token": source_token,
            })
            base_row = {
                "operation_sha256": operation_sha,
                "source_pair_sha256": source_hash,
                "id1": int(source_row["id1"]),
                "id2": int(source_row["id2"]),
                "category": str(source_row["category"]),
                "source_role": role,
                "source_token": source_token,
            }
            for target in cards:
                for replacement in target_tokens(target, role, source_token):
                    candidate_row = {**base_row, "replacement_token": replacement}
                    bundle = build_bundle(candidate_row, left, right, target)
                    if bundle is None or bundle["bundle_id"] in seen_bundles:
                        continue
                    try:
                        preview_1 = apply_bundle(left, bundle["mappings"])
                        preview_2 = apply_bundle(right, bundle["mappings"])
                    except ValueError:
                        continue
                    bundle["preview_name_1"] = preview_1["name"]
                    bundle["preview_name_2"] = preview_2["name"]
                    rejection = local_bundle_rejection(
                        candidate_row, bundle, left, right
                    )
                    if rejection is not None:
                        rejected[rejection.split(":", 1)[0]] += 1
                        continue
                    seen_bundles.add(str(bundle["bundle_id"]))
                    bundles.append(bundle)
                    if len(bundles) >= 4:
                        break
                if len(bundles) >= 4:
                    break
            if not bundles:
                rejected["no_safe_bundle_for_source_token"] += 1
                continue
            output.append({
                **base_row,
                "item_1": public_item(left, max_attribute_chars=1100),
                "item_2": public_item(right, max_attribute_chars=1100),
                "bundles": bundles,
            })
    return output, rejected


def materialize(
    results: list[dict[str, object]],
    prepared: list[dict[str, object]],
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    prepared_by_operation = {
        str(row["operation_sha256"]): row for row in prepared
    }
    synthetic_items: list[dict[str, object]] = []
    synthetic_pairs: list[dict[str, object]] = []
    stale = 0
    for result in results:
        if result["decision"]["status"] != "choose":
            continue
        row = prepared_by_operation.get(str(result["operation_sha256"]))
        if row is None:
            stale += 1
            continue
        bundle = next(
            (
                value for value in row["bundles"]
                if value["bundle_id"] == result["decision"]["bundle_id"]
            ),
            None,
        )
        if bundle is None:
            stale += 1
            continue
        left = items.loc[int(row["id1"])]
        right = items.loc[int(row["id2"])]
        left_changed = apply_bundle(left, bundle["mappings"])
        right_changed = apply_bundle(right, bundle["mappings"])
        operation_sha = stable_hash({
            "source_pair_sha256": row["source_pair_sha256"],
            "bundle_id": bundle["bundle_id"],
        })
        left_id = stable_synthetic_id(operation_sha, "left")
        right_id = stable_synthetic_id(operation_sha, "right")
        mapping_text = json.dumps(
            bundle["mappings"], ensure_ascii=False, separators=(",", ":")
        )
        for source, changed, synthetic_id, side in (
            (left, left_changed, left_id, "left"),
            (right, right_changed, right_id, "right"),
        ):
            synthetic_items.append({
                "id": synthetic_id,
                "name": changed["name"],
                "category": str(source["category"]),
                "attributes": json.dumps(
                    changed["attributes"], ensure_ascii=False, separators=(",", ":")
                ),
                "source_item_id": int(source["id"]),
                "source_pair_sha256": row["source_pair_sha256"],
                "operation_sha256": operation_sha,
                "bundle_id": bundle["bundle_id"],
                "source_role": row["source_role"],
                "identity_mappings": mapping_text,
                "side": side,
            })
        base = {
            "source_id1": int(row["id1"]),
            "source_id2": int(row["id2"]),
            "source_pair_sha256": row["source_pair_sha256"],
            "operation_sha256": operation_sha,
            "bundle_id": bundle["bundle_id"],
            "source_role": row["source_role"],
            "quality_tier": "grounded_bundle_llm_judged_v1",
            "sample_weight": 0.5,
        }
        synthetic_pairs.append({
            **base, "id1": left_id, "id2": right_id,
            "target": 1, "variant": "both_replaced",
        })
        replace_left = int(operation_sha[-1], 16) % 2 == 0
        synthetic_pairs.append({
            **base,
            "id1": left_id if replace_left else int(row["id1"]),
            "id2": int(row["id2"]) if replace_left else right_id,
            "target": 0,
            "variant": "left_only" if replace_left else "right_only",
        })
    return pd.DataFrame(synthetic_items), pd.DataFrame(synthetic_pairs), stale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--exclude-pairs", type=Path, action="append", default=[])
    parser.add_argument("--model-registry", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.count < 0 or min(args.batch_size, args.workers) <= 0:
        raise ValueError("invalid_limits")
    if args.execute and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id", drop=False)
    excluded_source_pairs: set[str] = set()
    for path in args.exclude_pairs:
        frame = pd.read_parquet(path, columns=["source_pair_sha256"])
        excluded_source_pairs.update(frame["source_pair_sha256"].astype(str))
    prepared, deterministic_rejections = prepare_rows(
        args.source_pack, items, excluded_source_pairs
    )
    journal = args.output_dir / "selection_journal.jsonl"
    records = read_jsonl(journal)
    attempted = {
        str(value)
        for record in records
        for value in record.get("operation_sha256s", [])
    }
    remaining = [
        row for row in prepared if str(row["operation_sha256"]) not in attempted
    ]
    selected = [] if args.count == 0 else balanced_sample(remaining, args.count)
    preview = [{
        "operation_sha256": row["operation_sha256"],
        "source_pair_sha256": row["source_pair_sha256"],
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
        batches = [
            selected[index:index + args.batch_size]
            for index in range(0, len(selected), args.batch_size)
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for index, batch in enumerate(batches):
                submitted = {
                    "request_status": "submitted",
                    "batch_sha256": stable_hash(
                        [row["operation_sha256"] for row in batch]
                    ),
                    "operation_sha256s": [
                        row["operation_sha256"] for row in batch
                    ],
                }
                append_jsonl(journal, submitted)
                futures[executor.submit(
                    process_batch, batch, client, router, index
                )] = submitted
            for future in as_completed(futures):
                append_jsonl(journal, future.result())
    records = read_jsonl(journal)
    results = [
        result
        for record in records if record.get("request_status") == "complete"
        for result in record.get("results", [])
    ]
    synthetic_items, synthetic_pairs, stale = materialize(results, prepared, items)
    if not synthetic_pairs.empty:
        if synthetic_items["id"].duplicated().any():
            raise RuntimeError("synthetic_item_duplicates")
        if synthetic_pairs[["id1", "id2"]].duplicated().any():
            raise RuntimeError("synthetic_pair_duplicates")
        targets = synthetic_pairs.groupby("operation_sha256")["target"].agg(list)
        if any(sorted(values) != [0, 1] for values in targets):
            raise RuntimeError("target_balance_failed")
        synthetic_items.to_parquet(args.output_dir / "synthetic_items.parquet", index=False)
        synthetic_pairs.to_parquet(args.output_dir / "synthetic_pairs.parquet", index=False)
    chosen = int(synthetic_pairs["operation_sha256"].nunique()) if not synthetic_pairs.empty else 0
    manifest = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "source_pack_sha256": sha256_file(args.source_pack),
        "excluded_source_pairs": len(excluded_source_pairs),
        "prepared_operations": len(prepared),
        "deterministic_rejections": dict(sorted(deterministic_rejections.items())),
        "selected_this_run": len(selected),
        "completed_results": len(results),
        "chosen_operations": chosen,
        "synthetic_pairs": int(len(synthetic_pairs)),
        "positive_pairs": int(synthetic_pairs["target"].eq(1).sum()) if not synthetic_pairs.empty else 0,
        "negative_pairs": int(synthetic_pairs["target"].eq(0).sum()) if not synthetic_pairs.empty else 0,
        "stale_decisions": stale,
        "terminal_batches": sum(
            record.get("request_status") == "terminal_error" for record in records
        ),
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
    }
    if not synthetic_pairs.empty:
        manifest["synthetic_items_sha256"] = sha256_file(
            args.output_dir / "synthetic_items.parquet"
        )
        manifest["synthetic_pairs_sha256"] = sha256_file(
            args.output_dir / "synthetic_pairs.parquet"
        )
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
