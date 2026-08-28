"""Generate balanced counterfactual pairs from real, role-matched spec values."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd

from analyze_spec_replacement_capacity import shared_name_tokens, strict_source_role
from filter_token_replacement_training_candidates import (
    ALLOWED_SPEC_ROLES,
    mutate_spec_item,
    mutation_rejection,
    unordered_pairs,
)
from generate_token_replacement_counterfactuals import (
    read_jsonl,
    replace_text,
    sha256_file,
    stable_synthetic_id,
)
from generate_token_replacement_fast_v2 import (
    ReplacementBank,
    context_tokens,
    infer_source_role,
)


PIPELINE_VERSION = 2
NUMBER_VALUE_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
RATIO_LIMITED_ROLES = {
    "volume", "weight", "length", "width", "height", "diameter"
}


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def collect_sources(pairs: pd.DataFrame, items: pd.DataFrame) -> list[dict[str, object]]:
    output = []
    for pair in pairs.itertuples(index=False):
        if pair.id1 not in items.index or pair.id2 not in items.index:
            continue
        left = items.loc[int(pair.id1)]
        right = items.loc[int(pair.id2)]
        offered = []
        for token in shared_name_tokens(str(left["name"]), str(right["name"])):
            role = strict_source_role(left, right, token)
            if role is None:
                continue
            _, rejection = mutation_rejection(left, right, token, token)
            if rejection is None:
                offered.append(token)
        if not offered:
            continue
        output.append({
            "id1": int(pair.id1),
            "id2": int(pair.id2),
            "category": str(pair.category),
            "source_pair_sha256": stable_hash({
                "id1": min(int(pair.id1), int(pair.id2)),
                "id2": max(int(pair.id1), int(pair.id2)),
            }),
            "offered_tokens": offered,
        })
    return output


def candidate_operations(
    sources: list[dict[str, object]],
    items: pd.DataFrame,
    targets_per_token: int,
    minimum_context_overlap: int,
) -> list[dict[str, object]]:
    bank = ReplacementBank(sources, items)
    output = []
    seen: set[str] = set()
    for row in sources:
        left = items.loc[int(row["id1"])]
        right = items.loc[int(row["id2"])]
        context = context_tokens(f"{left['name']} {right['name']}")
        for source in row["offered_tokens"]:
            role = infer_source_role(left, right, str(source))
            if role not in ALLOWED_SPEC_ROLES:
                continue
            candidates = bank.candidates(
                str(row["category"]),
                str(source),
                str(role),
                context,
                set(str(value) for value in row["offered_tokens"]),
                max(12, targets_per_token * 6),
            )
            accepted = 0
            for candidate in candidates:
                target = str(candidate["replacement_token"])
                if int(candidate.get("context_overlap", 0)) < minimum_context_overlap:
                    continue
                if replace_text(str(left["name"]), target, "__probe__")[1]:
                    continue
                if replace_text(str(right["name"]), target, "__probe__")[1]:
                    continue
                evidence_context = context_tokens(str(candidate.get("evidence_name", "")))
                title_context_overlap = len(context & evidence_context)
                if title_context_overlap < 2:
                    continue
                checked_role, rejection = mutation_rejection(
                    left, right, str(source), target
                )
                if rejection is not None or checked_role != role:
                    continue
                if role in RATIO_LIMITED_ROLES:
                    source_number = str(source).replace(",", ".")
                    target_number = target.replace(",", ".")
                    if NUMBER_VALUE_RE.fullmatch(str(source)) and NUMBER_VALUE_RE.fullmatch(target):
                        ratio = float(target_number) / max(float(source_number), 1e-12)
                        if not 0.1 <= ratio <= 10.0:
                            continue
                operation_sha = stable_hash({
                    "source_pair_sha256": row["source_pair_sha256"],
                    "source": str(source).casefold(),
                    "target": target.casefold(),
                    "role": role,
                    "pipeline_version": PIPELINE_VERSION,
                })
                if operation_sha in seen:
                    continue
                seen.add(operation_sha)
                output.append({
                    **row,
                    "source_token": str(source),
                    "replacement_token": target,
                    "source_role": str(role),
                    "operation_sha256": operation_sha,
                    "evidence_name": str(candidate.get("evidence_name", "")),
                    "bank_frequency": int(candidate.get("bank_frequency", 0)),
                    "context_overlap": int(candidate.get("context_overlap", 0)),
                    "title_context_overlap": title_context_overlap,
                })
                accepted += 1
                if accepted >= targets_per_token:
                    break
    return output


def balanced_select(
    operations: list[dict[str, object]], target_operations: int
) -> list[dict[str, object]]:
    by_category: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    for row in sorted(
        operations,
        key=lambda value: (
            -int(value["context_overlap"]),
            -int(value["bank_frequency"]),
            str(value["operation_sha256"]),
        ),
    ):
        by_category[str(row["category"])].append(row)
    selected = []
    categories = sorted(by_category)
    while len(selected) < target_operations:
        progressed = False
        for category in categories:
            if by_category[category]:
                selected.append(by_category[category].popleft())
                progressed = True
                if len(selected) == target_operations:
                    break
        if not progressed:
            break
    return selected


def materialize(
    operations: list[dict[str, object]], items: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    synthetic_items = []
    synthetic_pairs = []
    preview = []
    for operation in operations:
        left = items.loc[int(operation["id1"])]
        right = items.loc[int(operation["id2"])]
        key = f"grounded-spec-v{PIPELINE_VERSION}:{operation['operation_sha256']}"
        left_id = stable_synthetic_id(key, "left")
        right_id = stable_synthetic_id(key, "right")
        left_prime, left_changes = mutate_spec_item(
            left,
            str(operation["source_role"]),
            str(operation["source_token"]),
            str(operation["replacement_token"]),
            left_id,
        )
        right_prime, right_changes = mutate_spec_item(
            right,
            str(operation["source_role"]),
            str(operation["source_token"]),
            str(operation["replacement_token"]),
            right_id,
        )
        item_meta = {
            "source_pair_sha256": operation["source_pair_sha256"],
            "operation_sha256": operation["operation_sha256"],
            "source_token": operation["source_token"],
            "replacement_token": operation["replacement_token"],
            "source_role": operation["source_role"],
            "quality_tier": "grounded_role_spec_v2",
        }
        left_prime.update({
            **item_meta,
            "source_item_id": int(operation["id1"]),
            "replacement_count": left_changes,
        })
        right_prime.update({
            **item_meta,
            "source_item_id": int(operation["id2"]),
            "replacement_count": right_changes,
        })
        synthetic_items.extend([left_prime, right_prime])
        pair_meta = {
            "source_id1": int(operation["id1"]),
            "source_id2": int(operation["id2"]),
            **item_meta,
            "evidence_name": operation["evidence_name"],
            "context_overlap": int(operation["context_overlap"]),
            "title_context_overlap": int(operation["title_context_overlap"]),
            "sample_weight": 0.5,
        }
        synthetic_pairs.append({
            **pair_meta, "id1": left_id, "id2": right_id,
            "target": 1, "variant": "both_replaced",
        })
        replace_left = int(str(operation["operation_sha256"])[-1], 16) % 2 == 0
        synthetic_pairs.append({
            **pair_meta,
            "id1": left_id if replace_left else int(operation["id1"]),
            "id2": int(operation["id2"]) if replace_left else right_id,
            "target": 0,
            "variant": "left_only" if replace_left else "right_only",
        })
        preview.append({
            "category": str(left["category"]),
            "role": operation["source_role"],
            "source": operation["source_token"],
            "target": operation["replacement_token"],
            "source_name_1": str(left["name"]),
            "source_name_2": str(right["name"]),
            "synthetic_name_1": left_prime["name"],
            "synthetic_name_2": right_prime["name"],
            "evidence_name": operation["evidence_name"],
            "context_overlap": int(operation["context_overlap"]),
            "title_context_overlap": int(operation["title_context_overlap"]),
        })
    return pd.DataFrame(synthetic_items), pd.DataFrame(synthetic_pairs), preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/splits/matches_item_disjoint_70_30_seed42.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-operations", type=Path)
    parser.add_argument("--target-operations", type=int, default=6000)
    parser.add_argument("--targets-per-token", type=int, default=2)
    parser.add_argument("--minimum-context-overlap", type=int, default=2)
    args = parser.parse_args()
    if min(
        args.target_operations, args.targets_per_token, args.minimum_context_overlap
    ) <= 0:
        raise ValueError("limits_must_be_positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id", drop=False)
    all_pairs = pd.read_parquet(
        args.pairs, columns=["id1", "id2", "target", "category", "split"]
    )
    train_positive = all_pairs[
        all_pairs["split"].eq("train") & all_pairs["target"].eq(1)
    ]
    validation = all_pairs[all_pairs["split"].eq("val")]
    validation_ids = set(validation["id1"]) | set(validation["id2"])
    if args.prepared_operations:
        operations = read_jsonl(args.prepared_operations)
        strict_source_count = len({
            str(row["source_pair_sha256"]) for row in operations
        })
        prepared_path = args.prepared_operations
        print(f"loaded_prepared_operations={len(operations)}", flush=True)
    else:
        sources = collect_sources(train_positive, items)
        strict_source_count = len(sources)
        print(f"strict_spec_sources={strict_source_count}", flush=True)
        operations = candidate_operations(
            sources, items, args.targets_per_token, args.minimum_context_overlap
        )
        prepared_path = args.output_dir / "prepared_operations.jsonl"
        with prepared_path.open("w", encoding="utf-8") as handle:
            for row in operations:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"candidate_operations={len(operations)}", flush=True)
    selected = balanced_select(operations, args.target_operations)
    item_frame, pair_frame, preview = materialize(selected, items)
    if not selected:
        raise RuntimeError("no_operations_selected")
    if item_frame["id"].duplicated().any():
        raise RuntimeError("synthetic_item_id_duplicates")
    if pair_frame[["id1", "id2"]].duplicated().any():
        raise RuntimeError("ordered_pair_duplicates")
    if len(unordered_pairs(pair_frame)) != len(pair_frame):
        raise RuntimeError("unordered_pair_duplicates")
    targets = pair_frame.groupby("operation_sha256")["target"].agg(list)
    if any(sorted(value) != [0, 1] for value in targets):
        raise RuntimeError("operation_pair_balance_failed")
    source_ids = set(pair_frame["source_id1"]) | set(pair_frame["source_id2"])
    if source_ids & validation_ids:
        raise RuntimeError("validation_source_overlap")

    item_path = args.output_dir / "synthetic_items.parquet"
    pair_path = args.output_dir / "synthetic_pairs.parquet"
    item_frame.to_parquet(item_path, index=False)
    pair_frame.to_parquet(pair_path, index=False)
    with (args.output_dir / "quality_preview.jsonl").open("w", encoding="utf-8") as handle:
        for row in preview:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    category_counts = Counter(str(row["category"]) for row in selected)
    role_counts = Counter(str(row["source_role"]) for row in selected)
    manifest = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "train_positive_rows": int(len(train_positive)),
        "strict_spec_sources": strict_source_count,
        "candidate_operations": len(operations),
        "prepared_operations_sha256": sha256_file(prepared_path),
        "target_operations": args.target_operations,
        "selected_operations": len(selected),
        "target_reached": len(selected) == args.target_operations,
        "synthetic_pairs": len(pair_frame),
        "positive_pairs": int(pair_frame["target"].eq(1).sum()),
        "negative_pairs": int(pair_frame["target"].eq(0).sum()),
        "category_counts": dict(sorted(category_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "targets_per_token": args.targets_per_token,
        "minimum_context_overlap": args.minimum_context_overlap,
        "ordered_pair_duplicates": 0,
        "unordered_pair_duplicates": 0,
        "validation_source_item_overlap": 0,
        "ids_sent": False,
        "labels_sent": False,
        "validation_sent": False,
        "secrets_sent": False,
        "synthetic_items_sha256": sha256_file(item_path),
        "synthetic_pairs_sha256": sha256_file(pair_path),
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
