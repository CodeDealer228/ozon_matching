"""Build a conservative training subset from fast-v2 counterfactual pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from generate_token_replacement_aggressive_rag import semantic_type
from generate_token_replacement_counterfactuals import (
    parse_attributes,
    replace_text,
    sha256_file,
    stable_synthetic_id,
)
from generate_token_replacement_fast_v2 import (
    attribute_leaves,
    infer_source_role,
    key_role,
)


FILTER_VERSION = 1
ALLOWED_SPEC_ROLES = {
    "volume", "weight", "count", "length", "width", "height", "diameter",
    "power", "resolution", "format", "size",
}
UNIT_RE = re.compile(r"[a-zа-я]+$", re.IGNORECASE)


def unit_suffix(value: str) -> str:
    match = UNIT_RE.search(value.strip())
    return match.group(0).casefold() if match else ""


def role_occurrences(attributes: object, role: str, token: str) -> int:
    return sum(
        replace_text(str(value), token, "__probe__")[1]
        for key, value in attribute_leaves(attributes)
        if key_role(key) == role
    )


def replace_role_attributes(
    value: object,
    role: str,
    source: str,
    target: str,
    path: tuple[str, ...] = (),
) -> tuple[object, int]:
    if isinstance(value, dict):
        output: dict[str, object] = {}
        changes = 0
        for key, nested in value.items():
            output[key], changed = replace_role_attributes(
                nested, role, source, target, (*path, str(key))
            )
            changes += changed
        return output, changes
    if isinstance(value, list):
        output_list: list[object] = []
        changes = 0
        for nested in value:
            replaced, changed = replace_role_attributes(
                nested, role, source, target, path
            )
            output_list.append(replaced)
            changes += changed
        return output_list, changes
    if key_role(" ".join(path).casefold()) != role:
        return value, 0
    if isinstance(value, str):
        return replace_text(value, source, target)
    return value, 0


def mutation_rejection(
    left: pd.Series,
    right: pd.Series,
    source: str,
    target: str,
) -> tuple[str | None, str | None]:
    role = infer_source_role(left, right, source)
    if role not in ALLOWED_SPEC_ROLES:
        return role, "identity_or_unsupported_role"
    if semantic_type(source) != semantic_type(target):
        return role, "semantic_type_mismatch"
    source_unit = unit_suffix(source)
    target_unit = unit_suffix(target)
    if source_unit != target_unit:
        return role, "unit_suffix_mismatch"
    for item in (left, right):
        if replace_text(str(item["name"]), source, "__probe__")[1] == 0:
            return role, "source_missing_from_name"
        attributes = parse_attributes(item["attributes"])
        if role_occurrences(attributes, role, source) == 0:
            return role, "source_missing_from_role_attribute"
    return role, None


def mutate_spec_item(
    item: pd.Series,
    role: str,
    source: str,
    target: str,
    synthetic_id: int,
) -> tuple[dict[str, object], int]:
    name, name_changes = replace_text(str(item["name"]), source, target)
    attributes = parse_attributes(item["attributes"])
    changed_attributes, attribute_changes = replace_role_attributes(
        attributes, role, source, target
    )
    if name_changes == 0 or attribute_changes == 0:
        raise ValueError("strict_spec_replacement_not_applied")
    return {
        "id": synthetic_id,
        "name": name,
        "category": str(item["category"]),
        "attributes": json.dumps(
            changed_attributes, ensure_ascii=False, separators=(",", ":")
        ),
    }, name_changes + attribute_changes


def unordered_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (min(int(row.id1), int(row.id2)), max(int(row.id1), int(row.id2)))
        for row in frame.itertuples(index=False)
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument(
        "--canonical-split",
        type=Path,
        default=Path("data/splits/matches_item_disjoint_70_30_seed42.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_pairs = pd.read_parquet(args.generation_dir / "synthetic_pairs.parquet")
    operations = input_pairs.drop_duplicates("operation_sha256")
    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id", drop=False)
    validation = pd.read_parquet(args.canonical_split)
    if "split" not in validation.columns:
        raise RuntimeError("canonical_split_column_missing")
    validation = validation[validation["split"].eq("val")]
    validation_ids = set(validation["id1"]) | set(validation["id2"])

    rejected: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    kept_role_counts: Counter[str] = Counter()
    synthetic_items: list[dict[str, object]] = []
    synthetic_pairs: list[dict[str, object]] = []
    preview: list[dict[str, object]] = []

    for operation in operations.itertuples(index=False):
        left = items.loc[int(operation.source_id1)]
        right = items.loc[int(operation.source_id2)]
        source = str(operation.source_token)
        target = str(operation.replacement_token)
        role, rejection = mutation_rejection(left, right, source, target)
        role_counts[str(role or "none")] += 1
        if rejection is not None:
            rejected[rejection] += 1
            continue
        assert role is not None
        clean_key = f"clean-spec-v{FILTER_VERSION}:{operation.operation_sha256}"
        left_id = stable_synthetic_id(clean_key, "left")
        right_id = stable_synthetic_id(clean_key, "right")
        left_prime, left_changes = mutate_spec_item(
            left, role, source, target, left_id
        )
        right_prime, right_changes = mutate_spec_item(
            right, role, source, target, right_id
        )
        base_item = {
            "source_pair_sha256": str(operation.source_pair_sha256),
            "source_token": source,
            "replacement_token": target,
            "source_role": role,
            "quality_tier": "strict_role_aware_spec_v1",
        }
        left_prime.update({
            **base_item,
            "source_item_id": int(operation.source_id1),
            "replacement_count": left_changes,
        })
        right_prime.update({
            **base_item,
            "source_item_id": int(operation.source_id2),
            "replacement_count": right_changes,
        })
        synthetic_items.extend([left_prime, right_prime])
        base_pair = {
            "source_id1": int(operation.source_id1),
            "source_id2": int(operation.source_id2),
            "source_pair_sha256": str(operation.source_pair_sha256),
            "source_token": source,
            "replacement_token": target,
            "operation_sha256": str(operation.operation_sha256),
            "source_role": role,
            "quality_tier": "strict_role_aware_spec_v1",
            "sample_weight": 0.5,
        }
        synthetic_pairs.append({
            **base_pair,
            "id1": left_id,
            "id2": right_id,
            "target": 1,
            "variant": "both_replaced",
        })
        replace_left = int(str(operation.operation_sha256)[-1], 16) % 2 == 0
        synthetic_pairs.append({
            **base_pair,
            "id1": left_id if replace_left else int(operation.source_id1),
            "id2": int(operation.source_id2) if replace_left else right_id,
            "target": 0,
            "variant": "left_only" if replace_left else "right_only",
        })
        kept_role_counts[role] += 1
        if len(preview) < 100:
            preview.append({
                "category": str(left["category"]),
                "role": role,
                "source": source,
                "target": target,
                "source_name_1": str(left["name"]),
                "source_name_2": str(right["name"]),
                "synthetic_name_1": left_prime["name"],
                "synthetic_name_2": right_prime["name"],
            })

    item_frame = pd.DataFrame(synthetic_items)
    pair_frame = pd.DataFrame(synthetic_pairs)
    if item_frame["id"].duplicated().any():
        raise RuntimeError("synthetic_item_id_duplicates")
    if pair_frame[["id1", "id2"]].duplicated().any():
        raise RuntimeError("ordered_pair_duplicates")
    if len(unordered_pairs(pair_frame)) != len(pair_frame):
        raise RuntimeError("unordered_pair_duplicates")
    per_operation = pair_frame.groupby("operation_sha256")["target"].agg(list)
    if any(sorted(values) != [0, 1] for values in per_operation):
        raise RuntimeError("operation_pair_balance_failed")
    source_ids = set(pair_frame["source_id1"]) | set(pair_frame["source_id2"])
    synthetic_ids = set(item_frame["id"])
    validation_source_overlap = len(source_ids & validation_ids)
    validation_synthetic_overlap = len(synthetic_ids & validation_ids)
    if validation_source_overlap or validation_synthetic_overlap:
        raise RuntimeError("validation_item_overlap")

    item_path = args.output_dir / "synthetic_items.parquet"
    pair_path = args.output_dir / "synthetic_pairs.parquet"
    item_frame.to_parquet(item_path, index=False)
    pair_frame.to_parquet(pair_path, index=False)
    with (args.output_dir / "quality_preview.jsonl").open("w", encoding="utf-8") as handle:
        for row in preview:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    category_counts = (
        item_frame.drop_duplicates("source_pair_sha256")["category"].value_counts().sort_index()
    )
    manifest = {
        "status": "complete",
        "filter_version": FILTER_VERSION,
        "input_pairs_sha256": sha256_file(args.generation_dir / "synthetic_pairs.parquet"),
        "input_operations": int(len(operations)),
        "kept_operations": int(len(per_operation)),
        "kept_pairs": int(len(pair_frame)),
        "positive_pairs": int(pair_frame["target"].eq(1).sum()),
        "negative_pairs": int(pair_frame["target"].eq(0).sum()),
        "role_counts": dict(sorted(role_counts.items())),
        "kept_role_counts": dict(sorted(kept_role_counts.items())),
        "rejection_counts": dict(sorted(rejected.items())),
        "category_source_counts": {
            str(key): int(value) for key, value in category_counts.items()
        },
        "ordered_pair_duplicates": 0,
        "unordered_pair_duplicates": 0,
        "validation_source_item_overlap": validation_source_overlap,
        "validation_synthetic_item_overlap": validation_synthetic_overlap,
        "old_identity_operations_used": 0,
        "grounded_pilot_rows_used": 0,
        "sample_weight": 0.5,
        "synthetic_items_sha256": sha256_file(item_path),
        "synthetic_pairs_sha256": sha256_file(pair_path),
    }
    write_json(args.output_dir / "clean_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
