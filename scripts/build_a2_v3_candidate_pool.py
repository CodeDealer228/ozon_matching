"""Mine A2 v3 Tier-A sources from the complete canonical human train.

Unlike the legacy 500-row pool, selection is opportunity-first: a source is
kept only when the local fact contract has both a safe non-trivial operation
and a reliable negative anchor.  Both orientations of a human-negative pair
are evaluated because A2 rewrites only item B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import analyze_synthetic_negative_rules as rules
from a2_fact_contract import build_fact_contract, stable_hash


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_BYTES = 6_192_178
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_ITEMS_SHA256 = "07d4f09ac8cfa90a2dfc2a7802dec09fc1e188576e1281e781aa1340b3336f4a"
EXPECTED_TRAIN_ROWS = 256_006
EXPECTED_VAL_ROWS = 109_648
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
TARGET_CATEGORIES = tuple(sorted(rules.SEMANTIC_ALIASES))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_pairs(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        sorted(f"{int(left)}-{int(right)}" for left, right in zip(frame["id1"], frame["id2"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_for_orientation(
    left: dict[str, object], right: dict[str, object], similarity: tuple[float, float, float]
) -> dict[str, object]:
    ordered_hash = stable_hash("A2V3", int(left["id"]), int(right["id"]), 0)
    return {
        "branch": "A2V3",
        "category": str(left["category"]),
        "id1": int(left["id"]),
        "id2": int(right["id"]),
        "label": 0,
        "name1": str(left["name"]),
        "name2": str(right["name"]),
        "attributes1": str(left["attributes"]),
        "attributes2": str(right["attributes"]),
        "name_similarity": float(similarity[0]),
        "sequence_similarity": float(similarity[1]),
        "token_jaccard": float(similarity[2]),
        "source_pair_sha256": ordered_hash,
        "unordered_pair_sha256": stable_hash(
            "A2V3_UNORDERED", min(int(left["id"]), int(right["id"])),
            max(int(left["id"]), int(right["id"])), 0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-path", type=Path,
        default=REPO_ROOT / "data" / "splits" / "matches_item_disjoint_70_30_seed42.parquet",
    )
    parser.add_argument(
        "--items-path", type=Path,
        default=REPO_ROOT / "data" / "items_human.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_candidates",
    )
    parser.add_argument("--max-per-category", type=int, default=100)
    parser.add_argument("--max-item-use", type=int, default=1)
    args = parser.parse_args()
    started = time.time()

    if args.split_path.stat().st_size != EXPECTED_SPLIT_BYTES:
        raise RuntimeError("Canonical split size mismatch")
    if file_sha256(args.split_path) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Canonical split SHA-256 mismatch")
    if file_sha256(args.items_path) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("items_human SHA-256 mismatch")

    pairs = pd.read_parquet(args.split_path, columns=["id1", "id2", "label", "split"])
    train = pairs[pairs["split"].eq("train")].copy()
    val = pairs[pairs["split"].eq("val")].copy()
    if len(train) != EXPECTED_TRAIN_ROWS or len(val) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Canonical split row counts mismatch")
    if fingerprint_pairs(val) != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Validation fingerprint mismatch")
    train_ids = set(map(int, train["id1"])) | set(map(int, train["id2"]))
    val_ids = set(map(int, val["id1"])) | set(map(int, val["id2"]))
    if train_ids & val_ids:
        raise RuntimeError("Train/validation item overlap")

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    items = items[items["id"].isin(train_ids) & items["category"].isin(TARGET_CATEGORIES)].copy()
    records = {
        int(row.id): {
            "id": int(row.id),
            "name": str(row.name or ""),
            "attributes": str(row.attributes or ""),
            "category": str(row.category),
        }
        for row in items.itertuples(index=False)
    }

    candidates: list[tuple[dict[str, object], dict[str, object]]] = []
    scanned_orientations = 0
    eligible_pairs = 0
    for pair in train[train["label"].eq(0)].itertuples(index=False):
        first = records.get(int(pair.id1))
        second = records.get(int(pair.id2))
        if first is None or second is None or first["category"] != second["category"]:
            continue
        similarity = rules.name_similarity(str(first["name"]), str(second["name"]))
        if not (0.30 <= similarity[0] <= 0.95):
            continue
        if min(len(rules.normalize_text(first["name"])), len(rules.normalize_text(second["name"]))) < 8:
            continue
        eligible_pairs += 1
        if eligible_pairs % 10_000 == 0:
            print(
                json.dumps(
                    {
                        "progress_eligible_pairs": eligible_pairs,
                        "scanned_orientations": scanned_orientations,
                        "tier_a_found": len(candidates),
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for left, right in ((first, second), (second, first)):
            scanned_orientations += 1
            source = row_for_orientation(left, right, similarity)
            contract = build_fact_contract(source)
            if contract["tier"] == "A":
                candidates.append((source, contract))

    candidates.sort(
        key=lambda value: (
            str(value[0]["category"]),
            -len(value[1]["allowed_operations"]),
            -len(value[1]["negative_anchors"]),
            -float(value[0]["name_similarity"]),
            str(value[0]["source_pair_sha256"]),
        )
    )
    selected: list[tuple[dict[str, object], dict[str, object]]] = []
    item_use: Counter[int] = Counter()
    category_use: Counter[str] = Counter()
    unordered_seen: set[str] = set()
    for source, contract in candidates:
        category = str(source["category"])
        if category_use[category] >= args.max_per_category:
            continue
        if str(source["unordered_pair_sha256"]) in unordered_seen:
            continue
        if item_use[int(source["id1"])] >= args.max_item_use or item_use[int(source["id2"])] >= args.max_item_use:
            continue
        selected.append((source, contract))
        category_use[category] += 1
        item_use[int(source["id1"])] += 1
        item_use[int(source["id2"])] += 1
        unordered_seen.add(str(source["unordered_pair_sha256"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.DataFrame([source for source, _ in selected])
    pool_path = args.output_dir / "a2_v3_source_pool.parquet"
    pool.to_parquet(pool_path, index=False)
    plan_path = args.output_dir / "a2_v3_generation_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as stream:
        for _, contract in selected:
            stream.write(json.dumps(contract, ensure_ascii=False, sort_keys=True) + "\n")

    selected_ids = set(map(int, pool.get("id1", []))) | set(map(int, pool.get("id2", [])))
    if selected_ids & val_ids:
        raise RuntimeError("Selected A2 v3 pool overlaps validation items")
    summary = {
        "status": "complete",
        "scope": "canonical_human_train_only",
        "validation_content_used": False,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "shared_train_validation_items": 0,
        "scanned_orientations": scanned_orientations,
        "eligible_pairs": eligible_pairs,
        "tier_a_before_caps": len(candidates),
        "selected_rows": len(selected),
        "selected_by_category": dict(sorted(category_use.items())),
        "selected_validation_item_overlap": len(selected_ids & val_ids),
        "max_per_category": args.max_per_category,
        "max_item_use": args.max_item_use,
        "pool_path": str(pool_path.resolve()),
        "pool_sha256": file_sha256(pool_path),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "a2_v3_candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
