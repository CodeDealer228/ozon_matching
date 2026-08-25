"""Build the leak-checked all-category source pool for A2 v4.

The pool contains both orientations of every canonical human-train negative
pair.  Validation is read only to prove that neither items nor unordered pairs
leak into the generation source.  No validation content is written to output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_BYTES = 6_192_178
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
EXPECTED_TRAIN_ROWS = 256_006
EXPECTED_VAL_ROWS = 109_648
EXPECTED_CATEGORY_COUNT = 20
TARGET_ACCEPTED_PER_CATEGORY = 10_000
MAX_CANDIDATES_PER_SOURCE = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def unordered_key(left: object, right: object) -> tuple[int, int]:
    first, second = int(left), int(right)
    return (min(first, second), max(first, second))


def unordered_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {unordered_key(left, right) for left, right in zip(frame.id1, frame.id2)}


def fingerprint_pairs(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        sorted(f"{int(left)}-{int(right)}" for left, right in zip(frame.id1, frame.id2))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-path",
        type=Path,
        default=REPO_ROOT / "data/splits/matches_item_disjoint_70_30_seed42.parquet",
    )
    parser.add_argument(
        "--items-path", type=Path, default=REPO_ROOT / "data/items_human.parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/a2_scale_200k/source_pool",
    )
    parser.add_argument("--target-per-category", type=int, default=TARGET_ACCEPTED_PER_CATEGORY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_path = args.split_path.resolve()
    items_path = args.items_path.resolve()
    if split_path.stat().st_size != EXPECTED_SPLIT_BYTES:
        raise RuntimeError(f"Canonical split size mismatch: {split_path.stat().st_size}")
    split_sha = file_sha256(split_path)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Canonical split SHA-256 mismatch: {split_sha}")

    pairs = pd.read_parquet(split_path, columns=["id1", "id2", "label", "split"])
    train = pairs[pairs.split.eq("train")].copy()
    val = pairs[pairs.split.eq("val")].copy()
    if len(train) != EXPECTED_TRAIN_ROWS or len(val) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected split counts: train={len(train)} val={len(val)}")
    if fingerprint_pairs(val) != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Validation fingerprint mismatch")

    train_ids = set(map(int, train.id1)) | set(map(int, train.id2))
    val_ids = set(map(int, val.id1)) | set(map(int, val.id2))
    train_val_item_overlap = train_ids & val_ids
    train_val_pair_overlap = unordered_pairs(train) & unordered_pairs(val)
    if train_val_item_overlap or train_val_pair_overlap:
        raise RuntimeError(
            "Canonical split leak: "
            f"items={len(train_val_item_overlap)} pairs={len(train_val_pair_overlap)}"
        )

    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    if items.id.duplicated().any():
        raise RuntimeError("items_human contains duplicate item IDs")
    items = items[items.id.isin(train_ids)].copy()
    item_map = items.set_index("id")
    category_by_id = item_map.category

    negatives = train[train.label.eq(0)].copy()
    negatives["category1"] = negatives.id1.map(category_by_id)
    negatives["category2"] = negatives.id2.map(category_by_id)
    negatives = negatives[
        negatives.category1.notna() & negatives.category1.eq(negatives.category2)
    ].copy()
    negatives["pair_key"] = [
        f"{left}:{right}" for left, right in map(unordered_key, negatives.id1, negatives.id2)
    ]
    duplicates_before = int(negatives.pair_key.duplicated().sum())
    negatives = negatives.sort_values(["pair_key", "id1", "id2"]).drop_duplicates("pair_key")
    categories = sorted(map(str, negatives.category1.unique()))
    if len(categories) != EXPECTED_CATEGORY_COUNT:
        raise RuntimeError(f"Expected 20 categories, found {len(categories)}: {categories}")

    # Build the two orientations vectorially.  Per-row ``DataFrame.loc`` is
    # prohibitively slow for the full 20-category pool.
    pair_keys = [unordered_key(left, right) for left, right in zip(negatives.id1, negatives.id2)]
    pair_hashes = [
        stable_hash(EXPECTED_SPLIT_SHA256, left, right, 0) for left, right in pair_keys
    ]
    name_by_id = item_map["name"]
    attributes_by_id = item_map["attributes"]

    def orientation_frame(a_column: str, b_column: str, orientation: str) -> pd.DataFrame:
        oriented = pd.DataFrame(
            {
                "source_pair_sha256": pair_hashes,
                "orientation": orientation,
                "category": negatives.category1.astype(str).to_numpy(),
                "label": 0,
                "id1": negatives[a_column].astype("int64").to_numpy(),
                "id2": negatives[b_column].astype("int64").to_numpy(),
            }
        )
        oriented["source_orientation_sha256"] = [
            stable_hash(pair_hash, a_id, b_id)
            for pair_hash, a_id, b_id in zip(
                oriented.source_pair_sha256, oriented.id1, oriented.id2
            )
        ]
        oriented["name1"] = oriented.id1.map(name_by_id).fillna("").astype(str)
        oriented["name2"] = oriented.id2.map(name_by_id).fillna("").astype(str)
        oriented["attributes2"] = oriented.id2.map(attributes_by_id).fillna("").astype(str)
        return oriented

    pool = pd.concat(
        [
            orientation_frame("id1", "id2", "forward"),
            orientation_frame("id2", "id1", "reverse"),
        ],
        ignore_index=True,
    ).sort_values(
        ["category", "source_orientation_sha256"]
    ).reset_index(drop=True)
    if pool.source_orientation_sha256.duplicated().any():
        raise RuntimeError("Duplicate source orientation identity")
    if set(pool.label.unique()) != {0}:
        raise RuntimeError("Source pool contains a non-negative label")
    if (set(map(int, pool.id1)) | set(map(int, pool.id2))) & val_ids:
        raise RuntimeError("Validation item leaked into source pool")

    counts = pool.category.value_counts().sort_index()
    capacity = {
        category: {
            "human_negative_pairs": int(counts[category] // 2),
            "source_orientations": int(counts[category]),
            "max_candidates_at_two_per_source": int(counts[category] * MAX_CANDIDATES_PER_SOURCE),
            "target_accepted": int(args.target_per_category),
            "capacity_margin": int(counts[category] * MAX_CANDIDATES_PER_SOURCE - args.target_per_category),
        }
        for category in categories
    }
    insufficient = [
        category
        for category, values in capacity.items()
        if values["max_candidates_at_two_per_source"] < args.target_per_category
    ]
    if insufficient:
        raise RuntimeError(f"Insufficient source capacity: {insufficient}")

    summary = {
        "status": "dry_run_ok" if args.dry_run else "complete",
        "pipeline": "a2_v4_all_categories_200k",
        "canonical_split_sha256": split_sha,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_validation_item_overlap": 0,
        "train_validation_unordered_pair_overlap": 0,
        "human_negative_unordered_pairs": len(negatives),
        "duplicate_human_negative_rows_removed": duplicates_before,
        "source_orientations": len(pool),
        "category_count": len(categories),
        "categories": categories,
        "target_per_category": int(args.target_per_category),
        "target_total": int(args.target_per_category * len(categories)),
        "max_candidates_per_source": MAX_CANDIDATES_PER_SOURCE,
        "capacity_by_category": capacity,
        "validation_content_written": False,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "source_orientations.parquet"
    pool.to_parquet(pool_path, index=False)
    summary["source_pool_sha256"] = file_sha256(pool_path)
    summary["source_pool_bytes"] = pool_path.stat().st_size
    (args.output_dir / "source_pool_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
