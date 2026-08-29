"""Build matched human-control/synthetic-treatment manifests for BERTA C1/T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


GLOBAL_BATCH = 512
REPLAY_FACTOR = 4
REPLAY_WEIGHT = 0.5
ORDER_SEED = 420032


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    synthetic_dir: Path, split_path: Path, items_path: Path, output_dir: Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_pairs = pd.read_parquet(synthetic_dir / "synthetic_pairs.parquet")
    synthetic_items = pd.read_parquet(synthetic_dir / "synthetic_items.parquet")
    split = pd.read_parquet(split_path)
    items = pd.read_parquet(items_path, columns=["id", "category"])
    category_by_id = dict(zip(items["id"].astype(np.int64), items["category"].astype(str)))
    if "target" not in split.columns:
        raise RuntimeError("canonical_target_column_missing")
    split["label"] = split["target"]
    train = split[split["split"].eq("train")].copy()
    val = split[split["split"].eq("val")].copy()
    for frame in (train, val):
        frame["label"] = frame["label"].astype(np.int8)
        frame["category"] = frame["category"].astype(str)

    validation_ids = set(val["id1"].astype(np.int64)) | set(val["id2"].astype(np.int64))
    source_ids = set(synthetic_pairs["source_id1"].astype(np.int64)) | set(
        synthetic_pairs["source_id2"].astype(np.int64)
    )
    if source_ids & validation_ids:
        raise RuntimeError("synthetic_source_validation_overlap")
    synthetic_ids = set(synthetic_items["id"].astype(np.int64))
    referenced_synthetic = (
        set(synthetic_pairs["id1"].astype(np.int64))
        | set(synthetic_pairs["id2"].astype(np.int64))
    ) & synthetic_ids
    if referenced_synthetic != synthetic_ids:
        raise RuntimeError("synthetic_item_reference_identity_mismatch")
    if len(synthetic_pairs) != 12_000 or synthetic_pairs["operation_sha256"].nunique() != 6_000:
        raise RuntimeError("synthetic_count_identity_mismatch")
    if synthetic_pairs["target"].value_counts().to_dict() != {1: 6000, 0: 6000}:
        raise RuntimeError("synthetic_label_balance_mismatch")

    pools: dict[tuple[str, int], np.ndarray] = {}
    cursors: defaultdict[tuple[str, int], int] = defaultdict(int)
    rng = np.random.default_rng(ORDER_SEED)
    for key, group in train.groupby(["category", "label"], sort=True):
        values = group.index.to_numpy(np.int64)
        pools[(str(key[0]), int(key[1]))] = rng.permutation(values)

    human = pd.DataFrame({
        "row_id": [f"human:{index:06d}" for index in range(len(train))],
        "row_kind": "human",
        "label": train["label"].to_numpy(np.int8),
        "sample_weight": np.float32(1.0),
        "c1_id1": train["id1"].to_numpy(np.int64),
        "c1_id2": train["id2"].to_numpy(np.int64),
        "t1_id1": train["id1"].to_numpy(np.int64),
        "t1_id2": train["id2"].to_numpy(np.int64),
        "t1_syn1": False,
        "t1_syn2": False,
        "replay_index": np.int16(-1),
    })

    replay_rows = []
    for replay_index in range(REPLAY_FACTOR):
        for row in synthetic_pairs.itertuples(index=False):
            category = category_by_id[int(row.source_id1)]
            label = int(row.target)
            key = (category, label)
            pool = pools.get(key)
            if pool is None or not len(pool):
                raise RuntimeError(f"missing_control_pool:{key}")
            control = train.loc[int(pool[cursors[key] % len(pool)])]
            cursors[key] += 1
            replay_rows.append({
                "row_id": f"replay:{row.operation_sha256}:{row.variant}:{replay_index}",
                "row_kind": "replay",
                "label": label,
                "sample_weight": np.float32(REPLAY_WEIGHT),
                "c1_id1": int(control.id1),
                "c1_id2": int(control.id2),
                "t1_id1": int(row.id1),
                "t1_id2": int(row.id2),
                "t1_syn1": int(row.id1) in synthetic_ids,
                "t1_syn2": int(row.id2) in synthetic_ids,
                "replay_index": np.int16(replay_index),
            })
    replay = pd.DataFrame(replay_rows)
    paired = pd.concat([human, replay], ignore_index=True)
    before_padding = len(paired)
    pad_rows = (-before_padding) % GLOBAL_BATCH
    if pad_rows:
        pad = human.sample(pad_rows, replace=False, random_state=ORDER_SEED + 1).copy()
        pad["row_id"] = [f"pad:{index:04d}" for index in range(pad_rows)]
        pad["row_kind"] = "pad"
        paired = pd.concat([paired, pad], ignore_index=True)
    permutation = np.random.default_rng(ORDER_SEED).permutation(len(paired))
    paired = paired.iloc[permutation].reset_index(drop=True)
    paired["global_row_index"] = np.arange(len(paired), dtype=np.int64)
    paired["global_step_index"] = (paired["global_row_index"] // GLOBAL_BATCH).astype(np.int32)
    paired["position_in_global_batch"] = (paired["global_row_index"] % GLOBAL_BATCH).astype(np.int16)
    if paired["row_id"].duplicated().any() or len(paired) % GLOBAL_BATCH:
        raise RuntimeError("paired_manifest_integrity_failed")

    val_manifest = val[["id1", "id2", "label", "category"]].reset_index(drop=True)
    paired_path = output_dir / "paired_train_manifest.parquet"
    val_path = output_dir / "historical_val_manifest.parquet"
    item_path = output_dir / "synthetic_items.parquet"
    pair_path = output_dir / "synthetic_pairs.parquet"
    paired.to_parquet(paired_path, index=False)
    val_manifest.to_parquet(val_path, index=False)
    synthetic_items.to_parquet(item_path, index=False)
    synthetic_pairs.to_parquet(pair_path, index=False)
    manifest = {
        "status": "complete",
        "contract_version": 2,
        "global_batch": GLOBAL_BATCH,
        "replay_factor": REPLAY_FACTOR,
        "replay_weight": REPLAY_WEIGHT,
        "order_seed": ORDER_SEED,
        "historical_train_rows": len(train),
        "historical_val_rows": len(val),
        "synthetic_pairs": len(synthetic_pairs),
        "synthetic_items": len(synthetic_items),
        "replay_rows": len(replay),
        "paired_rows_before_padding": before_padding,
        "pad_rows": pad_rows,
        "paired_rows": len(paired),
        "global_steps": len(paired) // GLOBAL_BATCH,
        "validation_source_overlap": 0,
        "files": {},
    }
    for path in (paired_path, val_path, item_path, pair_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size, "sha256": sha256_file(path)
        }
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/splits/matches_item_disjoint_70_30_seed42.parquet"))
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.synthetic_dir, args.split, args.items, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
