#!/usr/bin/env python3
"""Build the immutable paired C1/T1 replay manifest for the BERTA A2 test.

The historical BERTA split is row based.  A2 source membership is therefore
resolved with an *unordered* item-pair key so that a reversed (id2, id1)
orientation cannot leak a validation source into training.

The output is a single paired manifest.  C1 and T1 consume the same rows,
weights and order; only replay rows choose a different item-2 representation:

* C1: the original item identified by ``id2``;
* T1: ``name2_prime`` identified by the collision-free ``synthetic_key``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DEFAULT_SYNTHETIC = Path("data/a2_synthetic_all_categories_v4.parquet")
DEFAULT_SPLIT = Path("data/splits/matches_split_seed42.parquet")
DEFAULT_OUTPUT = Path("artifacts/synthetic_v2/berta_a2_training_v1")
GLOBAL_BATCH = 512
REPLAY_FACTOR = 4
REPLAY_WEIGHT = 0.5
ORDER_SEED = 420031
EXPECTED_SYNTHETIC_SHA256 = (
    "c4c0242fd1ae273d3f13c7358d639a503f9b4e2d4efed200d1d6e05d8521ba00"
)
EXPECTED_SPLIT_ROWS = 365_654
EXPECTED_TRAIN_ROWS = 292_523
EXPECTED_VAL_ROWS = 73_131
EXPECTED_CLEAN_A2_ROWS = 10_123
EXPECTED_LEAK_A2_ROWS = 2_710


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_synthetic_key(source_orientation_sha256: str, candidate_index: int) -> str:
    payload = f"{source_orientation_sha256}:{int(candidate_index)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unordered_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.min_horizontal("id1", "id2").cast(pl.Int64).alias("unordered_id_low"),
        pl.max_horizontal("id1", "id2").cast(pl.Int64).alias("unordered_id_high"),
    )


def _write_parquet_compat(frame: pl.DataFrame, path: Path) -> None:
    # Disabling page statistics avoids the repetition-level histogram issue
    # seen when this project's native-Polars parquet is opened by pyarrow 19.
    frame.write_parquet(path, compression="zstd", statistics=False)


def build_manifests(
    synthetic_path: Path,
    split_path: Path,
    output_dir: Path,
    *,
    strict_expected_counts: bool = True,
) -> dict[str, Any]:
    synthetic_path = synthetic_path.resolve()
    split_path = split_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_sha = sha256_file(synthetic_path)
    split_sha = sha256_file(split_path)
    if strict_expected_counts and synthetic_sha != EXPECTED_SYNTHETIC_SHA256:
        raise RuntimeError(
            f"Unexpected synthetic SHA-256: {synthetic_sha}; "
            f"expected {EXPECTED_SYNTHETIC_SHA256}"
        )

    synthetic = pl.read_parquet(synthetic_path)
    split = pl.read_parquet(split_path)

    required_synth = {
        "source_orientation_sha256",
        "source_pair_sha256",
        "category",
        "model",
        "candidate_index",
        "name2_prime",
        "id1",
        "id2",
        "name1",
        "name2",
    }
    required_split = {"id1", "id2", "label", "split"}
    if missing := required_synth.difference(synthetic.columns):
        raise RuntimeError(f"Synthetic columns missing: {sorted(missing)}")
    if missing := required_split.difference(split.columns):
        raise RuntimeError(f"Split columns missing: {sorted(missing)}")

    if synthetic.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise RuntimeError("Synthetic dataset contains nulls")
    if split.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise RuntimeError("Split dataset contains nulls")

    split = _unordered_keys(split)
    duplicate_split_keys = (
        split.group_by(["unordered_id_low", "unordered_id_high"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_split_keys.height:
        raise RuntimeError(
            f"Historical split has {duplicate_split_keys.height} duplicate unordered pairs"
        )

    synthetic = _unordered_keys(synthetic)
    joined = synthetic.join(
        split.select(
            ["unordered_id_low", "unordered_id_high", "label", "split"]
        ),
        on=["unordered_id_low", "unordered_id_high"],
        how="left",
        validate="m:1",
    )
    unmatched = joined.filter(pl.col("split").is_null()).height
    if unmatched:
        raise RuntimeError(f"{unmatched} synthetic rows do not map to the historical split")
    if joined.filter(pl.col("label") != 0).height:
        raise RuntimeError("A2 contains a source whose historical label is not negative")

    clean = joined.filter(pl.col("split") == "train")
    leaked = joined.filter(pl.col("split") == "val")
    if strict_expected_counts:
        expected = {
            "split_rows": EXPECTED_SPLIT_ROWS,
            "train_rows": EXPECTED_TRAIN_ROWS,
            "val_rows": EXPECTED_VAL_ROWS,
            "clean_a2_rows": EXPECTED_CLEAN_A2_ROWS,
            "leaked_a2_rows": EXPECTED_LEAK_A2_ROWS,
        }
        actual = {
            "split_rows": split.height,
            "train_rows": split.filter(pl.col("split") == "train").height,
            "val_rows": split.filter(pl.col("split") == "val").height,
            "clean_a2_rows": clean.height,
            "leaked_a2_rows": leaked.height,
        }
        if actual != expected:
            raise RuntimeError(f"Count identity mismatch: actual={actual}, expected={expected}")

    clean = (
        clean.with_columns(
            pl.struct(["source_orientation_sha256", "candidate_index"])
            .map_elements(
                lambda row: stable_synthetic_key(
                    row["source_orientation_sha256"], row["candidate_index"]
                ),
                return_dtype=pl.String,
            )
            .alias("synthetic_key")
        )
        .sort(["source_orientation_sha256", "candidate_index"])
        .select(
            [
                "synthetic_key",
                "source_orientation_sha256",
                "source_pair_sha256",
                "category",
                "model",
                "candidate_index",
                "id1",
                "id2",
                "name1",
                "name2",
                "name2_prime",
                "unordered_id_low",
                "unordered_id_high",
            ]
        )
    )
    if clean.select(pl.col("synthetic_key").n_unique()).item() != clean.height:
        raise RuntimeError("synthetic_key is not unique")

    human_train = split.filter(pl.col("split") == "train").select(
        ["id1", "id2", pl.col("label").cast(pl.Int8), "unordered_id_low", "unordered_id_high"]
    )
    val = split.filter(pl.col("split") == "val").select(
        ["id1", "id2", pl.col("label").cast(pl.Int8), "unordered_id_low", "unordered_id_high"]
    )

    train_pairs = set(zip(human_train["unordered_id_low"].to_list(), human_train["unordered_id_high"].to_list()))
    val_pairs = set(zip(val["unordered_id_low"].to_list(), val["unordered_id_high"].to_list()))
    overlap = len(train_pairs.intersection(val_pairs))
    if overlap:
        raise RuntimeError(f"Historical train/val unordered overlap is {overlap}, expected 0")

    human_rows = human_train.with_row_index("source_row_index").with_columns(
        (pl.lit("human:") + pl.col("source_row_index").cast(pl.String).str.pad_start(6, "0")).alias("row_id"),
        pl.lit("human").alias("row_kind"),
        pl.lit(1.0).cast(pl.Float32).alias("sample_weight"),
        pl.lit(None).cast(pl.String).alias("synthetic_key"),
        pl.lit(None).cast(pl.String).alias("name2_prime"),
        pl.lit(-1).cast(pl.Int16).alias("replay_index"),
    )

    replay_frames: list[pl.DataFrame] = []
    replay_base = clean.select(["synthetic_key", "id1", "id2", "name2_prime"])
    for replay_index in range(REPLAY_FACTOR):
        replay_frames.append(
            replay_base.with_columns(
                (
                    pl.lit("replay:")
                    + pl.col("synthetic_key")
                    + pl.lit(f":{replay_index}")
                ).alias("row_id"),
                pl.lit("replay").alias("row_kind"),
                pl.lit(0).cast(pl.Int8).alias("label"),
                pl.lit(REPLAY_WEIGHT).cast(pl.Float32).alias("sample_weight"),
                pl.lit(replay_index).cast(pl.Int16).alias("replay_index"),
                pl.min_horizontal("id1", "id2").cast(pl.Int64).alias("unordered_id_low"),
                pl.max_horizontal("id1", "id2").cast(pl.Int64).alias("unordered_id_high"),
            )
        )
    replay_rows = pl.concat(replay_frames, how="diagonal_relaxed")

    base_columns = [
        "row_id",
        "row_kind",
        "id1",
        "id2",
        "label",
        "sample_weight",
        "synthetic_key",
        "name2_prime",
        "replay_index",
        "unordered_id_low",
        "unordered_id_high",
    ]
    paired = pl.concat(
        [human_rows.select(base_columns), replay_rows.select(base_columns)],
        how="vertical",
    )

    pad_rows = (-paired.height) % GLOBAL_BATCH
    if pad_rows:
        rng_pad = np.random.default_rng(ORDER_SEED + 1)
        pad_indices = rng_pad.choice(
            human_rows.height,
            size=pad_rows,
            replace=pad_rows > human_rows.height,
        )
        pad = human_rows[pad_indices.tolist()].select(base_columns).with_columns(
            (pl.lit("pad:") + pl.int_range(0, pad_rows, eager=True).cast(pl.String).str.pad_start(4, "0")).alias("row_id"),
            pl.lit("pad").alias("row_kind"),
        )
        paired = pl.concat([paired, pad], how="vertical")

    rng_order = np.random.default_rng(ORDER_SEED)
    permutation = rng_order.permutation(paired.height)
    paired = paired[permutation.tolist()].with_row_index("global_row_index")
    paired = paired.with_columns(
        (pl.col("global_row_index") // GLOBAL_BATCH).cast(pl.Int32).alias("global_step_index"),
        (pl.col("global_row_index") % GLOBAL_BATCH).cast(pl.Int16).alias("position_in_global_batch"),
    )

    if paired.height % GLOBAL_BATCH:
        raise RuntimeError("Paired manifest is not divisible by global batch")
    if paired.select(pl.col("row_id").n_unique()).item() != paired.height:
        raise RuntimeError("row_id is not unique after padding")
    if paired.filter(pl.col("row_kind") == "replay").height != clean.height * REPLAY_FACTOR:
        raise RuntimeError("Replay multiplicity mismatch")
    if paired.filter(pl.col("row_kind") == "replay").filter(pl.col("sample_weight") != REPLAY_WEIGHT).height:
        raise RuntimeError("Replay weight mismatch")

    clean_path = output_dir / "a2_train_clean.parquet"
    paired_path = output_dir / "paired_train_manifest.parquet"
    val_path = output_dir / "historical_val_manifest.parquet"
    leaked_path = output_dir / "excluded_validation_sources.parquet"
    _write_parquet_compat(clean, clean_path)
    _write_parquet_compat(paired, paired_path)
    _write_parquet_compat(val, val_path)
    _write_parquet_compat(
        leaked.select(
            [
                "source_orientation_sha256",
                "source_pair_sha256",
                "candidate_index",
                "id1",
                "id2",
                "unordered_id_low",
                "unordered_id_high",
            ]
        ),
        leaked_path,
    )

    category_counts = clean.group_by("category").len().sort("category").to_dicts()
    manifest: dict[str, Any] = {
        "status": "complete",
        "contract_version": 1,
        "synthetic_input": str(synthetic_path),
        "synthetic_sha256": synthetic_sha,
        "split_input": str(split_path),
        "split_sha256": split_sha,
        "global_batch": GLOBAL_BATCH,
        "replay_factor": REPLAY_FACTOR,
        "replay_weight": REPLAY_WEIGHT,
        "order_seed": ORDER_SEED,
        "historical_train_rows": human_train.height,
        "historical_val_rows": val.height,
        "clean_a2_rows": clean.height,
        "excluded_validation_a2_rows": leaked.height,
        "clean_a2_unique_source_orientations": clean.select(
            pl.col("source_orientation_sha256").n_unique()
        ).item(),
        "clean_a2_unique_unordered_pairs": clean.select(
            pl.struct(["unordered_id_low", "unordered_id_high"]).n_unique()
        ).item(),
        "paired_rows_before_padding": human_train.height + clean.height * REPLAY_FACTOR,
        "pad_rows": pad_rows,
        "paired_rows": paired.height,
        "global_steps": paired.height // GLOBAL_BATCH,
        "train_val_unordered_overlap": overlap,
        "category_counts": category_counts,
        "files": {},
    }
    for path in (clean_path, paired_path, val_path, leaked_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path = output_dir / "replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-strict-expected-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifests(
        args.synthetic,
        args.split,
        args.output_dir,
        strict_expected_counts=not args.no_strict_expected_counts,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
