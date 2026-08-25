from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.build_berta_a2_replay_manifests import (
    GLOBAL_BATCH,
    REPLAY_FACTOR,
    REPLAY_WEIGHT,
    build_manifests,
)


def test_unordered_membership_and_matched_contract(tmp_path: Path) -> None:
    split = pl.DataFrame(
        {
            "id1": [1, 3, 5, 7],
            "id2": [2, 4, 6, 8],
            "label": [0, 1, 0, 1],
            "split": ["train", "train", "val", "val"],
        }
    )
    synthetic = pl.DataFrame(
        {
            "source_orientation_sha256": ["a" * 64, "b" * 64],
            "source_pair_sha256": ["c" * 64, "d" * 64],
            "category": ["A", "A"],
            "model": ["qwen", "qwen"],
            "candidate_index": [0, 0],
            "name2_prime": ["synthetic train", "synthetic val"],
            # Both source orientations are reversed relative to split.
            "id1": [2, 6],
            "id2": [1, 5],
            "name1": ["a", "c"],
            "name2": ["b", "d"],
        }
    )
    split_path = tmp_path / "split.parquet"
    synthetic_path = tmp_path / "synthetic.parquet"
    output_dir = tmp_path / "output"
    split.write_parquet(split_path, statistics=False)
    synthetic.write_parquet(synthetic_path, statistics=False)

    manifest = build_manifests(
        synthetic_path,
        split_path,
        output_dir,
        strict_expected_counts=False,
    )

    assert manifest["clean_a2_rows"] == 1
    assert manifest["excluded_validation_a2_rows"] == 1
    assert manifest["train_val_unordered_overlap"] == 0
    assert manifest["paired_rows"] % GLOBAL_BATCH == 0

    paired = pl.read_parquet(output_dir / "paired_train_manifest.parquet")
    replay = paired.filter(pl.col("row_kind") == "replay")
    assert replay.height == REPLAY_FACTOR
    assert replay.select(pl.col("synthetic_key").n_unique()).item() == 1
    assert replay.select(pl.col("replay_index").n_unique()).item() == REPLAY_FACTOR
    assert replay.filter(pl.col("sample_weight") != REPLAY_WEIGHT).height == 0
    assert paired.select(pl.col("row_id").n_unique()).item() == paired.height
    assert paired["global_row_index"].to_list() == list(range(paired.height))


def test_positive_a2_source_is_rejected(tmp_path: Path) -> None:
    split = pl.DataFrame(
        {"id1": [1], "id2": [2], "label": [1], "split": ["train"]}
    )
    synthetic = pl.DataFrame(
        {
            "source_orientation_sha256": ["a" * 64],
            "source_pair_sha256": ["b" * 64],
            "category": ["A"],
            "model": ["qwen"],
            "candidate_index": [0],
            "name2_prime": ["x"],
            "id1": [1],
            "id2": [2],
            "name1": ["a"],
            "name2": ["b"],
        }
    )
    split_path = tmp_path / "split.parquet"
    synthetic_path = tmp_path / "synthetic.parquet"
    split.write_parquet(split_path, statistics=False)
    synthetic.write_parquet(synthetic_path, statistics=False)

    try:
        build_manifests(
            synthetic_path,
            split_path,
            tmp_path / "output",
            strict_expected_counts=False,
        )
    except RuntimeError as exc:
        assert "label is not negative" in str(exc)
    else:
        raise AssertionError("positive source must be rejected")
