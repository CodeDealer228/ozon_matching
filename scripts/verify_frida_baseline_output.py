"""Verify the persistent output of the FRIDA baseline validation kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N_VAL = 109_648
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
EXPECTED_MODEL_SHA256 = "30bfc85d44785d44f9316a0b4dda2cb7dfdb7c51e3d3e68c89eb9a5ef064b9a1"
REQUIRED_FILES = (
    "preflight.json",
    "split_identity.json",
    "model_identity.json",
    "val_predictions.parquet",
    "val_predictions.npy",
    "per_category_metrics.csv",
    "baseline_metrics.json",
    "frida_error_candidates.jsonl",
    "error_candidate_manifest.json",
    "run_manifest.json",
    "run_status.json",
)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("average precision is undefined without positive labels")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    cumulative_true = np.cumsum(sorted_labels)[group_ends]
    precision = cumulative_true / (group_ends + 1)
    recall = cumulative_true / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def import_validator(path: Path):
    spec = importlib.util.spec_from_file_location("frida_output_validator_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import validator source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(output_dir: Path, split_path: Path, kernel_source: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")
    empty = [name for name in REQUIRED_FILES if (output_dir / name).stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Empty required outputs: {empty}")

    run_status = read_json(output_dir / "run_status.json")
    run_manifest = read_json(output_dir / "run_manifest.json")
    split_identity = read_json(output_dir / "split_identity.json")
    model_identity = read_json(output_dir / "model_identity.json")
    metrics = read_json(output_dir / "baseline_metrics.json")
    candidate_manifest = read_json(output_dir / "error_candidate_manifest.json")

    if run_status.get("phase") != "complete" or run_manifest.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: status={run_status}, manifest={run_manifest}")
    if split_identity.get("sha256") != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Downloaded split identity has the wrong SHA-256")
    if split_identity.get("val_fingerprint_sha256") != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Downloaded split identity has the wrong validation fingerprint")
    if split_identity.get("shared_train_val_items") != 0:
        raise RuntimeError("Downloaded split identity reports train/validation item overlap")
    if split_identity.get("unordered_train_val_pair_overlap") != 0:
        raise RuntimeError("Downloaded split identity reports unordered pair overlap")
    if model_identity.get("archive_sha256") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Model archive identity does not match the submitted FRIDA archive")
    if model_identity.get("inference_contract", {}).get("old_human_cache_used") is not False:
        raise RuntimeError("Model identity does not prove that the old human cache was unused")

    source = import_validator(kernel_source)
    _full_split, canonical_validation, local_split_identity = source.verify_split(split_path)
    if local_split_identity["sha256"] != split_identity["sha256"]:
        raise RuntimeError("Local canonical split differs from the kernel split")
    if local_split_identity["val_fingerprint_sha256"] != split_identity["val_fingerprint_sha256"]:
        raise RuntimeError("Local validation fingerprint differs from the kernel fingerprint")

    probabilities = np.load(output_dir / "val_predictions.npy")
    if probabilities.shape != (EXPECTED_N_VAL,) or probabilities.dtype != np.float32:
        raise RuntimeError(f"Unexpected prediction array: shape={probabilities.shape}, dtype={probabilities.dtype}")
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise RuntimeError("Predictions contain non-finite or out-of-range values")

    predictions = pd.read_parquet(output_dir / "val_predictions.parquet")
    if len(predictions) != EXPECTED_N_VAL:
        raise RuntimeError(f"Parquet has {len(predictions)} rows instead of {EXPECTED_N_VAL}")
    required_columns = {"id1", "id2", "label", "target", "category", "score"}
    if not required_columns.issubset(predictions.columns):
        raise RuntimeError(f"Prediction parquet is missing columns: {sorted(required_columns - set(predictions.columns))}")
    if not np.array_equal(predictions["score"].to_numpy(dtype=np.float32), probabilities):
        raise RuntimeError("Parquet and NumPy prediction arrays are not bit-identical")
    for column in ("id1", "id2", "label", "target", "category"):
        if not predictions[column].reset_index(drop=True).equals(
            canonical_validation[column].reset_index(drop=True)
        ):
            raise RuntimeError(f"Prediction parquet column {column!r} differs from canonical validation")

    reconstructed = np.empty(EXPECTED_N_VAL, dtype=np.float32)
    seen = np.zeros(EXPECTED_N_VAL, dtype=np.bool_)
    rank_rows = 0
    for rank in range(8):
        status = read_json(output_dir / f"tpu_rank_{rank}_status.json")
        if status.get("status") != "complete":
            raise RuntimeError(f"TPU rank {rank} is incomplete: {status}")
        indices = np.load(output_dir / f"tpu_rank_{rank}_indices.npy")
        rank_probs = np.load(output_dir / f"tpu_rank_{rank}_probabilities.npy")
        valid = np.load(output_dir / f"tpu_rank_{rank}_valid.npy").astype(bool)
        real_indices = indices[valid]
        if seen[real_indices].any():
            raise RuntimeError(f"TPU rank {rank} overlaps an earlier rank")
        reconstructed[real_indices] = rank_probs[valid]
        seen[real_indices] = True
        rank_rows += int(valid.sum())
    if not seen.all() or rank_rows != EXPECTED_N_VAL:
        raise RuntimeError("TPU rank shards do not cover validation exactly once")
    if not np.array_equal(reconstructed, probabilities):
        raise RuntimeError("Reconstructed TPU shards differ from final predictions")

    labels = predictions["label"].to_numpy(dtype=np.int8)
    categories = predictions["category"].astype(str).to_numpy()
    per_category = pd.read_csv(output_dir / "per_category_metrics.csv")
    recomputed = []
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        recomputed.append((category, average_precision(labels[mask], probabilities[mask])))
    recomputed_frame = pd.DataFrame(recomputed, columns=["category", "average_precision"])
    metric_frame = per_category[["category", "average_precision"]].sort_values("category").reset_index(drop=True)
    recomputed_frame = recomputed_frame.sort_values("category").reset_index(drop=True)
    if not np.allclose(
        metric_frame["average_precision"].to_numpy(),
        recomputed_frame["average_precision"].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("Per-category metrics do not match predictions")
    macro_ap = float(recomputed_frame["average_precision"].mean())
    micro_ap = average_precision(labels, probabilities)
    if not np.isclose(macro_ap, metrics["macro_average_precision"], rtol=0, atol=1e-12):
        raise RuntimeError("Macro average precision does not match predictions")
    if not np.isclose(micro_ap, metrics["micro_average_precision"], rtol=0, atol=1e-12):
        raise RuntimeError("Micro average precision does not match predictions")

    with (output_dir / "frida_error_candidates.jsonl").open(encoding="utf-8") as handle:
        candidate_count = sum(1 for line in handle if line.strip())
    if candidate_count != int(candidate_manifest["total"]):
        raise RuntimeError("Error-candidate JSONL count differs from its manifest")

    manifest_hashes = run_manifest.get("artifacts", {})
    for name, identity in manifest_hashes.items():
        path = output_dir / name
        if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"Manifest identity mismatch for {name}")

    log_path = output_dir / "frida-full-sft-baseline-and-error-analysis.log"
    log_text = log_path.read_text(encoding="utf-8")
    if "[STATUS] complete" not in log_text:
        raise RuntimeError("Kernel log does not contain the final complete status")

    return {
        "status": "verified",
        "n_predictions": int(len(probabilities)),
        "prediction_min": float(probabilities.min()),
        "prediction_max": float(probabilities.max()),
        "macro_average_precision": macro_ap,
        "micro_average_precision": micro_ap,
        "n_categories": int(len(recomputed_frame)),
        "error_candidates": candidate_count,
        "split_sha256": split_identity["sha256"],
        "val_fingerprint_sha256": split_identity["val_fingerprint_sha256"],
        "shared_train_val_items": split_identity["shared_train_val_items"],
        "unordered_train_val_pair_overlap": split_identity["unordered_train_val_pair_overlap"],
        "model_archive_sha256": model_identity["archive_sha256"],
        "old_human_cache_used": model_identity["inference_contract"]["old_human_cache_used"],
        "output_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("split_path", type=Path)
    parser.add_argument("kernel_source", type=Path)
    args = parser.parse_args()
    report = verify(args.output_dir, args.split_path, args.kernel_source)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
