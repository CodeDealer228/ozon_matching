"""Validate the current submitted FRIDA full-SFT model on the canonical split.

This is deliberately validation-only.  It downloads (or finds) the exact
submitted archive, imports that archive's own preprocessing implementation,
and never touches any historical FRIDA token/feature cache.

Persistent outputs are written incrementally to /kaggle/working so a hard
session cap still leaves honest partial prediction shards and JSON state.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path


SUBMISSION_URL = (
    "https://storage.yandexcloud.net/ds-ods/files/submissions/"
    "0d0183cf-8864-4331-88be-f78da0c68dd2/8d18ca15/"
    "matching-frida_continue_c_llm3m_188-submit.zip"
)
SUBMISSION_ZIP_NAME = "matching-frida_continue_c_llm3m_188-submit.zip"
EXPECTED_SUBMISSION_SIZE = 1_309_768_430
EXPECTED_SUBMISSION_SHA256 = "30bfc85d44785d44f9316a0b4dda2cb7dfdb7c51e3d3e68c89eb9a5ef064b9a1"

SPLIT_NAME = "matches_item_disjoint_70_30_seed42.parquet"
EXPECTED_SPLIT_SIZE = 6_192_178
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
EXPECTED_N_PAIRS = 365_654
EXPECTED_N_TRAIN = 256_006
EXPECTED_N_VAL = 109_648
EXPECTED_SEED = 42

OUTPUT_DIR = Path("/kaggle/working")
TEMP_DIR = Path("/kaggle/temp/frida-full-sft-error-analysis")
INPUT_ROOT = Path("/kaggle/input")

# Forward-only batch 72/core at seq=192 is already used by the project's
# full-SFT TPU kernel.  This validator is slightly shorter (seq=188), has no
# optimizer/gradients, and deliberately keeps that measured conservative value.
TPU_PER_CORE_BATCH = 72
ERROR_CANDIDATES_PER_CLASS_PER_CATEGORY = 200
CLASSIFICATION_THRESHOLD = 0.5
MAX_RUNTIME_SECONDS = 8.0 * 60 * 60


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    os.replace(tmp, path)


def update_status(phase: str, **extra) -> None:
    payload = {
        "phase": phase,
        "updated_at_unix": time.time(),
        **extra,
    }
    write_json(OUTPUT_DIR / "run_status.json", payload)
    print(f"[STATUS] {phase}: {extra}", flush=True)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_input_file(filename: str) -> Path | None:
    if not INPUT_ROOT.exists():
        return None
    matches = sorted(INPUT_ROOT.rglob(filename))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[INPUT] {filename}: found {len(matches)} copies; using {matches[0]}", flush=True)
    return matches[0]


def require_input_file(filename: str) -> Path:
    path = find_input_file(filename)
    if path is None:
        raise FileNotFoundError(f"Required Kaggle input file not found: {filename}")
    return path


def download_exact_submission() -> Path:
    """Prefer a mounted byte-identical archive, otherwise download to temp."""
    mounted = find_input_file(SUBMISSION_ZIP_NAME)
    if mounted is not None:
        size = mounted.stat().st_size
        digest = sha256_file(mounted)
        if size == EXPECTED_SUBMISSION_SIZE and digest == EXPECTED_SUBMISSION_SHA256:
            print(f"[MODEL] using mounted exact archive: {mounted}", flush=True)
            return mounted
        raise RuntimeError(
            f"Mounted {SUBMISSION_ZIP_NAME} identity mismatch: size={size}, sha256={digest}"
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    destination = TEMP_DIR / SUBMISSION_ZIP_NAME
    if destination.exists():
        if (
            destination.stat().st_size == EXPECTED_SUBMISSION_SIZE
            and sha256_file(destination) == EXPECTED_SUBMISSION_SHA256
        ):
            return destination
        destination.unlink()

    free_bytes = shutil.disk_usage(TEMP_DIR).free
    required_bytes = EXPECTED_SUBMISSION_SIZE + 1_659_496_120 + 2_000_000_000
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient temp disk: free={free_bytes:,}, required={required_bytes:,}"
        )

    request = urllib.request.Request(SUBMISSION_URL, headers={"User-Agent": "frida-validator/1.0"})
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = 0
    last_log = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        reported = response.headers.get("Content-Length")
        if reported is not None and int(reported) != EXPECTED_SUBMISSION_SIZE:
            raise RuntimeError(
                f"Remote Content-Length mismatch: {reported} != {EXPECTED_SUBMISSION_SIZE}"
            )
        while True:
            chunk = response.read(16 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_log >= 128 * 1024 * 1024:
                print(f"[MODEL] downloaded {downloaded:,}/{EXPECTED_SUBMISSION_SIZE:,}", flush=True)
                last_log = downloaded
    os.replace(partial, destination)

    size = destination.stat().st_size
    digest = sha256_file(destination)
    if size != EXPECTED_SUBMISSION_SIZE or digest != EXPECTED_SUBMISSION_SHA256:
        raise RuntimeError(f"Downloaded model identity mismatch: size={size}, sha256={digest}")
    return destination


def safe_extract_submission(archive: Path) -> tuple[Path, dict]:
    extract_dir = TEMP_DIR / "submission"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member in submission archive: {bad}")
        root = extract_dir.resolve()
        for member in bundle.infolist():
            target = (extract_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe zip path: {member.filename}")
        members = {
            "entry_count": len(bundle.infolist()),
            "uncompressed_bytes": sum(x.file_size for x in bundle.infolist()),
            "entries": [x.filename for x in bundle.infolist()],
        }
        bundle.extractall(extract_dir)

    model_files = list(extract_dir.rglob("model.safetensors"))
    if len(model_files) != 1:
        raise RuntimeError(f"Expected one model.safetensors, found {model_files}")
    base_model = model_files[0].parent
    submission_root = base_model.parent.parent
    classifier = submission_root / "classifier.pt"
    utils_file = submission_root / "src" / "utils.py"
    for required in (classifier, utils_file, base_model / "config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    file_identity = {}
    for path in sorted(p for p in submission_root.rglob("*") if p.is_file()):
        rel = path.relative_to(submission_root).as_posix()
        file_identity[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    identity = {
        "reference_url": SUBMISSION_URL,
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "zip": members,
        "files": file_identity,
    }
    return submission_root, identity


def load_submission_utils(submission_root: Path):
    root_str = str(submission_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    for name in ("src.utils", "src"):
        sys.modules.pop(name, None)
    return importlib.import_module("src.utils")


def verify_split(split_path: Path):
    import numpy as np
    import pandas as pd

    size = split_path.stat().st_size
    digest = sha256_file(split_path)
    if size != EXPECTED_SPLIT_SIZE or digest != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"Canonical split byte identity mismatch: size={size}, sha256={digest}"
        )

    frame = pd.read_parquet(split_path)
    required = {"id1", "id2", "label", "target", "category", "component", "split"}
    if set(frame.columns) != required:
        raise RuntimeError(f"Unexpected split columns: {frame.columns.tolist()}")

    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    if (len(frame), len(train), len(val)) != (EXPECTED_N_PAIRS, EXPECTED_N_TRAIN, EXPECTED_N_VAL):
        raise RuntimeError(
            f"Split counts mismatch: all/train/val={len(frame)}/{len(train)}/{len(val)}"
        )

    train_items = set(np.concatenate((train["id1"].to_numpy(), train["id2"].to_numpy())).tolist())
    val_items = set(np.concatenate((val["id1"].to_numpy(), val["id2"].to_numpy())).tolist())
    shared_items = train_items & val_items

    def unordered_pairs(part):
        a = part["id1"].to_numpy()
        b = part["id2"].to_numpy()
        return set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))

    pair_overlap = unordered_pairs(train) & unordered_pairs(val)
    val_fingerprint = hashlib.sha256(
        "\n".join(sorted(f"{a}-{b}" for a, b in zip(val["id1"], val["id2"]))).encode()
    ).hexdigest()
    label_matches_target = bool((frame["label"].astype(int) == frame["target"].astype(int)).all())
    binary_by_category = bool((val.groupby("category")["label"].nunique() == 2).all())

    identity = {
        "path": str(split_path),
        "bytes": size,
        "sha256": digest,
        "seed": EXPECTED_SEED,
        "n_pairs": len(frame),
        "n_train": len(train),
        "n_val": len(val),
        "n_train_items": len(train_items),
        "n_val_items": len(val_items),
        "shared_train_val_items": len(shared_items),
        "unordered_train_val_pair_overlap": len(pair_overlap),
        "val_fingerprint_sha256": val_fingerprint,
        "label_matches_binarized_target": label_matches_target,
        "validation_has_both_classes_in_every_category": binary_by_category,
        "verified": (
            not shared_items
            and not pair_overlap
            and val_fingerprint == EXPECTED_VAL_FINGERPRINT
            and label_matches_target
            and binary_by_category
        ),
    }
    if not identity["verified"]:
        raise RuntimeError(f"Canonical split semantic checks failed: {identity}")
    return frame, val.reset_index(drop=True), identity


_TPU_SHARED = {}


def _tpu_prediction_worker(_index: int) -> None:
    """Run one fixed-shape contiguous validation shard on one TPU core."""
    try:
        import numpy as np
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        from torch_xla import runtime as xr

        rank = xr.global_ordinal()
        world_size = xr.world_size()
        device = torch_xla.device()
        root = Path(_TPU_SHARED["submission_root"])
        input_ids = np.load(_TPU_SHARED["input_ids_path"], mmap_mode="r")
        attention_mask = np.load(_TPU_SHARED["attention_mask_path"], mmap_mode="r")
        n_rows = int(_TPU_SHARED["n_rows"])
        deadline_unix = float(_TPU_SHARED["deadline_unix"])

        per_rank_batches = max(1, (n_rows + TPU_PER_CORE_BATCH * world_size - 1)
                               // (TPU_PER_CORE_BATCH * world_size))
        per_rank_rows = per_rank_batches * TPU_PER_CORE_BATCH
        padded_total = per_rank_rows * world_size
        indices = np.arange(n_rows, dtype=np.int64)
        if padded_total > n_rows:
            indices = np.concatenate((indices, np.arange(padded_total - n_rows) % n_rows))
        start = rank * per_rank_rows
        end = (rank + 1) * per_rank_rows
        rank_indices = indices[start:end]
        rank_valid = np.arange(start, end) < n_rows

        utils = load_submission_utils(root)
        model = utils._load_model_weights(
            root / "models" / "frida-full-sft", root / "classifier.pt", device
        )
        model.eval()
        probabilities = np.empty(per_rank_rows, dtype=np.float32)

        with torch.no_grad():
            for batch_number, offset in enumerate(
                range(0, per_rank_rows, TPU_PER_CORE_BATCH), start=1
            ):
                if time.time() >= deadline_unix:
                    write_json(
                        OUTPUT_DIR / f"tpu_rank_{rank}_status.json",
                        {
                            "status": "deadline",
                            "rank": rank,
                            "completed_batches": batch_number - 1,
                            "total_batches": per_rank_batches,
                        },
                    )
                    return
                batch_indices = rank_indices[offset:offset + TPU_PER_CORE_BATCH]
                ids = torch.from_numpy(np.asarray(input_ids[batch_indices])).to(device)
                mask = torch.from_numpy(np.asarray(attention_mask[batch_indices])).to(device)
                logits = model(input_ids=ids, attention_mask=mask)
                batch_probabilities = torch.sigmoid(logits.float())
                xm.mark_step()
                probabilities[offset:offset + TPU_PER_CORE_BATCH] = (
                    batch_probabilities.cpu().numpy().astype(np.float32)
                )
                if batch_number == 1 or batch_number % 20 == 0 or batch_number == per_rank_batches:
                    write_json(
                        OUTPUT_DIR / f"tpu_rank_{rank}_status.json",
                        {
                            "status": "running",
                            "rank": rank,
                            "completed_batches": batch_number,
                            "total_batches": per_rank_batches,
                        },
                    )

        np.save(OUTPUT_DIR / f"tpu_rank_{rank}_probabilities.npy", probabilities)
        np.save(OUTPUT_DIR / f"tpu_rank_{rank}_indices.npy", rank_indices)
        np.save(OUTPUT_DIR / f"tpu_rank_{rank}_valid.npy", rank_valid)
        write_json(
            OUTPUT_DIR / f"tpu_rank_{rank}_status.json",
            {
                "status": "complete",
                "rank": rank,
                "completed_batches": per_rank_batches,
                "total_batches": per_rank_batches,
                "real_rows": int(rank_valid.sum()),
            },
        )
    except Exception as exc:  # noqa: BLE001 - persist exact worker failure
        rank = locals().get("rank", _index)
        write_json(
            OUTPUT_DIR / f"tpu_rank_{rank}_status.json",
            {
                "status": "error",
                "rank": rank,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run_tpu_inference(
    submission_root: Path,
    input_ids_path: Path,
    attention_mask_path: Path,
    n_rows: int,
    deadline_unix: float,
):
    import numpy as np
    import torch_xla

    # Kaggle pre-sets these for a different TPU topology.  They conflict
    # with torch_xla.launch(), which establishes its own 8-worker PJRT setup.
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _TPU_SHARED.update(
        submission_root=str(submission_root),
        input_ids_path=str(input_ids_path),
        attention_mask_path=str(attention_mask_path),
        n_rows=int(n_rows),
        deadline_unix=float(deadline_unix),
    )
    print(
        f"[INFERENCE] launching TPU workers: batch={TPU_PER_CORE_BATCH}/core, rows={n_rows:,}",
        flush=True,
    )
    torch_xla.launch(_tpu_prediction_worker, args=(), start_method="fork")

    # Discover the actual completed ranks from their persistent status
    # sidecars; the parent is outside the per-core runtime after launch.
    rank_statuses = sorted(OUTPUT_DIR.glob("tpu_rank_*_status.json"))
    if not rank_statuses:
        raise RuntimeError("TPU launch returned without any rank status files")
    ranks = []
    for status_path in rank_statuses:
        with status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("status") != "complete":
            raise RuntimeError(f"Incomplete TPU rank: {status_path.name}: {status}")
        ranks.append(int(status["rank"]))

    probabilities = np.empty(n_rows, dtype=np.float32)
    seen = np.zeros(n_rows, dtype=np.bool_)
    for rank in sorted(ranks):
        rank_probabilities = np.load(OUTPUT_DIR / f"tpu_rank_{rank}_probabilities.npy")
        rank_indices = np.load(OUTPUT_DIR / f"tpu_rank_{rank}_indices.npy")
        rank_valid = np.load(OUTPUT_DIR / f"tpu_rank_{rank}_valid.npy").astype(bool)
        real_indices = rank_indices[rank_valid]
        if seen[real_indices].any():
            raise RuntimeError(f"TPU rank {rank} overlaps an already written validation row")
        probabilities[real_indices] = rank_probabilities[rank_valid]
        seen[real_indices] = True
    if not seen.all():
        raise RuntimeError(f"TPU predictions missing {int((~seen).sum())} validation rows")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Predictions contain NaN or infinity")
    accelerator = {
        "type": "TPU",
        "ranks": sorted(ranks),
        "world_size": len(ranks),
        "per_core_batch": TPU_PER_CORE_BATCH,
    }
    return probabilities, accelerator


def compute_metrics(validation, probabilities):
    import numpy as np
    from sklearn.metrics import average_precision_score

    labels = validation["label"].to_numpy(dtype=np.int8)
    predicted = probabilities >= CLASSIFICATION_THRESHOLD
    rows = []
    for category, indices in validation.groupby("category", sort=True).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        truth = labels[idx]
        score = probabilities[idx]
        rows.append(
            {
                "category": str(category),
                "n_pairs": len(idx),
                "n_positive": int(truth.sum()),
                "n_negative": int(len(idx) - truth.sum()),
                "average_precision": float(average_precision_score(truth, score)),
                "tp_at_0_5": int(((truth == 1) & predicted[idx]).sum()),
                "fp_at_0_5": int(((truth == 0) & predicted[idx]).sum()),
                "tn_at_0_5": int(((truth == 0) & ~predicted[idx]).sum()),
                "fn_at_0_5": int(((truth == 1) & ~predicted[idx]).sum()),
                "mean_score": float(score.mean()),
            }
        )
    macro_ap = float(np.mean([row["average_precision"] for row in rows]))
    micro_ap = float(average_precision_score(labels, probabilities))
    rows.sort(key=lambda row: row["average_precision"])
    return rows, macro_ap, micro_ap


def _clean_value(value):
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return None
    try:
        if bool(value != value):
            return None
    except Exception:
        pass
    return value.item() if hasattr(value, "item") else value


def build_error_candidates(
    validation,
    probabilities,
    items_path: Path,
    tokenizer,
    utils,
    id_to_text,
    id_to_item,
    input_ids_all,
    attention_mask_all,
) -> dict:
    import numpy as np
    import pandas as pd

    ranked = validation.copy()
    ranked["score"] = probabilities
    selected_parts = []
    for _category, group in ranked.groupby("category", sort=True):
        negatives = group[group["label"] == 0].nlargest(
            ERROR_CANDIDATES_PER_CLASS_PER_CATEGORY, "score"
        ).copy()
        negatives["candidate_type"] = "high_score_negative"
        negatives["candidate_rank"] = range(1, len(negatives) + 1)
        positives = group[group["label"] == 1].nsmallest(
            ERROR_CANDIDATES_PER_CLASS_PER_CATEGORY, "score"
        ).copy()
        positives["candidate_type"] = "low_score_positive"
        positives["candidate_rank"] = range(1, len(positives) + 1)
        selected_parts.extend((negatives, positives))
    candidates = pd.concat(selected_parts).sort_index()

    ids = pd.unique(pd.concat((candidates["id1"], candidates["id2"]))).tolist()
    raw = pd.read_parquet(
        items_path,
        columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", ids)],
    )
    raw_by_id = {
        int(row.id): {
            "id": int(row.id),
            "name": _clean_value(row.name),
            "category": _clean_value(row.category),
            "attributes": _clean_value(row.attributes),
        }
        for row in raw.itertuples(index=False)
    }

    def item_char_diagnostics(item_id: int):
        item = raw_by_id[item_id]
        name = str(item["name"] or "")
        category = str(item["category"] or "")
        attributes = str(item["attributes"] or "")
        reordered = utils.reorder_attributes(attributes)
        assembled_before_final_cap = (
            f"Name: {name[:utils.MAX_NAME_CHARS]} "
            f"Category: {category[:utils.MAX_CATEGORY_CHARS]} "
            f"Attributes: {reordered[:utils.MAX_ATTR_CHARS]}"
        )
        return {
            "name_chars": len(name),
            "category_chars": len(category),
            "reordered_attribute_chars": len(reordered),
            "name_char_truncated": len(name) > utils.MAX_NAME_CHARS,
            "category_char_truncated": len(category) > utils.MAX_CATEGORY_CHARS,
            "attributes_char_truncated": len(reordered) > utils.MAX_ATTR_CHARS,
            "final_item_char_truncated": len(assembled_before_final_cap) > utils.MAX_ITEM_CHARS,
        }

    output_path = OUTPUT_DIR / "frida_error_candidates.jsonl"
    counts = {"high_score_negative": 0, "low_score_positive": 0}
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, row in candidates.iterrows():
            id1 = int(row["id1"])
            id2 = int(row["id2"])
            item1 = id_to_item[id1]
            item2 = id_to_item[id2]
            compact1 = utils._build_compact_ids(*item1)
            compact2 = utils._build_compact_ids(*item2)
            if len(compact1) + len(compact2) <= utils.PAIR_MAX_LENGTH:
                keep1, keep2 = len(compact1), len(compact2)
            else:
                keep1, keep2 = utils.allocate_pair_budget(
                    len(compact1), len(compact2), utils.PAIR_MAX_LENGTH
                )
            combined = np.concatenate((compact1[:keep1], compact2[:keep2])).astype(np.int64)
            observed = input_ids_all[row_index, : len(combined)]
            if not np.array_equal(combined, observed):
                raise RuntimeError(f"Exact-input reconstruction mismatch at validation row {row_index}")
            input_hash = hashlib.sha256(combined.astype("<i8", copy=False).tobytes()).hexdigest()
            candidate_type = str(row["candidate_type"])
            counts[candidate_type] += 1
            record = {
                "validation_row": int(row_index),
                "id1": id1,
                "id2": id2,
                "label": int(row["label"]),
                "category": str(row["category"]),
                "component": int(row["component"]),
                "score": float(row["score"]),
                "candidate_type": candidate_type,
                "candidate_rank_within_category": int(row["candidate_rank"]),
                "is_threshold_error_at_0_5": bool(
                    (int(row["label"]) == 0 and float(row["score"]) >= CLASSIFICATION_THRESHOLD)
                    or (int(row["label"]) == 1 and float(row["score"]) < CLASSIFICATION_THRESHOLD)
                ),
                "raw_item1": raw_by_id[id1],
                "raw_item2": raw_by_id[id2],
                "preprocessed_item1_text": id_to_text[id1],
                "preprocessed_item2_text": id_to_text[id2],
                "item1_char_diagnostics": item_char_diagnostics(id1),
                "item2_char_diagnostics": item_char_diagnostics(id2),
                "item1_compact_tokens": len(compact1),
                "item2_compact_tokens": len(compact2),
                "item1_kept_tokens": int(keep1),
                "item2_kept_tokens": int(keep2),
                "item1_pair_truncated": bool(keep1 < len(compact1)),
                "item2_pair_truncated": bool(keep2 < len(compact2)),
                "item1_item_token_budget_hit": bool(item1[1] == utils.ITEM_MAX_LENGTH),
                "item2_item_token_budget_hit": bool(item2[1] == utils.ITEM_MAX_LENGTH),
                "attention_tokens": int(attention_mask_all[row_index].sum()),
                "exact_item1_visible_input": tokenizer.decode(
                    compact1[:keep1].tolist(), skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "exact_item2_visible_input": tokenizer.decode(
                    compact2[:keep2].tolist(), skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "exact_pair_input_ids": combined.tolist(),
                "exact_pair_input_ids_sha256_i64le": input_hash,
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return {
        "selection": (
            f"top {ERROR_CANDIDATES_PER_CLASS_PER_CATEGORY} highest-scoring human negatives "
            "and lowest-scoring human positives per category"
        ),
        "threshold": CLASSIFICATION_THRESHOLD,
        "counts": counts,
        "total": sum(counts.values()),
        "path": str(output_path),
    }


def main() -> None:
    started = time.time()
    deadline = started + MAX_RUNTIME_SECONDS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    update_status("preflight")

    # Fail the teardown path cheaply before downloading/loading anything.
    write_json(OUTPUT_DIR / "write_probe.json", {"ok": True})
    (OUTPUT_DIR / "write_probe.json").unlink()

    import numpy as np
    import pandas as pd
    import pyarrow
    import sklearn
    import torch
    import torch_xla
    import transformers

    if not callable(getattr(torch_xla, "launch", None)):
        raise RuntimeError("torch_xla.launch is unavailable on this TPU image")

    package_versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "transformers": transformers.__version__,
    }
    write_json(
        OUTPUT_DIR / "preflight.json",
        {
            "packages": package_versions,
            "tpu_per_core_batch": TPU_PER_CORE_BATCH,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "output_free_bytes": shutil.disk_usage(OUTPUT_DIR).free,
            "temp_free_bytes": shutil.disk_usage(TEMP_DIR).free,
        },
    )

    split_path = require_input_file(SPLIT_NAME)
    items_path = require_input_file("items_human.parquet")
    full_split, validation, split_identity = verify_split(split_path)
    write_json(OUTPUT_DIR / "split_identity.json", split_identity)
    del full_split
    update_status("split_verified", **split_identity)

    archive = download_exact_submission()
    submission_root, model_identity = safe_extract_submission(archive)
    utils = load_submission_utils(submission_root)
    model_identity["inference_contract"] = {
        "max_name_chars": utils.MAX_NAME_CHARS,
        "max_category_chars": utils.MAX_CATEGORY_CHARS,
        "max_attr_chars": utils.MAX_ATTR_CHARS,
        "max_item_chars": utils.MAX_ITEM_CHARS,
        "prefix": utils.PREFIX,
        "item_max_length": utils.ITEM_MAX_LENGTH,
        "pair_max_length": utils.PAIR_MAX_LENGTH,
        "submitted_batch_size": utils.BATCH_SIZE_FIXED,
        "validation_batch_size_per_tpu_core": TPU_PER_CORE_BATCH,
        "precision": "torch.bfloat16 on TPU XLA (exact submitted weights and sigmoid-in-fp32)",
        "old_human_cache_used": False,
    }
    with (submission_root / "models" / "frida-full-sft" / "config.json").open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    model_identity["model_config"] = {
        key: config.get(key)
        for key in ("architectures", "d_model", "num_layers", "num_heads", "vocab_size", "torch_dtype")
    }
    write_json(OUTPUT_DIR / "model_identity.json", model_identity)
    update_status("model_verified", archive_sha256=model_identity["archive_sha256"])

    tokenizer = utils._load_tokenizer(submission_root / "models" / "frida-full-sft")
    needed_ids = pd.unique(pd.concat((validation["id1"], validation["id2"]))).tolist()
    id_to_text = utils._load_needed_item_texts(items_path, needed_ids)
    if len(id_to_text) != len(needed_ids):
        missing = sorted(set(needed_ids) - set(id_to_text))
        raise RuntimeError(f"items_human.parquet is missing {len(missing)} validation ids")
    id_to_item = utils._tokenize_items(tokenizer, id_to_text)
    input_ids_all, attention_mask_all = utils._build_pairs(
        validation, id_to_item, tokenizer.pad_token_id
    )
    if (attention_mask_all.sum(axis=1) == 0).any():
        raise RuntimeError("At least one validation pair has an all-zero attention mask")

    input_ids_path = TEMP_DIR / "validation_input_ids.npy"
    attention_mask_path = TEMP_DIR / "validation_attention_mask.npy"
    np.save(input_ids_path, input_ids_all)
    np.save(attention_mask_path, attention_mask_all)
    update_status(
        "exact_inputs_built",
        validation_pairs=len(validation),
        unique_items=len(id_to_item),
        pair_max_length=utils.PAIR_MAX_LENGTH,
    )

    probabilities, accelerator = run_tpu_inference(
        submission_root,
        input_ids_path,
        attention_mask_path,
        len(validation),
        deadline,
    )
    validation = validation.copy()
    validation["score"] = probabilities
    validation.to_parquet(OUTPUT_DIR / "val_predictions.parquet", index=False)
    np.save(OUTPUT_DIR / "val_predictions.npy", probabilities)
    update_status("predictions_complete", n_predictions=len(probabilities))

    per_category, macro_ap, micro_ap = compute_metrics(validation, probabilities)
    pd.DataFrame(per_category).to_csv(OUTPUT_DIR / "per_category_metrics.csv", index=False)
    metrics = {
        "metric": "unweighted mean of sklearn average_precision_score over categories",
        "macro_average_precision": macro_ap,
        "micro_average_precision": micro_ap,
        "classification_threshold_for_confusion_counts": CLASSIFICATION_THRESHOLD,
        "n_validation_pairs": len(validation),
        "n_categories": len(per_category),
        "suggested_target_categories_bottom_4": [row["category"] for row in per_category[:4]],
        "per_category": per_category,
    }
    write_json(OUTPUT_DIR / "baseline_metrics.json", metrics)
    update_status("metrics_complete", macro_ap=macro_ap, micro_ap=micro_ap)

    candidate_manifest = build_error_candidates(
        validation,
        probabilities,
        items_path,
        tokenizer,
        utils,
        id_to_text,
        id_to_item,
        input_ids_all,
        attention_mask_all,
    )
    write_json(OUTPUT_DIR / "error_candidate_manifest.json", candidate_manifest)

    artifact_names = [
        "preflight.json",
        "split_identity.json",
        "model_identity.json",
        "val_predictions.parquet",
        "val_predictions.npy",
        "per_category_metrics.csv",
        "baseline_metrics.json",
        "frida_error_candidates.jsonl",
        "error_candidate_manifest.json",
    ]
    artifact_identity = {
        name: {
            "bytes": (OUTPUT_DIR / name).stat().st_size,
            "sha256": sha256_file(OUTPUT_DIR / name),
        }
        for name in artifact_names
    }
    manifest = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "accelerator": accelerator,
        "packages": package_versions,
        "split_sha256": split_identity["sha256"],
        "val_fingerprint_sha256": split_identity["val_fingerprint_sha256"],
        "model_archive_sha256": model_identity["archive_sha256"],
        "macro_average_precision": macro_ap,
        "artifacts": artifact_identity,
    }
    write_json(OUTPUT_DIR / "run_manifest.json", manifest)
    update_status("complete", elapsed_seconds=manifest["elapsed_seconds"], macro_ap=macro_ap)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - persist the exact top-level failure
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        update_status("error", error=repr(exc), traceback=traceback.format_exc())
        raise
