"""Score accepted A2 v3 candidates with the exact frozen submitted Frida.

The script builds the original pair ``(A, B)`` and its hardened counterpart
``(A, B')`` with the submission archive's own text/tokenization code.  It does
not use a historical feature or token cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


EXPECTED_MODEL_SHA256 = "423c9772de237fb9cc1d797b649dff805b547cb9bdf607ef0616c9ab1c7d0591"
EXPECTED_CLASSIFIER_SHA256 = "e661edb385e9ea846e8a8783faec1244e2294f99f56b9e090dd8b7649effb771"
EXPECTED_GENERATION_PROFILE = "a2_qwen_contract_v3"


def file_sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def load_submission_utils(submission_root: Path):
    utils_path = submission_root / "src" / "utils.py"
    spec = importlib.util.spec_from_file_location("a2_frozen_frida_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import submission utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_item_text(utils, name: str, category: str, attributes: str) -> str:
    name = str(name)[: utils.MAX_NAME_CHARS]
    category = str(category)[: utils.MAX_CATEGORY_CHARS]
    attributes = utils._reorder_and_slice(str(attributes))
    return (
        f"Name: {name} Category: {category} Attributes: {attributes}"
    )[: utils.MAX_ITEM_CHARS]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=repo / "artifacts/synthetic_v2/a2_v3_pilot_plan/pilot_source_pool.parquet",
    )
    parser.add_argument(
        "--candidate-results",
        type=Path,
        default=repo / "artifacts/synthetic_v2/a2_pilot_v3/candidate_results.parquet",
    )
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=repo / "artifacts/synthetic_v2/a2_pilot_v3/generation_manifest.json",
    )
    parser.add_argument(
        "--submission-root",
        type=Path,
        default=(
            repo
            / "artifacts/synthetic_v2/baseline_frida/extracted/"
            "matching-frida_continue_c_llm3m_188-submit"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "artifacts/synthetic_v2/a2_pilot_v3_frida_scores",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    import numpy as np
    import pandas as pd
    import torch

    with args.generation_manifest.open(encoding="utf-8") as handle:
        generation_manifest = json.load(handle)
    if generation_manifest.get("status") != "complete":
        raise RuntimeError("Generation manifest is not complete")
    if generation_manifest.get("profile") != EXPECTED_GENERATION_PROFILE:
        raise RuntimeError("Unexpected generation profile")
    if generation_manifest.get("uncertain_submissions") != 0:
        raise RuntimeError("Cannot score generation with uncertain submissions")

    model_path = args.submission_root / "models/frida-full-sft/model.safetensors"
    classifier_path = args.submission_root / "classifier.pt"
    utils_path = args.submission_root / "src/utils.py"
    for required in (model_path, classifier_path, utils_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    model_sha256 = file_sha256(model_path)
    classifier_sha256 = file_sha256(classifier_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Model SHA-256 mismatch: {model_sha256}")
    if classifier_sha256 != EXPECTED_CLASSIFIER_SHA256:
        raise RuntimeError(f"Classifier SHA-256 mismatch: {classifier_sha256}")

    source = pd.read_parquet(args.source_pool)
    candidates = pd.read_parquet(args.candidate_results)
    accepted = candidates[candidates["accepted_by_deterministic_validator"]].copy()
    if accepted.empty:
        raise RuntimeError("No deterministically accepted candidates to score")
    if accepted["source_pair_sha256"].duplicated().any():
        raise RuntimeError("Expected at most one accepted candidate per source pair")

    required_source_columns = {
        "source_pair_sha256",
        "category",
        "name1",
        "name2",
        "attributes1",
        "attributes2",
    }
    missing_columns = required_source_columns - set(source.columns)
    if missing_columns:
        raise RuntimeError(f"Source pool missing columns: {sorted(missing_columns)}")
    joined = accepted.merge(
        source[list(required_source_columns)],
        on=["source_pair_sha256", "category"],
        how="left",
        validate="one_to_one",
    )
    if joined["name1"].isna().any():
        raise RuntimeError("Accepted candidate is missing from the source pool")

    utils = load_submission_utils(args.submission_root)
    tokenizer = utils._load_tokenizer(args.submission_root / "models/frida-full-sft")

    item_texts: dict[int, str] = {}
    pair_rows: list[dict] = []
    row_identity: list[tuple[int, str]] = []
    next_item_id = -1
    for row_index, row in joined.reset_index(drop=True).iterrows():
        a_id, old_b_id, new_b_id = next_item_id, next_item_id - 1, next_item_id - 2
        next_item_id -= 3
        item_texts[a_id] = build_item_text(
            utils, row["name1"], row["category"], row["attributes1"]
        )
        item_texts[old_b_id] = build_item_text(
            utils, row["name2"], row["category"], row["attributes2"]
        )
        item_texts[new_b_id] = build_item_text(
            utils, row["name2_prime"], row["category"], row["attributes2"]
        )
        pair_rows.extend(({"id1": a_id, "id2": old_b_id}, {"id1": a_id, "id2": new_b_id}))
        row_identity.extend(((row_index, "old"), (row_index, "new")))

    id_to_item = utils._tokenize_items(tokenizer, item_texts)
    pair_frame = pd.DataFrame(pair_rows)
    input_ids, attention_mask = utils._build_pairs(
        pair_frame, id_to_item, tokenizer.pad_token_id
    )
    if (attention_mask.sum(axis=1) == 0).any():
        raise RuntimeError("Built an all-pad Frida input")

    device = torch.device("cpu")
    model = utils._load_model_weights(
        args.submission_root / "models/frida-full-sft", classifier_path, device
    )
    with torch.inference_mode():
        probabilities = utils._run_inference(
            model, input_ids, attention_mask, device, args.batch_size
        )

    scores: dict[int, dict[str, float]] = {}
    for probability, (row_index, variant) in zip(probabilities.tolist(), row_identity):
        scores.setdefault(row_index, {})[variant] = float(probability)

    rows = []
    for row_index, row in joined.reset_index(drop=True).iterrows():
        old_score = scores[row_index]["old"]
        new_score = scores[row_index]["new"]
        record = row.to_dict()
        record["old_frida_score"] = old_score
        record["new_frida_score"] = new_score
        record["hardening_delta"] = new_score - old_score
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("hardening_delta", ascending=False)

    del model
    del tokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "hardening_scores.parquet"
    result.to_parquet(scores_path, index=False)
    deltas = result["hardening_delta"].to_numpy(dtype=np.float64)
    useful = deltas > 0
    summary = {
        "status": "complete",
        "profile": "a2_v3_frozen_frida_scoring",
        "generation_manifest_sha256": file_sha256(args.generation_manifest),
        "candidate_results_sha256": file_sha256(args.candidate_results),
        "source_pool_sha256": file_sha256(args.source_pool),
        "model_sha256": model_sha256,
        "classifier_sha256": classifier_sha256,
        "old_cache_used": False,
        "accepted_candidates_scored": int(len(result)),
        "positive_hardening_count": int(useful.sum()),
        "positive_hardening_fraction": float(useful.mean()),
        "hardening_delta": {
            "minimum": float(deltas.min()),
            "p25": float(np.quantile(deltas, 0.25)),
            "median": float(np.median(deltas)),
            "p75": float(np.quantile(deltas, 0.75)),
            "maximum": float(deltas.max()),
            "mean": float(deltas.mean()),
        },
        "scores_sha256": file_sha256(scores_path),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "hardening_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
