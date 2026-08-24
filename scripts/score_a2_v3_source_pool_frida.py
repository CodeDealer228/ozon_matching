"""Score the strict A2 v3 source pool before spending Qwen requests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from score_a2_v3_frozen_frida import (
    EXPECTED_CLASSIFIER_SHA256,
    EXPECTED_MODEL_SHA256,
    build_item_text,
    file_sha256,
    load_submission_utils,
    write_json,
)


EXPECTED_SOURCE_POOL_SHA256 = "67a111a5a81f2cbac0962ba2848d3b756427b47a1c773fb7e28d75162729f484"
EXPECTED_SOURCE_ROWS = 400


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=repo / "artifacts/synthetic_v2/a2_v3_candidates/a2_v3_source_pool.parquet",
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
        default=repo / "artifacts/synthetic_v2/a2_v3_source_frida_scores",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    import numpy as np
    import pandas as pd
    import torch

    if file_sha256(args.source_pool) != EXPECTED_SOURCE_POOL_SHA256:
        raise RuntimeError("Strict A2 v3 source-pool SHA-256 mismatch")
    source = pd.read_parquet(args.source_pool)
    if len(source) != EXPECTED_SOURCE_ROWS or set(source["label"].unique()) != {0}:
        raise RuntimeError("Strict A2 v3 source-pool identity mismatch")
    if source["source_pair_sha256"].duplicated().any():
        raise RuntimeError("Source pool contains duplicate oriented pairs")

    model_path = args.submission_root / "models/frida-full-sft/model.safetensors"
    classifier_path = args.submission_root / "classifier.pt"
    for required in (model_path, classifier_path, args.submission_root / "src/utils.py"):
        if not required.is_file():
            raise FileNotFoundError(required)
    model_sha256 = file_sha256(model_path)
    classifier_sha256 = file_sha256(classifier_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Model SHA-256 mismatch: {model_sha256}")
    if classifier_sha256 != EXPECTED_CLASSIFIER_SHA256:
        raise RuntimeError(f"Classifier SHA-256 mismatch: {classifier_sha256}")

    utils = load_submission_utils(args.submission_root)
    tokenizer = utils._load_tokenizer(args.submission_root / "models/frida-full-sft")
    item_texts: dict[int, str] = {}
    pair_rows: list[dict[str, int]] = []
    next_item_id = -1
    for row in source.itertuples(index=False):
        a_id, b_id = next_item_id, next_item_id - 1
        next_item_id -= 2
        item_texts[a_id] = build_item_text(
            utils, row.name1, row.category, row.attributes1
        )
        item_texts[b_id] = build_item_text(
            utils, row.name2, row.category, row.attributes2
        )
        pair_rows.append({"id1": a_id, "id2": b_id})

    id_to_item = utils._tokenize_items(tokenizer, item_texts)
    input_ids, attention_mask = utils._build_pairs(
        pd.DataFrame(pair_rows), id_to_item, tokenizer.pad_token_id
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

    result = source.copy()
    result["old_frida_score"] = probabilities
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "source_scores.parquet"
    result.to_parquet(scores_path, index=False)

    by_category = {}
    for category, group in result.groupby("category", sort=True):
        values = group["old_frida_score"].to_numpy(dtype=np.float64)
        by_category[str(category)] = {
            "count": int(len(values)),
            "minimum": float(values.min()),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        }
    values = result["old_frida_score"].to_numpy(dtype=np.float64)
    summary = {
        "status": "complete",
        "profile": "a2_v3_source_pool_frozen_frida_scoring",
        "source_pool_sha256": file_sha256(args.source_pool),
        "source_rows": int(len(result)),
        "model_sha256": model_sha256,
        "classifier_sha256": classifier_sha256,
        "old_cache_used": False,
        "batch_size": args.batch_size,
        "overall": {
            "minimum": float(values.min()),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        },
        "by_category": by_category,
        "scores_sha256": file_sha256(scores_path),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "source_score_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
