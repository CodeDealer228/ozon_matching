"""Strictly verify the versioned A2 few-shot pack against its train source pool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_helpers():
    path = REPO_ROOT / "scripts" / "analyze_synthetic_negative_rules.py"
    spec = importlib.util.spec_from_file_location("synthetic_negative_rules_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokens(value: object, helpers) -> set[str]:
    return set(re.findall(r"[0-9a-zа-я]+", helpers.normalize_text(value)))


def numeric_tokens(value: object, helpers) -> set[str]:
    return {token for token in tokens(value, helpers) if any(char.isdigit() for char in token)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=REPO_ROOT / "synthetic_research" / "a2_few_shot_v1.json",
    )
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "source_pools" / "a2_source_pool.parquet",
    )
    args = parser.parse_args()

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    pool_hash = file_sha256(args.source_pool)
    if pack["canonical_split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Few-shot pack has the wrong canonical split SHA-256")
    if pack["validation_fingerprint"] != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Few-shot pack has the wrong validation fingerprint")
    if pack["a2_source_pool_sha256"] != pool_hash:
        raise RuntimeError("Few-shot pack source-pool hash mismatch")
    examples = pack.get("examples", [])
    if not 3 <= len(examples) <= 10:
        raise RuntimeError(f"Expected 3-10 examples, got {len(examples)}")

    pool = pd.read_parquet(args.source_pool)
    helpers = load_helpers()
    results: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for index, example in enumerate(examples):
        key = (int(example["id1"]), int(example["id2"]))
        if key in seen_pairs:
            raise RuntimeError(f"Duplicate source pair in example {index}")
        seen_pairs.add(key)
        match = pool[pool["id1"].eq(key[0]) & pool["id2"].eq(key[1])]
        if len(match) != 1:
            raise RuntimeError(f"Example {index} is not exactly one A2 source-pool row")
        row = match.iloc[0]
        for field in ("category", "name1", "name2", "source_pair_sha256"):
            if str(example[field]) != str(row[field]):
                raise RuntimeError(f"Example {index} {field} differs from source pool")

        rewritten = str(example["rewritten_name2"]).strip()
        if not rewritten:
            raise RuntimeError(f"Example {index} has an empty rewrite")
        normalized_original = helpers.normalize_text(row["name2"])
        normalized_rewritten = helpers.normalize_text(rewritten)
        if normalized_rewritten == normalized_original:
            raise RuntimeError(f"Example {index} is only a trivial normalization")

        missing_numbers = numeric_tokens(row["name2"], helpers) - numeric_tokens(rewritten, helpers)
        if missing_numbers:
            raise RuntimeError(f"Example {index} loses numeric facts: {sorted(missing_numbers)}")
        copied_numbers = (
            numeric_tokens(row["name1"], helpers)
            - numeric_tokens(row["name2"], helpers)
        ) & numeric_tokens(rewritten, helpers)
        if copied_numbers:
            raise RuntimeError(f"Example {index} copies numeric facts from item1: {sorted(copied_numbers)}")

        original_tokens = tokens(row["name2"], helpers)
        rewritten_tokens = tokens(rewritten, helpers)
        added_tokens = {token for token in rewritten_tokens - original_tokens if len(token) >= 3}
        item2_evidence = tokens(row["attributes2"], helpers) | original_tokens
        unsupported_added = added_tokens - item2_evidence
        if unsupported_added:
            raise RuntimeError(
                f"Example {index} adds facts absent from item2 name/attributes: {sorted(unsupported_added)}"
            )

        old_similarity = helpers.name_similarity(row["name1"], row["name2"])[0]
        new_similarity = helpers.name_similarity(row["name1"], rewritten)[0]
        if new_similarity <= old_similarity:
            raise RuntimeError(
                f"Example {index} is not harder by name similarity: old={old_similarity} new={new_similarity}"
            )
        results.append(
            {
                "index": index,
                "category": row["category"],
                "source_pair_sha256": row["source_pair_sha256"],
                "old_name_similarity": old_similarity,
                "new_name_similarity": new_similarity,
                "name_similarity_delta": new_similarity - old_similarity,
                "added_tokens_confirmed_in_item2": sorted(added_tokens),
            }
        )

    summary = {
        "status": "verified",
        "pack_path": str(args.pack.resolve()),
        "pack_sha256": file_sha256(args.pack),
        "source_pool_path": str(args.source_pool.resolve()),
        "source_pool_sha256": pool_hash,
        "n_examples": len(examples),
        "categories": sorted({str(value["category"]) for value in results}),
        "mean_name_similarity_delta": sum(float(value["name_similarity_delta"]) for value in results) / len(results),
        "examples": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
