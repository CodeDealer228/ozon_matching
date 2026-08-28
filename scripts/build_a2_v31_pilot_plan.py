"""Build a new, pre-scored and disjoint A2 v3.1 pilot plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from a2_fact_contract import build_fact_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCORED_POOL_SHA256 = "d1866fd145f158b5eaff4e200f261093292a98636fe139840e33c77de3267456"
EXPECTED_V3_PILOT_SHA256 = "084a79f816e2b47ace88b0dea74b9ead69c70c371603837f05cf6a63ceca2770"
CATEGORY_QUOTAS = {
    "Автотовары": 4,
    "Обувь": 10,
    "Одежда": 8,
    "Ювелирные изделия": 10,
}
SCORE_BINS = ((0.20, 0.30), (0.30, 0.45), (0.45, 0.60))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_bin(value: float) -> int | None:
    for index, (lower, upper) in enumerate(SCORE_BINS):
        if lower <= value < upper or (index == len(SCORE_BINS) - 1 and value == upper):
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored-source-pool",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts/synthetic_v2/a2_v3_source_frida_scores/source_scores.parquet"
        ),
    )
    parser.add_argument(
        "--exclude-v3-pilot",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts/synthetic_v2/a2_v3_pilot_plan/pilot_source_pool.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/a2_v31_pilot_plan",
    )
    args = parser.parse_args()

    if file_sha256(args.scored_source_pool) != EXPECTED_SCORED_POOL_SHA256:
        raise RuntimeError("Scored A2 v3 source-pool SHA-256 mismatch")
    if file_sha256(args.exclude_v3_pilot) != EXPECTED_V3_PILOT_SHA256:
        raise RuntimeError("A2 v3 pilot exclusion SHA-256 mismatch")

    pool = pd.read_parquet(args.scored_source_pool)
    previous = pd.read_parquet(args.exclude_v3_pilot)
    if len(pool) != 400 or set(pool["label"].unique()) != {0}:
        raise RuntimeError("Unexpected scored source-pool identity")
    excluded = set(map(str, previous["source_pair_sha256"]))
    if len(excluded) != 20:
        raise RuntimeError("Expected exactly 20 v3 pilot exclusions")

    contracts: dict[str, dict[str, object]] = {}
    candidates: list[dict[str, object]] = []
    for row in pool.itertuples(index=False):
        source = row._asdict()
        source_hash = str(source["source_pair_sha256"])
        if source_hash in excluded:
            continue
        contract = build_fact_contract(source)
        contracts[source_hash] = contract
        bin_index = score_bin(float(source["old_frida_score"]))
        if contract["tier"] == "A" and bin_index is not None:
            source["score_bin"] = bin_index
            candidates.append(source)

    selected: list[dict[str, object]] = []
    operation_use: Counter[str] = Counter()
    bin_use: Counter[tuple[str, int]] = Counter()
    for category, quota in CATEGORY_QUOTAS.items():
        available = [row for row in candidates if str(row["category"]) == category]
        chosen_hashes: set[str] = set()
        while len(chosen_hashes) < quota:
            remaining = [
                row for row in available
                if str(row["source_pair_sha256"]) not in chosen_hashes
            ]
            if not remaining:
                raise RuntimeError(f"Only {len(chosen_hashes)} candidates for {category}")

            def priority(source: dict[str, object]) -> tuple:
                source_hash = str(source["source_pair_sha256"])
                concepts = sorted({
                    str(operation["concept"])
                    for operation in contracts[source_hash]["allowed_operations"]
                })
                return (
                    bin_use[(category, int(source["score_bin"]))],
                    max((operation_use[value] for value in concepts), default=0),
                    sum(operation_use[value] for value in concepts),
                    -len(contracts[source_hash]["negative_anchors"]),
                    source_hash,
                )

            choice = min(remaining, key=priority)
            source_hash = str(choice["source_pair_sha256"])
            chosen_hashes.add(source_hash)
            selected.append(choice)
            bin_use[(category, int(choice["score_bin"]))] += 1
            for operation in contracts[source_hash]["allowed_operations"]:
                operation_use[str(operation["concept"])] += 1

    if len(selected) != sum(CATEGORY_QUOTAS.values()):
        raise RuntimeError("A2 v3.1 pilot size mismatch")
    selected_hashes = {str(row["source_pair_sha256"]) for row in selected}
    if selected_hashes & excluded:
        raise RuntimeError("A2 v3.1 pilot overlaps the A2 v3 pilot")

    selected_ids = [int(row[column]) for row in selected for column in ("id1", "id2")]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("A2 v3.1 pilot reuses an item")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot = pd.DataFrame(selected).drop(columns=["score_bin"])
    pool_path = args.output_dir / "pilot_source_pool.parquet"
    pilot.to_parquet(pool_path, index=False)
    plan_path = args.output_dir / "pilot_generation_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as stream:
        for source in selected:
            stream.write(
                json.dumps(
                    contracts[str(source["source_pair_sha256"])],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "status": "complete",
        "profile": "a2_v31_prescored_disjoint_pilot",
        "contract_version": 3,
        "scored_source_pool_sha256": file_sha256(args.scored_source_pool),
        "excluded_v3_pilot_sha256": file_sha256(args.exclude_v3_pilot),
        "excluded_source_count": len(excluded),
        "source_overlap_with_v3_pilot": len(selected_hashes & excluded),
        "selected_rows": len(pilot),
        "category_quotas": CATEGORY_QUOTAS,
        "selected_by_category": {
            str(key): int(value)
            for key, value in pilot["category"].value_counts().sort_index().items()
        },
        "selected_score_bins": {
            f"{category}:{SCORE_BINS[index][0]:.2f}-{SCORE_BINS[index][1]:.2f}": count
            for (category, index), count in sorted(bin_use.items())
        },
        "selected_score_summary": {
            "minimum": float(pilot["old_frida_score"].min()),
            "median": float(pilot["old_frida_score"].median()),
            "maximum": float(pilot["old_frida_score"].max()),
            "mean": float(pilot["old_frida_score"].mean()),
        },
        "operation_counts": dict(sorted(operation_use.items())),
        "item_reuse_count": len(selected_ids) - len(set(selected_ids)),
        "validation_content_used": False,
        "paid_requests_made": 0,
        "pilot_source_pool_sha256": file_sha256(pool_path),
        "pilot_generation_plan_sha256": file_sha256(plan_path),
    }
    (args.output_dir / "pilot_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
