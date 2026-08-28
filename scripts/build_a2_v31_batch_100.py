"""Build the first 100-source A2 v3.1 scale batch.

The batch is selected only from the frozen 400-row, Frida-scored Tier-A
human-train pool and excludes both earlier v3/v3.1 pilots.  Selection is
balanced by category and source-score bin and never reuses an item.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from a2_fact_contract import build_fact_contract
from build_a2_v31_pilot_plan import SCORE_BINS, file_sha256, score_bin


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCORED_POOL_SHA256 = (
    "d1866fd145f158b5eaff4e200f261093292a98636fe139840e33c77de3267456"
)
EXPECTED_V3_PILOT_SHA256 = (
    "084a79f816e2b47ace88b0dea74b9ead69c70c371603837f05cf6a63ceca2770"
)
EXPECTED_V31_PILOT_SHA256 = (
    "4cd7635b24933a4ab9989517177c5102ce1a3e3fbb4ec020a2c81d63a1c5cb6d"
)
CATEGORY_QUOTAS = {
    "Автотовары": 25,
    "Обувь": 25,
    "Одежда": 24,
    "Ювелирные изделия": 26,
}


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
        "--exclude-v31-pilot",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts/synthetic_v2/a2_v31_pilot_plan/pilot_source_pool.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/a2_v31_batch_100_01_plan",
    )
    args = parser.parse_args()

    for path, expected, label in (
        (args.scored_source_pool, EXPECTED_SCORED_POOL_SHA256, "scored pool"),
        (args.exclude_v3_pilot, EXPECTED_V3_PILOT_SHA256, "v3 pilot"),
        (args.exclude_v31_pilot, EXPECTED_V31_PILOT_SHA256, "v3.1 pilot"),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Unexpected {label} SHA-256: {actual}")

    pool = pd.read_parquet(args.scored_source_pool)
    previous_v3 = pd.read_parquet(args.exclude_v3_pilot)
    previous_v31 = pd.read_parquet(args.exclude_v31_pilot)
    if len(pool) != 400 or set(pool["label"].unique()) != {0}:
        raise RuntimeError("Unexpected scored source-pool identity")
    excluded = set(previous_v3["source_pair_sha256"].astype(str)) | set(
        previous_v31["source_pair_sha256"].astype(str)
    )
    if len(excluded) != 52:
        raise RuntimeError("Expected 52 disjoint pilot exclusions")

    contracts: dict[str, dict[str, object]] = {}
    eligible: list[dict[str, object]] = []
    for source in pool.to_dict("records"):
        source_hash = str(source["source_pair_sha256"])
        if source_hash in excluded:
            continue
        contract = build_fact_contract(source)
        contracts[source_hash] = contract
        bin_index = score_bin(float(source["old_frida_score"]))
        if contract["tier"] == "A" and bin_index is not None:
            source["score_bin"] = bin_index
            eligible.append(source)

    selected: list[dict[str, object]] = []
    operation_use: Counter[str] = Counter()
    bin_use: Counter[tuple[str, int]] = Counter()
    used_items: set[int] = set()
    for category, quota in CATEGORY_QUOTAS.items():
        available = [row for row in eligible if str(row["category"]) == category]
        while sum(str(row["category"]) == category for row in selected) < quota:
            remaining = [
                row
                for row in available
                if int(row["id1"]) not in used_items
                and int(row["id2"]) not in used_items
                and row not in selected
            ]
            if not remaining:
                have = sum(str(row["category"]) == category for row in selected)
                raise RuntimeError(f"Only {have} unique-item candidates for {category}")

            def priority(source: dict[str, object]) -> tuple:
                source_hash = str(source["source_pair_sha256"])
                concepts = sorted(
                    {
                        str(operation["concept"])
                        for operation in contracts[source_hash]["allowed_operations"]
                    }
                )
                return (
                    bin_use[(category, int(source["score_bin"]))],
                    max((operation_use[value] for value in concepts), default=0),
                    sum(operation_use[value] for value in concepts),
                    -len(contracts[source_hash]["negative_anchors"]),
                    source_hash,
                )

            choice = min(remaining, key=priority)
            source_hash = str(choice["source_pair_sha256"])
            selected.append(choice)
            used_items.update((int(choice["id1"]), int(choice["id2"])))
            bin_use[(category, int(choice["score_bin"]))] += 1
            for operation in contracts[source_hash]["allowed_operations"]:
                operation_use[str(operation["concept"])] += 1

    selected_hashes = {str(row["source_pair_sha256"]) for row in selected}
    if len(selected) != 100 or len(selected_hashes) != 100:
        raise RuntimeError("A2 v3.1 scale batch must contain 100 unique sources")
    if selected_hashes & excluded:
        raise RuntimeError("Scale batch overlaps a previous pilot")
    if len(used_items) != 200:
        raise RuntimeError("Scale batch reuses an item")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = pd.DataFrame(selected).drop(columns=["score_bin"])
    pool_path = args.output_dir / "source_pool.parquet"
    batch.to_parquet(pool_path, index=False)
    plan_path = args.output_dir / "generation_plan.jsonl"
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
        "profile": "a2_v31_scale_batch_100_01",
        "contract_version": 3,
        "scored_source_pool_sha256": file_sha256(args.scored_source_pool),
        "excluded_v3_pilot_sha256": file_sha256(args.exclude_v3_pilot),
        "excluded_v31_pilot_sha256": file_sha256(args.exclude_v31_pilot),
        "excluded_source_count": len(excluded),
        "selected_rows": len(batch),
        "selected_by_category": {
            str(key): int(value)
            for key, value in batch["category"].value_counts().sort_index().items()
        },
        "selected_score_bins": {
            f"{category}:{SCORE_BINS[index][0]:.2f}-{SCORE_BINS[index][1]:.2f}": count
            for (category, index), count in sorted(bin_use.items())
        },
        "operation_counts": dict(sorted(operation_use.items())),
        "source_overlap_with_previous_pilots": len(selected_hashes & excluded),
        "item_reuse_count": 200 - len(used_items),
        "validation_content_used": False,
        "paid_requests_made": 0,
        "source_pool_sha256": file_sha256(pool_path),
        "generation_plan_sha256": file_sha256(plan_path),
    }
    (args.output_dir / "batch_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
