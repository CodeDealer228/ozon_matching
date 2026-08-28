"""Build a disjoint 200-source A2 v3.1 continuation batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from a2_fact_contract import build_fact_contract
from build_a2_v31_pilot_plan import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCORED_POOL_SHA256 = (
    "d1866fd145f158b5eaff4e200f261093292a98636fe139840e33c77de3267456"
)
EXCLUSIONS = (
    (
        "artifacts/synthetic_v2/a2_v3_pilot_plan/pilot_source_pool.parquet",
        "084a79f816e2b47ace88b0dea74b9ead69c70c371603837f05cf6a63ceca2770",
    ),
    (
        "artifacts/synthetic_v2/a2_v31_pilot_plan/pilot_source_pool.parquet",
        "4cd7635b24933a4ab9989517177c5102ce1a3e3fbb4ec020a2c81d63a1c5cb6d",
    ),
    (
        "artifacts/synthetic_v2/a2_v31_batch_100_01_plan/source_pool.parquet",
        "c7b8e8b02648dcb880a567bb6fb3be6a7e42740b42f7ad385543fc01724a1f0f",
    ),
)
CATEGORY_QUOTAS = {
    "Автотовары": 50,
    "Обувь": 50,
    "Одежда": 50,
    "Ювелирные изделия": 50,
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/a2_v31_batch_200_02_plan",
    )
    args = parser.parse_args()

    if file_sha256(args.scored_source_pool) != EXPECTED_SCORED_POOL_SHA256:
        raise RuntimeError("Scored source-pool SHA-256 mismatch")
    excluded: set[str] = set()
    exclusion_hashes: dict[str, str] = {}
    for relative, expected in EXCLUSIONS:
        path = REPO_ROOT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Exclusion SHA-256 mismatch: {relative}: {actual}")
        excluded.update(pd.read_parquet(path)["source_pair_sha256"].astype(str))
        exclusion_hashes[relative] = actual
    if len(excluded) != 152:
        raise RuntimeError(f"Expected 152 prior sources, found {len(excluded)}")

    pool = pd.read_parquet(args.scored_source_pool)
    if len(pool) != 400 or set(pool["label"].unique()) != {0}:
        raise RuntimeError("Unexpected scored source-pool identity")

    contracts: dict[str, dict[str, object]] = {}
    remaining: list[dict[str, object]] = []
    non_tier_a_excluded: list[str] = []
    for source in pool.to_dict("records"):
        source_hash = str(source["source_pair_sha256"])
        if source_hash in excluded:
            continue
        contract = build_fact_contract(source)
        if contract["tier"] != "A":
            non_tier_a_excluded.append(source_hash)
            continue
        contracts[source_hash] = contract
        remaining.append(source)
    if len(remaining) != 247 or len(non_tier_a_excluded) != 1:
        raise RuntimeError(
            "Expected 247 current Tier-A sources and one stricter-contract exclusion"
        )

    selected: list[dict[str, object]] = []
    operation_use: Counter[str] = Counter()
    used_items: set[int] = set()
    for category, quota in CATEGORY_QUOTAS.items():
        available = [row for row in remaining if str(row["category"]) == category]
        while sum(str(row["category"]) == category for row in selected) < quota:
            candidates = [
                row
                for row in available
                if row not in selected
                and int(row["id1"]) not in used_items
                and int(row["id2"]) not in used_items
            ]
            if not candidates:
                raise RuntimeError(f"Insufficient unique sources for {category}")

            def priority(source: dict[str, object]) -> tuple:
                source_hash = str(source["source_pair_sha256"])
                concepts = sorted(
                    str(operation["concept"])
                    for operation in contracts[source_hash]["allowed_operations"]
                )
                score = float(source["old_frida_score"])
                outside_preferred_band = not (0.20 <= score <= 0.60)
                return (
                    outside_preferred_band,
                    max((operation_use[value] for value in concepts), default=0),
                    sum(operation_use[value] for value in concepts),
                    -len(contracts[source_hash]["negative_anchors"]),
                    abs(score - 0.40),
                    source_hash,
                )

            choice = min(candidates, key=priority)
            selected.append(choice)
            used_items.update((int(choice["id1"]), int(choice["id2"])))
            source_hash = str(choice["source_pair_sha256"])
            for operation in contracts[source_hash]["allowed_operations"]:
                operation_use[str(operation["concept"])] += 1

    selected_hashes = {str(row["source_pair_sha256"]) for row in selected}
    if len(selected) != 200 or len(selected_hashes) != 200:
        raise RuntimeError("Continuation batch must contain 200 unique sources")
    if selected_hashes & excluded:
        raise RuntimeError("Continuation batch overlaps a previous batch")
    if len(used_items) != 400:
        raise RuntimeError("Continuation batch reuses an item")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(selected)
    pool_path = args.output_dir / "source_pool.parquet"
    frame.to_parquet(pool_path, index=False)
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
        "profile": "a2_v31_scale_batch_200_02",
        "scored_source_pool_sha256": file_sha256(args.scored_source_pool),
        "exclusion_hashes": exclusion_hashes,
        "excluded_source_count": len(excluded),
        "remaining_before_selection": len(remaining),
        "non_tier_a_excluded": non_tier_a_excluded,
        "selected_rows": len(frame),
        "selected_by_category": {
            str(key): int(value)
            for key, value in frame["category"].value_counts().sort_index().items()
        },
        "selected_old_score": {
            "minimum": float(frame["old_frida_score"].min()),
            "median": float(frame["old_frida_score"].median()),
            "maximum": float(frame["old_frida_score"].max()),
        },
        "operation_counts": dict(sorted(operation_use.items())),
        "source_overlap_with_previous": len(selected_hashes & excluded),
        "item_reuse_count": 400 - len(used_items),
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
