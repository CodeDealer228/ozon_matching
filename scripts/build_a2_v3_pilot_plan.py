"""Rebuild contracts for the selected A2 v3 pool and choose a diverse pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from a2_fact_contract import build_fact_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHA256 = "67a111a5a81f2cbac0962ba2848d3b756427b47a1c773fb7e28d75162729f484"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_candidates" / "a2_v3_source_pool.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_pilot_plan",
    )
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--soft-category-cap", type=int, default=5)
    parser.add_argument("--operation-cap", type=int, default=6)
    args = parser.parse_args()

    if file_sha256(args.source_pool) != EXPECTED_INPUT_SHA256:
        raise RuntimeError("A2 v3 opportunity pool SHA-256 mismatch")
    pool = pd.read_parquet(args.source_pool)
    contracts: dict[str, dict[str, object]] = {}
    eligible_rows: list[dict[str, object]] = []
    for row in pool.itertuples(index=False):
        source = row._asdict()
        contract = build_fact_contract(source)
        contracts[str(source["source_pair_sha256"])] = contract
        if contract["tier"] == "A":
            eligible_rows.append(source)

    eligible_rows.sort(
        key=lambda source: (
            len(contracts[str(source["source_pair_sha256"])]["allowed_operations"]),
            -len(contracts[str(source["source_pair_sha256"])]["negative_anchors"]),
            -float(source["name_similarity"]),
            str(source["source_pair_sha256"]),
        )
    )
    selected: list[dict[str, object]] = []
    selected_hashes: set[str] = set()
    category_use: Counter[str] = Counter()
    operation_use: Counter[str] = Counter()

    def can_add(source: dict[str, object], enforce_category_cap: bool) -> bool:
        contract = contracts[str(source["source_pair_sha256"])]
        concepts = {str(value["concept"]) for value in contract["allowed_operations"]}
        if enforce_category_cap and category_use[str(source["category"])] >= args.soft_category_cap:
            return False
        return all(operation_use[concept] < args.operation_cap for concept in concepts)

    for enforce_category_cap in (True, False):
        for source in eligible_rows:
            source_hash = str(source["source_pair_sha256"])
            if source_hash in selected_hashes or not can_add(source, enforce_category_cap):
                continue
            selected.append(source)
            selected_hashes.add(source_hash)
            category_use[str(source["category"])] += 1
            for operation in contracts[source_hash]["allowed_operations"]:
                operation_use[str(operation["concept"])] += 1
            if len(selected) >= args.pilot_size:
                break
        if len(selected) >= args.pilot_size:
            break
    if len(selected) < args.pilot_size:
        raise RuntimeError(f"Only {len(selected)} safe diverse pilot sources found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot = pd.DataFrame(selected)
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
        "contract_version": 3,
        "input_pool_sha256": file_sha256(args.source_pool),
        "input_rows": len(pool),
        "tier_a_after_contract_fix": len(eligible_rows),
        "pilot_rows": len(pilot),
        "pilot_by_category": dict(sorted(category_use.items())),
        "pilot_operation_counts": dict(sorted(operation_use.items())),
        "pilot_source_pool_sha256": file_sha256(pool_path),
        "pilot_generation_plan_sha256": file_sha256(plan_path),
        "validation_content_used": False,
        "automatic_retries": 0,
        "thinking": False,
    }
    (args.output_dir / "pilot_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
