"""Build the immutable local-only A2 v3 generation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from a2_fact_contract import build_fact_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "2dee9a0b5b2ef54c4210b07b114346057bddc45effe728e2e43056f42505dd28"
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"


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
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "source_pools" / "a2_source_pool.parquet",
    )
    parser.add_argument(
        "--source-manifest", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "source_pools" / "source_pool_manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_plan",
    )
    args = parser.parse_args()

    if file_sha256(args.source_pool) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("A2 source-pool SHA-256 mismatch")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if manifest["split"]["sha256"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Canonical split SHA-256 mismatch")
    if manifest["split"]["validation_fingerprint"] != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Validation fingerprint mismatch")
    if any(int(value) for value in manifest["leak_checks"].values()):
        raise RuntimeError(f"Source manifest leak check failed: {manifest['leak_checks']}")

    pool = pd.read_parquet(args.source_pool)
    if len(pool) != 500 or set(pool["label"].unique()) != {0}:
        raise RuntimeError("Unexpected A2 source pool identity")
    contracts = [build_fact_contract(row._asdict()) for row in pool.itertuples(index=False)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "generation_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as stream:
        for contract in contracts:
            stream.write(json.dumps(contract, ensure_ascii=False, sort_keys=True) + "\n")

    tiers = Counter(str(value["tier"]) for value in contracts)
    by_category = Counter((str(value["category"]), str(value["tier"])) for value in contracts)
    operation_concepts = Counter(
        str(operation["concept"])
        for contract in contracts
        for operation in contract["allowed_operations"]
    )
    anchor_concepts = Counter(
        str(anchor["concept"])
        for contract in contracts
        for anchor in contract["negative_anchors"]
    )
    summary = {
        "status": "complete",
        "scope": "canonical_human_train_only",
        "validation_content_used": False,
        "contract_version": 3,
        "source_pool": {
            "path": str(args.source_pool.resolve()),
            "rows": len(pool),
            "sha256": file_sha256(args.source_pool),
        },
        "split_identity": {
            "sha256": EXPECTED_SPLIT_SHA256,
            "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
            "leak_checks": manifest["leak_checks"],
        },
        "tier_counts": dict(sorted(tiers.items())),
        "tier_counts_by_category": {
            category: {
                tier: by_category.get((category, tier), 0)
                for tier in ("A", "B", "C")
            }
            for category in sorted(set(pool["category"]))
        },
        "operation_concepts": dict(sorted(operation_concepts.items())),
        "negative_anchor_concepts": dict(sorted(anchor_concepts.items())),
        "paid_pilot_policy": "tier_A_only_max_20_no_forced_category_balance",
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
    }
    summary_path = args.output_dir / "generation_plan_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
