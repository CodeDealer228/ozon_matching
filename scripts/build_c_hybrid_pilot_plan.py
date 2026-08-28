#!/usr/bin/env python3
"""Build a deterministic, item-disjoint Method C pilot plan.

This script never calls an external model and never creates synthetic text. It
selects audited replacement opportunities and writes a provider-safe request
pack without dataset item IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_QUOTAS = {
    "c_auto_part_number": 40,
    "c_jewelry_base_metal": 40,
    "c_auto_platform": 20,
}
DEFAULT_RESEARCH_OVERRIDES = {"c_auto_platform"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_value_count(raw_json: object) -> int:
    values = json.loads(str(raw_json))
    return sum(len(value) if isinstance(value, list) else 1 for value in values.values())


def stable_order_key(seed: int, row: pd.Series) -> tuple[int, int, str]:
    visible_rank = 0 if bool(row["target_visible_in_source_or_donor_name"]) else 1
    contract_complexity = contract_value_count(row["source_values"]) + contract_value_count(
        row["replacement_values"]
    )
    identity = hashlib.sha256(
        f"{seed}|{row['rule_id']}|{row['opportunity_sha256']}".encode("utf-8")
    ).hexdigest()
    return visible_rank, contract_complexity, identity


def select_pilot(
    opportunities: pd.DataFrame,
    quotas: dict[str, int],
    research_overrides: set[str],
    seed: int,
) -> pd.DataFrame:
    required = {
        "rule_id",
        "category",
        "confidence_tier",
        "recommended_sample_weight",
        "generation_allowed_without_manual_override",
        "source_id",
        "source_name",
        "source_attributes",
        "replacement_donor_id",
        "changed_target_concepts",
        "source_values",
        "replacement_values",
        "context_values",
        "target_visible_in_source_or_donor_name",
        "operation",
        "opportunity_sha256",
    }
    missing = required - set(opportunities.columns)
    if missing:
        raise ValueError(f"opportunities missing columns: {sorted(missing)}")

    used_item_ids: set[int] = set()
    selected: list[dict[str, object]] = []
    for rule_id, quota in quotas.items():
        if quota <= 0:
            raise ValueError(f"quota must be positive for {rule_id}")
        candidates = opportunities.loc[opportunities.rule_id == rule_id].copy()
        if candidates.empty:
            raise ValueError(f"no opportunities for {rule_id}")
        generation_allowed = bool(
            candidates["generation_allowed_without_manual_override"].iloc[0]
        )
        override = rule_id in research_overrides
        if not generation_allowed and not override:
            raise ValueError(f"{rule_id} requires an explicit research override")

        candidates["_order"] = candidates.apply(
            lambda row: stable_order_key(seed, row), axis=1
        )
        candidates = candidates.sort_values("_order", kind="stable")
        rule_selected = 0
        for _, row in candidates.iterrows():
            source_id = int(row["source_id"])
            donor_id = int(row["replacement_donor_id"])
            if source_id == donor_id:
                continue
            if source_id in used_item_ids or donor_id in used_item_ids:
                continue
            record = row.drop(labels=["_order"]).to_dict()
            record["manual_override_applied"] = bool(override and not generation_allowed)
            record["fact_contract_value_count"] = contract_value_count(
                row["source_values"]
            ) + contract_value_count(row["replacement_values"])
            record["pilot_row_index"] = len(selected)
            selected.append(record)
            used_item_ids.update((source_id, donor_id))
            rule_selected += 1
            if rule_selected >= quota:
                break
        if rule_selected != quota:
            raise ValueError(
                f"could select only {rule_selected}/{quota} item-disjoint rows for {rule_id}"
            )

    result = pd.DataFrame(selected)
    if result.source_id.duplicated().any():
        raise AssertionError("duplicate source_id")
    all_ids = pd.concat(
        [result.source_id.astype("int64"), result.replacement_donor_id.astype("int64")]
    )
    if all_ids.duplicated().any():
        raise AssertionError("source/donor item reuse")
    if result.opportunity_sha256.duplicated().any():
        raise AssertionError("duplicate opportunity")
    return result


def provider_payloads(rows: pd.DataFrame) -> Iterable[dict[str, object]]:
    """Yield the future provider contract. Dataset IDs are intentionally absent."""
    for row in rows.itertuples(index=False):
        yield {
            "pilot_row_index": int(row.pilot_row_index),
            "category": str(row.category),
            "rule_id": str(row.rule_id),
            "operation": str(row.operation),
            "source_name": str(row.source_name),
            "source_attributes": json.loads(str(row.source_attributes)),
            "changed_target_concepts": json.loads(str(row.changed_target_concepts)),
            "source_values": json.loads(str(row.source_values)),
            "replacement_values": json.loads(str(row.replacement_values)),
            "context_values": json.loads(str(row.context_values)),
            "required_output": {
                "status": "ok_or_skip",
                "synthetic_name": "natural marketplace title when status=ok",
                "synthetic_attributes": "JSON object consistent with the title",
            },
        }


def parse_quotas(values: list[str] | None) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_QUOTAS)
    result: dict[str, int] = {}
    for value in values:
        rule_id, separator, raw_count = value.partition("=")
        if not separator or not rule_id:
            raise ValueError(f"invalid quota {value!r}; expected RULE=COUNT")
        result[rule_id] = int(raw_count)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("artifacts/synthetic_v2/c_pattern_audit_v5")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/synthetic_v2/c_hybrid_pilot_plan_v1")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quota", action="append")
    parser.add_argument("--research-override", action="append")
    args = parser.parse_args()

    quotas = parse_quotas(args.quota)
    overrides = (
        set(args.research_override)
        if args.research_override is not None
        else set(DEFAULT_RESEARCH_OVERRIDES)
    )
    manifest_path = args.audit_dir / "manifest.json"
    opportunities_path = args.audit_dir / "replacement_opportunities.parquet"
    audit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit_manifest.get("status") != "complete":
        raise ValueError("upstream audit is not complete")
    expected_sha = audit_manifest["outputs"]["replacement_opportunities"]["sha256"]
    actual_sha = sha256_file(opportunities_path)
    if actual_sha != expected_sha:
        raise ValueError("replacement opportunities SHA-256 mismatch")
    if audit_manifest["leak_checks"]["opportunity_validation_item_overlap"] != 0:
        raise ValueError("upstream audit reports validation item overlap")

    opportunities = pd.read_parquet(opportunities_path)
    selected = select_pilot(opportunities, quotas, overrides, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "pilot_plan.parquet"
    payload_path = args.output_dir / "provider_request_pack.jsonl"
    selected.to_parquet(plan_path, index=False)
    with payload_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in provider_payloads(selected):
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    provider_text = payload_path.read_text(encoding="utf-8")
    forbidden_provider_keys = ('"source_id"', '"replacement_donor_id"')
    if any(key in provider_text for key in forbidden_provider_keys):
        raise AssertionError("provider request pack contains dataset IDs")

    manifest = {
        "status": "complete",
        "profile": "c_hybrid_pilot_plan_v1",
        "external_calls_performed": False,
        "synthetic_generation_performed": False,
        "seed": args.seed,
        "quotas": quotas,
        "research_overrides": sorted(overrides),
        "rows": len(selected),
        "counts_by_rule": selected.rule_id.value_counts().sort_index().to_dict(),
        "confidence_tiers": selected.confidence_tier.value_counts().sort_index().to_dict(),
        "unique_source_items": int(selected.source_id.nunique()),
        "unique_donor_items": int(selected.replacement_donor_id.nunique()),
        "source_donor_item_reuse": 0,
        "validation_item_overlap": 0,
        "upstream": {
            "audit_manifest_sha256": sha256_file(manifest_path),
            "opportunities_sha256": actual_sha,
            "split_sha256": audit_manifest["split"]["sha256"],
            "validation_fingerprint": audit_manifest["split"]["validation_fingerprint"],
        },
        "outputs": {
            "pilot_plan": {"bytes": plan_path.stat().st_size, "sha256": sha256_file(plan_path)},
            "provider_request_pack": {
                "bytes": payload_path.stat().st_size,
                "sha256": sha256_file(payload_path),
                "contains_dataset_ids": False,
            },
        },
    }
    manifest_path_out = args.output_dir / "manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
