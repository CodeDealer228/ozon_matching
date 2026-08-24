"""Build and verify the six-example A2 v3 policy pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from a2_fact_contract import validate_candidate


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PILOT_PLAN_SHA256 = "57a4c747c9a122cb1e29279c7c8e4fe5d3c3bc96965cc1890430b5b98483465b"
EXPECTED_LEGACY_PLAN_SHA256 = "0d9471241bb3e1927cf1482a0eb444cbc89be81b1bedaff0c803eda7d002a196"
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"

OK_CANDIDATES = {
    "9c18cd34c264": "Кардиган мужской Slava Zaitsev",
    "d959aa21323a": "Полусапоги Schtosen, черные",
    "17d4b449eb2a": "Кольцо из желтого золота 585 пробы, без вставок, р. 17,5 Delta л090119",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_contract(contract: dict[str, object]) -> dict[str, object]:
    return {
        "category": contract["category"],
        "a_safe_template": contract["a_safe_template"],
        "name2": contract["name2"],
        "allowed_operations": contract["allowed_operations"],
        "protected_numeric_mentions": contract["protected_numeric_mentions"],
        "protected_identifiers": contract["protected_identifiers"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-plan", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_pilot_plan" / "pilot_generation_plan.jsonl",
    )
    parser.add_argument(
        "--skip-plan", type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_v3_plan" / "generation_plan.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "synthetic_research" / "a2_few_shot_v3.json",
    )
    args = parser.parse_args()
    if file_sha256(args.pilot_plan) != EXPECTED_PILOT_PLAN_SHA256:
        raise RuntimeError("Pilot plan SHA-256 mismatch")
    if file_sha256(args.skip_plan) != EXPECTED_LEGACY_PLAN_SHA256:
        raise RuntimeError("Skip-source plan SHA-256 mismatch")

    pilot = read_jsonl(args.pilot_plan)
    skips = read_jsonl(args.skip_plan)
    examples: list[dict[str, object]] = []
    for prefix, candidate in OK_CANDIDATES.items():
        matches = [value for value in pilot if str(value["source_pair_sha256"]).startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one pilot contract for {prefix}")
        contract = matches[0]
        operations = list(contract["allowed_operations"])
        used_fact_ids = [str(value["fact_id"]) for value in operations]
        applied_operation_ids = [str(value["operation_id"]) for value in operations]
        ok, reasons, diagnostics = validate_candidate(
            contract, candidate, used_fact_ids, applied_operation_ids
        )
        if not ok:
            raise RuntimeError(f"Few-shot candidate {prefix} failed: {reasons} {diagnostics}")
        examples.append({
            "kind": "ok",
            "source_pair_sha256": contract["source_pair_sha256"],
            "input": compact_contract(contract),
            "output": {
                "status": "ok",
                "candidates": [{
                    "name2_prime": candidate,
                    "used_fact_ids": used_fact_ids,
                    "applied_operation_ids": applied_operation_ids,
                }],
            },
        })

    used_categories: set[str] = set()
    for contract in skips:
        if contract["tier"] != "C" or contract["allowed_operations"]:
            continue
        category = str(contract["category"])
        if category in used_categories:
            continue
        used_categories.add(category)
        examples.append({
            "kind": "skip",
            "source_pair_sha256": contract["source_pair_sha256"],
            "input": compact_contract(contract),
            "output": {"status": "skip", "reason": "no_safe_nontrivial_operation"},
        })
        if len(used_categories) == 3:
            break
    if len(examples) != 6:
        raise RuntimeError(f"Expected 6 few-shot examples, got {len(examples)}")

    pack = {
        "version": 3,
        "branch": "A2",
        "purpose": "policy examples for contract-constrained identity-preserving hardening",
        "source_scope": "canonical_human_train_only",
        "validation_examples_used": False,
        "canonical_split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "pilot_generation_plan_sha256": EXPECTED_PILOT_PLAN_SHA256,
        "counts": {"ok": 3, "skip": 3},
        "examples": examples,
    }
    args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "verified",
        "path": str(args.output.resolve()),
        "sha256": file_sha256(args.output),
        "examples": len(examples),
        "ok": 3,
        "skip": 3,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
