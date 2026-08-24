"""Build the two-candidate policy examples for A2 v3.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from a2_fact_contract import validate_candidate


REPO_ROOT = Path(__file__).resolve().parents[1]
V3_PACK = REPO_ROOT / "synthetic_research/a2_few_shot_v3.json"
V3_PLAN = REPO_ROOT / "artifacts/synthetic_v2/a2_v3_pilot_plan/pilot_generation_plan.jsonl"
V31_PLAN_SUMMARY = REPO_ROOT / "artifacts/synthetic_v2/a2_v31_pilot_plan/pilot_plan_summary.json"
OUTPUT = REPO_ROOT / "synthetic_research/a2_few_shot_v31.json"
EXPECTED_V3_PACK_SHA256 = "00314f0b40fdc2fbc621ede716cc7e74deed02be6ae957e96e1a68aa597d200c"
EXPECTED_V3_PLAN_SHA256 = "57a4c747c9a122cb1e29279c7c8e4fe5d3c3bc96965cc1890430b5b98483465b"


VARIANTS = {
    "9c18cd34c264d8b716dba53d3c882bbd08b5760ff388ccd156a0cad7e8c3cc79": [
        ("minimal", "Кардиган мужской Slava Zaitsev"),
        ("alternate_placement", "Мужской кардиган Slava Zaitsev"),
    ],
    "d959aa21323a322c076af1cbf964441dbaa87147108ab1d606ff60297795f0ed": [
        ("minimal", "Полусапоги Schtosen, черные"),
        ("alternate_placement", "Черные полусапоги Schtosen"),
    ],
    "17d4b449eb2a0457f03b7b362d41fd302d6ba0c1008c707e9bef19d0d51827d8": [
        (
            "minimal",
            "Кольцо из желтого золота 585 пробы, без вставок, р. 17,5 Delta л090119",
        ),
        (
            "alternate_placement",
            "Кольцо 585 пробы из желтого золота, без вставок, р. 17,5 Delta л090119",
        ),
    ],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if file_sha256(V3_PACK) != EXPECTED_V3_PACK_SHA256:
        raise RuntimeError("A2 v3 few-shot pack SHA-256 mismatch")
    if file_sha256(V3_PLAN) != EXPECTED_V3_PLAN_SHA256:
        raise RuntimeError("A2 v3 plan SHA-256 mismatch")

    pack = json.loads(V3_PACK.read_text(encoding="utf-8"))
    contracts = {
        str(value["source_pair_sha256"]): value
        for value in (
            json.loads(line)
            for line in V3_PLAN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    plan_summary = json.loads(V31_PLAN_SUMMARY.read_text(encoding="utf-8"))

    for example in pack["examples"]:
        if example["kind"] != "ok":
            continue
        source_hash = str(example["source_pair_sha256"])
        old_candidate = example["output"]["candidates"][0]
        candidates = []
        for variant_kind, name2_prime in VARIANTS[source_hash]:
            accepted, reasons, _diagnostics = validate_candidate(
                contracts[source_hash],
                name2_prime,
                old_candidate["used_fact_ids"],
                old_candidate["applied_operation_ids"],
            )
            if not accepted:
                raise RuntimeError(
                    f"Invalid v3.1 few-shot candidate for {source_hash}: {reasons}"
                )
            candidates.append({
                "variant_kind": variant_kind,
                "name2_prime": name2_prime,
                "used_fact_ids": old_candidate["used_fact_ids"],
                "applied_operation_ids": old_candidate["applied_operation_ids"],
            })
        if len({value["name2_prime"].casefold() for value in candidates}) != 2:
            raise RuntimeError(f"Non-distinct v3.1 candidates for {source_hash}")
        example["output"]["candidates"] = candidates

    pack.update({
        "version": "3.1",
        "purpose": "two-candidate contract-constrained identity-preserving hardening",
        "based_on_v3_pack_sha256": file_sha256(V3_PACK),
        "pilot_generation_plan_sha256": plan_summary["pilot_generation_plan_sha256"],
        "candidate_policy": {
            "status_ok_requires_exactly": 2,
            "variant_kinds": ["minimal", "alternate_placement"],
            "new_facts_between_variants_allowed": False,
        },
    })
    OUTPUT.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "path": str(OUTPUT),
        "examples": len(pack["examples"]),
        "ok_examples": pack["counts"]["ok"],
        "skip_examples": pack["counts"]["skip"],
        "sha256": file_sha256(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
