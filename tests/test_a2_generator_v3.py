from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_a2_pilot_qwen_v3 import build_messages, compact_contract, parse_response


class ProviderPayloadTests(unittest.TestCase):
    def test_compact_contract_excludes_local_identity_and_raw_attributes(self) -> None:
        contract = {
            "category": "Одежда",
            "a_safe_template": "кардиган мужской <brand_A_ONLY>",
            "name2": "кардиган B",
            "allowed_operations": [],
            "protected_numeric_mentions": [],
            "protected_identifiers": [],
            "source_pair_sha256": "secret-local-hash",
            "id1": 123,
            "id2": 456,
            "attributes2": "raw attributes",
        }
        payload = compact_contract(contract)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("secret-local-hash", serialized)
        self.assertNotIn("id1", serialized)
        self.assertNotIn("id2", serialized)
        self.assertNotIn("raw attributes", serialized)

    def test_messages_contain_no_source_ids(self) -> None:
        example_input = {
            "category": "Одежда",
            "a_safe_template": "A",
            "name2": "B",
            "allowed_operations": [],
            "protected_numeric_mentions": [],
            "protected_identifiers": [],
        }
        pack = {"examples": [{
            "input": example_input,
            "output": {"status": "skip", "reason": "no_safe_nontrivial_operation"},
        }]}
        contract = dict(example_input)
        contract.update({"source_pair_sha256": "never-send", "id1": 111, "id2": 222})
        serialized = json.dumps(build_messages(pack, contract, True), ensure_ascii=False)
        self.assertNotIn("never-send", serialized)
        self.assertNotIn('"id1"', serialized)
        self.assertNotIn('"id2"', serialized)


class ResponseSchemaTests(unittest.TestCase):
    def test_parse_skip(self) -> None:
        value = parse_response('{"status":"skip","reason":"no_safe_nontrivial_operation"}')
        self.assertEqual(value["status"], "skip")

    def test_parse_two_candidates(self) -> None:
        value = parse_response(json.dumps({
            "status": "ok",
            "candidates": [
                {"name2_prime": "B one", "used_fact_ids": ["f"], "applied_operation_ids": ["o"]},
                {"name2_prime": "B two", "used_fact_ids": ["f"], "applied_operation_ids": ["o"]},
            ],
        }))
        self.assertEqual(len(value["candidates"]), 2)

    def test_reject_more_than_two_candidates(self) -> None:
        with self.assertRaises(ValueError):
            parse_response(json.dumps({
                "status": "ok",
                "candidates": [
                    {"name2_prime": str(index), "used_fact_ids": [], "applied_operation_ids": []}
                    for index in range(3)
                ],
            }))


if __name__ == "__main__":
    unittest.main()
