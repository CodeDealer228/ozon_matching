from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_a2_pilot_qwen_v31 import parse_response


def candidate(kind: str, name: str, facts=None, operations=None) -> dict:
    return {
        "variant_kind": kind,
        "name2_prime": name,
        "used_fact_ids": ["f"] if facts is None else facts,
        "applied_operation_ids": ["o"] if operations is None else operations,
    }


class V31ResponseSchemaTests(unittest.TestCase):
    def test_accepts_two_distinct_same_fact_candidates(self) -> None:
        value = parse_response(json.dumps({
            "status": "ok",
            "candidates": [
                candidate("minimal", "B fact"),
                candidate("alternate_placement", "fact B"),
            ],
        }))
        self.assertEqual(len(value["candidates"]), 2)

    def test_rejects_one_candidate(self) -> None:
        with self.assertRaises(ValueError):
            parse_response(json.dumps({
                "status": "ok",
                "candidates": [candidate("minimal", "B fact")],
            }))

    def test_rejects_duplicate_surface_forms(self) -> None:
        with self.assertRaises(ValueError):
            parse_response(json.dumps({
                "status": "ok",
                "candidates": [
                    candidate("minimal", "B fact"),
                    candidate("alternate_placement", "  b FACT  "),
                ],
            }))

    def test_rejects_different_fact_sets(self) -> None:
        with self.assertRaises(ValueError):
            parse_response(json.dumps({
                "status": "ok",
                "candidates": [
                    candidate("minimal", "B fact"),
                    candidate("alternate_placement", "fact B", facts=["different"]),
                ],
            }))


if __name__ == "__main__":
    unittest.main()
