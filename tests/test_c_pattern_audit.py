from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_c_pattern_catalog import (
    confidence_result,
    identifier_like,
    item_scope_allowed,
    parse_pattern_markdown,
)


class PatternParserTests(unittest.TestCase):
    def test_parses_fields_and_pair_references(self) -> None:
        text = """# Test

## Generation pattern 7

**Generalized rule:** First line
continues here.

**Scope:** Narrow family.

**Generation operation:** Replace one value.

**Guards:** Keep every other fact.

**Confidence:** strict

**Negative evidence:** pair_0001, pair_0002

**Positive counterexamples:** pair_0003
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_patterns.md"
            path.write_text(text, encoding="utf-8")
            parsed = parse_pattern_markdown(path, "Example")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["pattern_index"], 7)
        self.assertEqual(parsed[0]["generalized_rule"], "First line continues here.")
        self.assertEqual(parsed[0]["negative_evidence_ids"], ["pair_0001", "pair_0002"])
        self.assertEqual(parsed[0]["positive_counterexample_ids"], ["pair_0003"])
        self.assertEqual(parsed[0]["missing_required_fields"], [])


class ConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = {
            "prior_confidence": "strict",
            "min_context_matches": 1,
        }
        self.catalog = {"catalog_reference_audit_passed": True}

    def test_supported_clean_rule_reaches_tier_a(self) -> None:
        rows = [
            {
                "label": 0,
                "target_visible": True,
                "matched_context_count": 1,
            }
            for _ in range(7)
        ]
        result = confidence_result(self.rule, self.catalog, rows)
        self.assertEqual(result["confidence_tier"], "A")
        self.assertEqual(result["recommended_sample_weight"], 0.5)

    def test_counterexamples_reduce_rule_to_research(self) -> None:
        rows = [
            {
                "label": label,
                "target_visible": True,
                "matched_context_count": 1,
            }
            for label in (0, 1, 1, 1)
        ]
        result = confidence_result(self.rule, self.catalog, rows)
        self.assertEqual(result["confidence_tier"], "research")
        self.assertFalse(result["generation_allowed_without_manual_override"])

    def test_guarded_prior_cannot_reach_tier_a(self) -> None:
        self.rule["prior_confidence"] = "guarded"
        rows = [
            {"label": 0, "target_visible": True, "matched_context_count": 1}
            for _ in range(20)
        ]
        result = confidence_result(self.rule, self.catalog, rows)
        self.assertEqual(result["confidence_tier"], "B")


class ValuePolicyTests(unittest.TestCase):
    def test_identifier_like_requires_a_digit(self) -> None:
        self.assertTrue(identifier_like("Galaxy 5"))
        self.assertTrue(identifier_like("2310801"))
        self.assertFalse(identifier_like("повседневная модель"))


class ItemScopeTests(unittest.TestCase):
    def test_name_deny_regex_catches_noisy_item_type(self) -> None:
        rule = {"item_name_deny_regex": r"автошин|\bшина\b"}
        self.assertFalse(
            item_scope_allowed(None, None, {"name": "Autoparts автошина R15"}, rule)
        )
        self.assertTrue(
            item_scope_allowed(None, None, {"name": "Lynxauto ремень 6PK1520"}, rule)
        )


if __name__ == "__main__":
    unittest.main()
