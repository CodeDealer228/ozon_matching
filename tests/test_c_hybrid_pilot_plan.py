import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_c_hybrid_pilot_plan.py"
SPEC = importlib.util.spec_from_file_location("build_c_hybrid_pilot_plan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def opportunity(rule_id, source_id, donor_id, allowed=True):
    return {
        "rule_id": rule_id,
        "category": "test",
        "confidence_tier": "A" if allowed else "research",
        "recommended_sample_weight": 0.5 if allowed else 0.15,
        "generation_allowed_without_manual_override": allowed,
        "source_id": source_id,
        "source_name": f"source {source_id}",
        "source_attributes": "{}",
        "replacement_donor_id": donor_id,
        "changed_target_concepts": '["article"]',
        "source_values": '{"article": ["old"]}',
        "replacement_values": '{"article": ["new"]}',
        "context_values": "{}",
        "target_visible_in_source_or_donor_name": True,
        "operation": "replace",
        "opportunity_sha256": f"{rule_id}-{source_id}-{donor_id}",
    }


class PilotSelectionTest(unittest.TestCase):
    def test_simpler_fact_contract_is_preferred(self):
        simple = opportunity("a", 1, 2)
        complex_row = opportunity("a", 3, 4)
        complex_row["replacement_values"] = '{"article": ["a", "b", "c"]}'
        selected = MODULE.select_pilot(
            pd.DataFrame([complex_row, simple]), {"a": 1}, set(), 42
        )
        self.assertEqual(int(selected.source_id.iloc[0]), 1)

    def test_selection_is_item_disjoint(self):
        rows = [
            opportunity("a", 1, 2),
            opportunity("a", 1, 3),
            opportunity("a", 4, 5),
            opportunity("b", 6, 7),
            opportunity("b", 2, 8),
        ]
        selected = MODULE.select_pilot(pd.DataFrame(rows), {"a": 2, "b": 1}, set(), 42)
        ids = list(selected.source_id) + list(selected.replacement_donor_id)
        self.assertEqual(len(ids), len(set(ids)))

    def test_research_rule_requires_override(self):
        rows = pd.DataFrame([opportunity("research", 1, 2, allowed=False)])
        with self.assertRaisesRegex(ValueError, "explicit research override"):
            MODULE.select_pilot(rows, {"research": 1}, set(), 42)
        selected = MODULE.select_pilot(rows, {"research": 1}, {"research"}, 42)
        self.assertTrue(bool(selected.manual_override_applied.iloc[0]))

    def test_provider_payload_has_no_dataset_ids(self):
        rows = pd.DataFrame([opportunity("a", 1, 2)])
        selected = MODULE.select_pilot(rows, {"a": 1}, set(), 42)
        payload = next(iter(MODULE.provider_payloads(selected)))
        self.assertNotIn("source_id", payload)
        self.assertNotIn("replacement_donor_id", payload)


if __name__ == "__main__":
    unittest.main()
