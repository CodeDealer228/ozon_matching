import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from c_hybrid_contract import validate_response


class ContractValidationTests(unittest.TestCase):
    def jewelry_plan(self):
        return {
            "rule_id": "c_jewelry_base_metal",
            "source_name": "Кольцо из серебра 925 с фианитом",
            "source_attributes": '{"материал":"серебро","проба":"925","вставка":"фианит"}',
            "source_values": '{"jewelry_material":["silver"],"metal_fineness":["925"]}',
            "replacement_values": '{"jewelry_material":["gold"],"metal_fineness":["585"]}',
        }

    def test_accepts_coherent_metal_and_fineness_patch(self):
        response = {
            "status": "ok",
            "synthetic_name": "Кольцо из золота 585 с фианитом",
            "attribute_updates": {"материал": "золото", "проба": "585"},
        }
        accepted, reasons, diagnostics = validate_response(self.jewelry_plan(), response)
        self.assertTrue(accepted, reasons)
        self.assertIn('"вставка":"фианит"', diagnostics["synthetic_attributes"])

    def test_rejects_update_outside_target_scope(self):
        response = {
            "status": "ok",
            "synthetic_name": "Кольцо из золота 585 с рубином",
            "attribute_updates": {
                "материал": "золото",
                "проба": "585",
                "вставка": "рубин",
            },
        }
        accepted, reasons, _ = validate_response(self.jewelry_plan(), response)
        self.assertFalse(accepted)
        self.assertIn("attribute_update_outside_target_scope", reasons)

    def test_rejects_old_target_in_title(self):
        response = {
            "status": "ok",
            "synthetic_name": "Кольцо серебро, золото 585 с фианитом",
            "attribute_updates": {"материал": "золото", "проба": "585"},
        }
        accepted, reasons, _ = validate_response(self.jewelry_plan(), response)
        self.assertFalse(accepted)
        self.assertIn("old_target_remains", reasons)

    def test_skip_is_terminal_but_not_accepted(self):
        accepted, reasons, diagnostics = validate_response(
            self.jewelry_plan(), {"status": "skip", "reason": "unsafe mutation"}
        )
        self.assertFalse(accepted)
        self.assertEqual(reasons, [])
        self.assertEqual(diagnostics["model_status"], "skip")


if __name__ == "__main__":
    unittest.main()
