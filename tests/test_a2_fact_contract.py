from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from a2_fact_contract import (
    _mask_forbidden,
    canonical_identifiers,
    canonical_numeric_mentions,
    validate_candidate,
)


class NumericCanonicalizationTests(unittest.TestCase):
    def test_unit_spacing_is_equivalent(self) -> None:
        self.assertEqual(canonical_numeric_mentions("37-38см"), canonical_numeric_mentions("37-38 см"))
        self.assertEqual(canonical_numeric_mentions("20ml"), canonical_numeric_mentions("20 ml"))

    def test_leading_zero_is_preserved(self) -> None:
        self.assertNotEqual(canonical_numeric_mentions("007 мм"), canonical_numeric_mentions("7 мм"))

    def test_quantity_is_canonicalized(self) -> None:
        self.assertEqual(canonical_numeric_mentions("1шт"), canonical_numeric_mentions("1 шт"))

    def test_model_identifier_is_preserved(self) -> None:
        self.assertIn("g11", canonical_identifiers("Модель G11"))


class CandidateValidationTests(unittest.TestCase):
    def test_mask_uses_only_fact_visible_in_name(self) -> None:
        masked = _mask_forbidden(
            "Белый, васильковый халат",
            [{"concept": "product_color", "name_representation": "белый"}],
        )
        self.assertEqual(masked, "<product_color_A_ONLY>, васильковый халат")


    def test_rejects_copied_a_only_number(self) -> None:
        contract = {
            "name2": "Товар B 20 мл",
            "safe_shared_facts": [{"fact_id": "shared.brand.x"}],
            "allowed_operations": [{"operation_id": "insert.shared.brand.x"}],
            "protected_numeric_mentions": ["20ml"],
            "forbidden_numeric_mentions": ["30ml"],
            "protected_identifiers": [],
            "forbidden_identifiers": [],
        }
        ok, reasons, _ = validate_candidate(
            contract,
            "Бренд Товар B 20 мл 30 мл",
            ["shared.brand.x"],
            ["insert.shared.brand.x"],
        )
        self.assertFalse(ok)
        self.assertIn("copied_numeric_fact_from_item1", reasons)

    def test_accepts_unit_spacing_change_with_allowed_operation(self) -> None:
        contract = {
            "name2": "Товар B 37-38см",
            "safe_shared_facts": [{"fact_id": "shared.brand.x"}],
            "allowed_operations": [{"operation_id": "insert.shared.brand.x"}],
            "protected_numeric_mentions": ["37-38cm"],
            "forbidden_numeric_mentions": [],
            "protected_identifiers": [],
            "forbidden_identifiers": [],
        }
        ok, reasons, _ = validate_candidate(
            contract,
            "Бренд Товар B 37-38 см",
            ["shared.brand.x"],
            ["insert.shared.brand.x"],
        )
        self.assertTrue(ok, reasons)


if __name__ == "__main__":
    unittest.main()
