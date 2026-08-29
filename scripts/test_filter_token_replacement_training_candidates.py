import json

import pandas as pd

from filter_token_replacement_training_candidates import (
    mutate_spec_item,
    mutation_rejection,
)


def item(name: str, attributes: dict[str, str]) -> pd.Series:
    return pd.Series({
        "name": name,
        "category": "Тест",
        "attributes": json.dumps(attributes, ensure_ascii=False),
    })


def test_role_aware_mutation_preserves_unrelated_numbers() -> None:
    source = item(
        "набор 28 шт 220 в",
        {"количество, штук": "28", "напряжение": "220 в"},
    )
    changed, count = mutate_spec_item(source, "count", "28", "14", -1)
    assert changed["name"] == "набор 14 шт 220 в"
    attributes = json.loads(changed["attributes"])
    assert attributes["количество, штук"] == "14"
    assert attributes["напряжение"] == "220 в"
    assert count == 2


def test_brand_is_never_kept_as_safe_spec() -> None:
    left = item("samsung phone", {"бренд": "samsung"})
    right = item("phone samsung", {"бренд": "samsung"})
    role, rejection = mutation_rejection(left, right, "samsung", "xiaomi")
    assert role == "brand"
    assert rejection == "identity_or_unsupported_role"


if __name__ == "__main__":
    test_role_aware_mutation_preserves_unrelated_numbers()
    test_brand_is_never_kept_as_safe_spec()
