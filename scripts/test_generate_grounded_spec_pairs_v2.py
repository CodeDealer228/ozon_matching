import json

import pandas as pd

from analyze_spec_replacement_capacity import strict_source_role
from generate_grounded_spec_pairs_v2 import balanced_select, collect_sources, materialize


def test_collect_and_materialize_strict_volume() -> None:
    items = pd.DataFrame([
        {
            "id": 1,
            "category": "beauty",
            "name": "шампунь 500 мл",
            "attributes": json.dumps({"тип": "шампунь", "объем": "500 мл"}),
        },
        {
            "id": 2,
            "category": "beauty",
            "name": "шампунь для волос 500 мл",
            "attributes": json.dumps({"тип": "шампунь", "объём": "500 мл"}),
        },
    ]).set_index("id", drop=False)
    pairs = pd.DataFrame([
        {"id1": 1, "id2": 2, "category": "beauty"},
    ])
    sources = collect_sources(pairs, items)
    assert sources[0]["offered_tokens"] == ["500"]
    operation = {
        **sources[0],
        "source_token": "500",
        "replacement_token": "250",
        "source_role": "volume",
        "operation_sha256": "a" * 64,
        "evidence_name": "шампунь 250 мл",
        "bank_frequency": 2,
        "context_overlap": 2,
        "title_context_overlap": 2,
    }
    synthetic_items, synthetic_pairs, _ = materialize([operation], items)
    assert len(synthetic_items) == 2
    assert sorted(synthetic_pairs["target"].tolist()) == [0, 1]
    assert all("250" in value for value in synthetic_items["name"])


def test_balanced_select_round_robin() -> None:
    rows = [
        {"category": category, "context_overlap": 2, "bank_frequency": 1,
         "operation_sha256": f"{category}-{index}"}
        for category in ("a", "b")
        for index in range(3)
    ]
    selected = balanced_select(rows, 4)
    assert [row["category"] for row in selected] == ["a", "b", "a", "b"]


def test_rejects_duplicate_equivalent_volume_in_title() -> None:
    row = pd.Series({
        "name": "tattoo ink 4 oz - 120 мл",
        "attributes": json.dumps({"объем": "120 мл"}),
    })
    assert strict_source_role(row, row, "120") is None


if __name__ == "__main__":
    test_collect_and_materialize_strict_volume()
    test_balanced_select_round_robin()
    test_rejects_duplicate_equivalent_volume_in_title()
