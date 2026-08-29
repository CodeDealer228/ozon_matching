from types import SimpleNamespace

import pandas as pd

from repair_token_replacement_identity_bundles import (
    apply_mappings,
    parse_decisions,
    span_candidates,
)


def test_span_candidates() -> None:
    spans = span_candidates("matador mp 30 sibir ice 2")
    assert "matador" in spans
    assert "mp 30" in spans
    assert "sibir ice 2" in spans


def test_parse_is_restricted_to_offered_spans() -> None:
    row = {
        "source_spans": ["samsung", "galaxy a14"],
        "target_spans": ["xiaomi", "redmi note 13"],
        "source_token": "samsung",
        "replacement_token": "xiaomi",
    }
    message = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(
        name="submit_identity_bundle_repairs",
        arguments=(
            '{"decisions":[{"slot":0,"status":"repair","mappings":['
            '{"source_span":"samsung","target_span":"xiaomi"},'
            '{"source_span":"galaxy a14","target_span":"redmi note 13"}],'
            '"reason":"grounded"}]}'
        ),
    ))])
    decisions = parse_decisions(message, [row])
    assert decisions[0]["status"] == "repair"
    assert len(decisions[0]["mappings"]) == 2


def test_apply_mappings_preserves_unrelated_specs() -> None:
    item = pd.Series({
        "name": "samsung galaxy a14 4/128 гб, черный",
        "attributes": (
            '{"бренд":"samsung","модель":"galaxy a14",'
            '"встроенная память":"128 гб","цвет товара":"черный"}'
        ),
    })
    repaired = apply_mappings(item, [
        {"source_span": "galaxy a14", "target_span": "redmi note 13"},
        {"source_span": "samsung", "target_span": "xiaomi"},
    ])
    assert repaired["name"] == "xiaomi redmi note 13 4/128 гб, черный"
    assert repaired["attributes"]["бренд"] == "xiaomi"
    assert repaired["attributes"]["модель"] == "redmi note 13"
    assert repaired["attributes"]["встроенная память"] == "128 гб"
    assert repaired["attributes"]["цвет товара"] == "черный"


if __name__ == "__main__":
    test_span_candidates()
    test_parse_is_restricted_to_offered_spans()
    test_apply_mappings_preserves_unrelated_specs()
