from types import SimpleNamespace

import pandas as pd

from judge_token_replacement_grounded_bundles import (
    apply_bundle,
    build_bundle,
    identity_fields,
    identity_kind,
    local_bundle_rejection,
    parse_decisions,
)


def item(name: str, attributes: dict[str, str]) -> pd.Series:
    import json

    return pd.Series({"name": name, "attributes": json.dumps(attributes, ensure_ascii=False)})


def test_identity_kind_excludes_compatibility_model() -> None:
    assert identity_kind("бренд") == "brand"
    assert identity_kind("модель") == "model"
    assert identity_kind("партномер (артикул производителя)") == "article"
    assert identity_kind("модель автомобиля") is None
    assert identity_kind("объем памяти") is None


def test_apply_bundle_never_changes_specs() -> None:
    source = item(
        "samsung galaxy a14 4/128 черный",
        {
            "бренд": "samsung",
            "модель": "galaxy a14",
            "совместимые модели": "samsung galaxy a14",
            "объем памяти": "128 гб",
            "цвет товара": "черный",
        },
    )
    result = apply_bundle(source, [
        {"kind": "model", "source_span": "galaxy a14", "target_span": "redmi note 13"},
        {"kind": "brand", "source_span": "samsung", "target_span": "xiaomi"},
    ])
    assert result["name"] == "xiaomi redmi note 13 4/128 черный"
    assert result["attributes"]["бренд"] == "xiaomi"
    assert result["attributes"]["модель"] == "redmi note 13"
    assert result["attributes"]["совместимые модели"] == "samsung galaxy a14"
    assert result["attributes"]["объем памяти"] == "128 гб"


def test_brand_bundle_uses_grounded_model_and_article() -> None:
    left = item("samsung galaxy a14 sm-a145", {
        "бренд": "samsung", "модель": "galaxy a14", "артикул": "sm-a145"
    })
    right = item("samsung galaxy a14 sm-a145", {
        "бренд": "samsung", "модель": "galaxy a14", "артикул": "sm-a145"
    })
    target = item("xiaomi redmi note 13 23129rai4g", {
        "бренд": "xiaomi", "модель": "redmi note 13", "артикул": "23129rai4g"
    })
    row = {
        "operation_sha256": "a" * 64,
        "source_role": "brand",
        "source_token": "samsung",
        "replacement_token": "xiaomi",
    }
    bundle = build_bundle(row, left, right, target)
    assert bundle is not None
    mappings = {
        (value["source_span"], value["target_span"]) for value in bundle["mappings"]
    }
    assert ("samsung", "xiaomi") in mappings
    assert ("galaxy a14", "redmi note 13") in mappings
    assert ("sm-a145", "23129rai4g") in mappings


def test_model_can_only_choose_offered_bundle() -> None:
    row = {"bundles": [{"bundle_id": "bundle_ok"}]}
    message = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(
        name="select_grounded_identity_bundles",
        arguments=(
            '{"decisions":[{"slot":0,"status":"choose",'
            '"bundle_id":"bundle_ok","reason":"coherent"}]}'
        ),
    ))])
    assert parse_decisions(message, [row])[0]["bundle_id"] == "bundle_ok"


def test_brand_bundle_rejects_old_alias_and_unsupported_line() -> None:
    left = item("xiaomi mi mini backpack", {"бренд": "xiaomi"})
    right = item("рюкзак ксиоми mi mini", {"бренд": "xiaomi"})
    bundle = {
        "target_card": {"name": "рюкзак piquadro urban", "attributes": {"бренд": "piquadro"}},
        "mappings": [{"kind": "brand", "source_span": "xiaomi", "target_span": "piquadro"}],
        "preview_name_1": "piquadro mi mini backpack",
        "preview_name_2": "рюкзак ксиоми mi mini",
    }
    row = {"source_role": "brand", "source_token": "xiaomi", "replacement_token": "piquadro"}
    assert local_bundle_rejection(row, bundle, left, right) is not None


def test_bundle_rejects_grounded_target_spec_conflict() -> None:
    left = item("kaimana uk-21 ppm фиолетовая", {"бренд": "kaimana", "модель": "uk-21 ppm", "цвет товара": "фиолетовый"})
    right = item("укулеле kaimana uk-21 ppm", {"бренд": "kaimana", "модель": "uk-21 ppm", "цвет товара": "фиолетовый"})
    bundle = {
        "target_card": {"name": "kaimana uk-21 pkm розовая", "attributes": {"бренд": "kaimana", "модель": "uk-21 pkm", "цвет товара": "розовый"}},
        "mappings": [{"kind": "model", "source_span": "uk-21 ppm", "target_span": "uk-21 pkm"}],
        "preview_name_1": "kaimana uk-21 pkm фиолетовая",
        "preview_name_2": "укулеле kaimana uk-21 pkm",
    }
    row = {"source_role": "article", "source_token": "uk-21 ppm", "replacement_token": "uk-21 pkm"}
    assert local_bundle_rejection(row, bundle, left, right) == "grounded_target_spec_conflict:цвет товара"


if __name__ == "__main__":
    test_identity_kind_excludes_compatibility_model()
    test_apply_bundle_never_changes_specs()
    test_brand_bundle_uses_grounded_model_and_article()
    test_model_can_only_choose_offered_bundle()
    test_brand_bundle_rejects_old_alias_and_unsupported_line()
    test_bundle_rejects_grounded_target_spec_conflict()
