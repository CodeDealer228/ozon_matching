from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "a2_scale", ROOT / "scripts/generate_a2_scale_200k.py"
)
assert SPEC and SPEC.loader
a2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a2)


def row(name1: str, name2: str, attributes2: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(name1=name1, name2=name2, attributes2=attributes2)


def test_accepts_natural_reordering_and_morphology() -> None:
    source = row(
        "Женские зимние ботинки чёрные",
        "Ботинки из натуральной кожи женские размер 38",
        '{"Материал":"Натуральная кожа","Размер":"38"}',
    )
    accepted, reasons, _ = a2.validate_candidate(
        source, "Женские кожаные ботинки из натуральной кожи, размер 38"
    )
    assert accepted, reasons


def test_rejects_lost_number() -> None:
    source = row("Кабель USB 2 м", "Кабель Type-C 1 м")
    accepted, reasons, _ = a2.validate_candidate(source, "Кабель Type-C")
    assert not accepted
    assert "lost_numeric_fact" in reasons


def test_rejects_a_only_model_copy() -> None:
    source = row("Чехол Samsung A55", "Чехол Xiaomi Redmi Note 13")
    accepted, reasons, _ = a2.validate_candidate(
        source, "Чехол Xiaomi Redmi Note 13 для Samsung A55"
    )
    assert not accepted
    assert "copied_a_only_identifier" in reasons


def test_parse_one_or_two_candidates() -> None:
    parsed = a2.parse_response(
        '{"status":"ok","candidates":[{"name2_prime":"Первый"},{"name2_prime":"Второй"}]}'
    )
    assert parsed == {"status": "ok", "candidates": ["Первый", "Второй"]}


def test_provider_payload_has_attributes_b_but_no_ids() -> None:
    source = SimpleNamespace(
        category="Электроника",
        id1=111,
        id2=222,
        name1="Кабель USB 2 м",
        name2="Кабель Type-C 1 м",
        attributes2='{"Длина":"1 м"}',
    )
    messages = a2.build_messages(source, explicit_cache=False)
    payload = messages[-1]["content"]
    assert "attributes_b" in payload
    assert "111" not in payload
    assert "222" not in payload


def test_seed_is_within_signed_32_bit_range() -> None:
    seed = a2.bounded_seed("source", "qwen3.6-27b")
    assert 0 <= seed < 2_147_483_647


def test_prompt_forbids_merging_separate_ingredients() -> None:
    source = SimpleNamespace(
        category="Товары для животных",
        name1="Корм с перепелкой",
        name2="Корм с курицей, с печенью",
        attributes2="{}",
    )
    system = a2.build_messages(source, explicit_cache=False)[0]["content"]
    assert "с куриной печенью" in system


def test_rejects_unsupported_content_word_copied_from_a() -> None:
    source = row(
        "Лонгслив женский оверсайз",
        "Кардиган Koton",
        '{"Пол":"женский","Цвет":"черный"}',
    )
    accepted, reasons, _ = a2.validate_candidate(source, "Кардиган Koton женский оверсайз")
    assert not accepted
    assert "copied_a_only_unsupported_word" in reasons


def test_accepts_content_word_supported_by_b_attributes() -> None:
    source = row(
        "Лонгслив женский черный",
        "Кардиган Koton",
        '{"Пол":"женский","Цвет":"черный"}',
    )
    accepted, reasons, _ = a2.validate_candidate(source, "Кардиган Koton женский черный")
    assert accepted, reasons


def test_scheduler_prioritizes_categories_with_fewer_attempts() -> None:
    categories = ["A", "B", "C", "D"]
    selected = a2.choose_balanced_categories(
        categories,
        Counter({"A": 300, "B": 290, "C": 5, "D": 6}),
        Counter({"A": 500, "B": 500, "C": 10, "D": 11}),
        Counter(),
        target_per_category=10_000,
        max_new_sources_per_category=None,
        limit=2,
    )
    assert selected == ["C", "D"]
