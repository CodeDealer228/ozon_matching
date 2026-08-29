import json

from token_replacement_web_search_tool import (
    build_research_input,
    build_research_payload,
    parse_research_output,
    validate_research_brief,
)


payload = build_research_payload(
    category="Бытовая техника",
    product_type_hint="робот-пылесос",
    source_token="xiaomi",
    semantic_type_hint="brand",
)
assert set(payload) == {
    "schema_version",
    "category",
    "product_type_hint",
    "source_token",
    "semantic_type_hint",
    "max_candidates",
    "output_contract",
}
assert "Верни JSON" in build_research_input(payload)[1]["content"]

brief = {
    "status": "found",
    "semantic_type": "brand",
    "queries": ["робот-пылесос бренды производители"],
    "candidates": [
        {
            "replacement_token": "dreame",
            "reason": "реальный производитель роботов-пылесосов",
            "evidence": [
                {
                    "url": "https://example.org/dreame-robot",
                    "title": "Dreame robot vacuum",
                    "claim": "страница описывает робот-пылесос Dreame",
                }
            ],
        }
    ],
    "skip_reason": "",
}
assert validate_research_brief(brief, source_token="xiaomi", max_candidates=5) == brief
assert parse_research_output(
    json.dumps(brief, ensure_ascii=False),
    source_token="xiaomi",
    max_candidates=5,
)["status"] == "found"

skip = {
    "status": "skip",
    "semantic_type": "other",
    "queries": [],
    "candidates": [],
    "skip_reason": "тип токена не удалось установить",
}
assert validate_research_brief(skip, source_token="x1", max_candidates=5)["status"] == "skip"

for bad, expected in (
    ({**brief, "candidates": [{**brief["candidates"][0], "replacement_token": "two words"}]}, "research_replacement_contains_whitespace"),
    ({**brief, "candidates": [{**brief["candidates"][0], "replacement_token": "XIAOMI"}]}, "research_replacement_does_not_change_token"),
    ({**brief, "candidates": [{**brief["candidates"][0], "evidence": [{"url": "http://example.org", "title": "x", "claim": "y"}]}]}, "evidence_url_not_https"),
):
    try:
        validate_research_brief(bad, source_token="xiaomi", max_candidates=5)
    except ValueError as exc:
        assert str(exc) == expected
    else:
        raise AssertionError(f"invalid brief passed: {expected}")
