"""Evidence-search helper for token-replacement counterfactual generation.

The module deliberately accepts only a redacted public-search context. Product
IDs, labels, full item cards, and validation rows do not belong in this API.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse


SEARCH_SCHEMA_VERSION = 1
SEARCH_STATUS = {"found", "skip"}
SEMANTIC_TYPES = {
    "brand",
    "model",
    "article",
    "size_or_quantity",
    "compatibility_code",
    "other",
}
WHITESPACE_RE = re.compile(r"\s")

RESEARCH_SYSTEM_PROMPT = """Ты — исследователь реальных товарных значений для
контрфактуальной генерации marketplace-пар. Твоя задача — найти в открытом
интернете правдоподобные замены одного токена, не редактируя исходную пару и не
решая за финальный генератор.

Правила:
1. Установи смысловой тип source_token: brand, model, article,
   size_or_quantity, compatibility_code или other.
2. Замена обязана сохранять смысловой тип.
3. Кандидат должен реально относиться к product_type_hint.
4. Предпочитай официальные страницы производителя, каталоги и крупные
   marketplace/retailer pages. Достаточно одной официальной страницы либо двух
   независимых товарных страниц.
5. Не предлагай алиас, другую транслитерацию или вариант регистра source_token.
6. Не придумывай факты. Если доказательств нет или тип неясен, верни skip.
7. replacement_token — один токен без пробелов для буквальной замены.
8. Выполняй узкие запросы по классу товара и типу значения. Не ищи приватную
   исходную карточку целиком.
9. Верни только валидный JSON по заданному контракту. Сохрани URL, title и
   короткое собственное изложение доказательства, без длинных цитат.
"""


def build_research_payload(
    *,
    category: str,
    product_type_hint: str,
    source_token: str,
    semantic_type_hint: str = "other",
    max_candidates: int = 5,
) -> dict[str, object]:
    """Build the only payload allowed to reach the web-search model."""
    fields = {
        "category": category.strip(),
        "product_type_hint": product_type_hint.strip(),
        "source_token": source_token.strip(),
        "semantic_type_hint": semantic_type_hint.strip(),
    }
    if any(not value for value in fields.values()):
        raise ValueError("research_context_field_empty")
    if fields["semantic_type_hint"] not in SEMANTIC_TYPES:
        raise ValueError("invalid_semantic_type_hint")
    if not 1 <= max_candidates <= 10:
        raise ValueError("max_candidates_out_of_range")
    if WHITESPACE_RE.search(fields["source_token"]):
        raise ValueError("source_token_contains_whitespace")
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        **fields,
        "max_candidates": max_candidates,
        "output_contract": {
            "status": "found|skip",
            "semantic_type": "brand|model|article|size_or_quantity|compatibility_code|other",
            "queries": ["string"],
            "candidates": [
                {
                    "replacement_token": "single token",
                    "reason": "short explanation",
                    "evidence": [
                        {"url": "https://...", "title": "string", "claim": "short paraphrase"}
                    ],
                }
            ],
            "skip_reason": "string",
        },
    }


def build_research_input(payload: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Верни JSON.\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _validate_https_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("evidence_url_not_string")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("evidence_url_not_https")
    return value


def validate_research_brief(
    value: object,
    *,
    source_token: str,
    max_candidates: int,
) -> dict[str, object]:
    """Fail closed on malformed evidence while keeping semantic scores soft."""
    if not isinstance(value, dict):
        raise ValueError("research_brief_not_object")
    expected = {"status", "semantic_type", "queries", "candidates", "skip_reason"}
    if set(value) != expected:
        raise ValueError("research_brief_fields_mismatch")
    status = value["status"]
    semantic_type = value["semantic_type"]
    queries = value["queries"]
    candidates = value["candidates"]
    skip_reason = value["skip_reason"]
    if status not in SEARCH_STATUS:
        raise ValueError("invalid_research_status")
    if semantic_type not in SEMANTIC_TYPES:
        raise ValueError("invalid_research_semantic_type")
    if not isinstance(queries, list) or any(not isinstance(query, str) for query in queries):
        raise ValueError("invalid_research_queries")
    if not isinstance(candidates, list) or len(candidates) > max_candidates:
        raise ValueError("invalid_research_candidates")
    if not isinstance(skip_reason, str):
        raise ValueError("invalid_research_skip_reason")
    if status == "skip":
        if candidates or not skip_reason.strip():
            raise ValueError("invalid_research_skip")
        return value
    if not candidates or skip_reason.strip():
        raise ValueError("invalid_research_found")

    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "replacement_token",
            "reason",
            "evidence",
        }:
            raise ValueError("invalid_research_candidate_object")
        replacement = candidate["replacement_token"]
        reason = candidate["reason"]
        evidence = candidate["evidence"]
        if not isinstance(replacement, str) or not replacement or len(replacement) > 64:
            raise ValueError("invalid_research_replacement")
        if WHITESPACE_RE.search(replacement):
            raise ValueError("research_replacement_contains_whitespace")
        if replacement.casefold() == source_token.casefold():
            raise ValueError("research_replacement_does_not_change_token")
        if replacement.casefold() in seen:
            raise ValueError("duplicate_research_replacement")
        seen.add(replacement.casefold())
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError("invalid_research_reason")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 5:
            raise ValueError("invalid_research_evidence")
        for source in evidence:
            if not isinstance(source, dict) or set(source) != {"url", "title", "claim"}:
                raise ValueError("invalid_research_evidence_object")
            _validate_https_url(source["url"])
            if any(
                not isinstance(source[field], str) or not source[field].strip()
                for field in ("title", "claim")
            ):
                raise ValueError("invalid_research_evidence_text")
            if len(source["claim"]) > 500:
                raise ValueError("research_evidence_claim_too_long")
    return value


def parse_research_output(
    text: str,
    *,
    source_token: str,
    max_candidates: int,
) -> dict[str, object]:
    """Parse exact JSON; fenced or prose-wrapped responses are terminal errors."""
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("research_output_not_json") from exc
    return validate_research_brief(
        value,
        source_token=source_token,
        max_candidates=max_candidates,
    )


def run_alibaba_web_research(
    client: Any,
    *,
    model: str,
    payload: dict[str, object],
) -> tuple[dict[str, object], object]:
    """Run Alibaba Responses built-in search; caller persists the raw response."""
    response = client.responses.create(
        model=model,
        input=build_research_input(payload),
        tools=[{"type": "web_search"}, {"type": "web_extractor"}],
        extra_body={"enable_thinking": True},
    )
    brief = parse_research_output(
        response.output_text,
        source_token=str(payload["source_token"]),
        max_candidates=int(payload["max_candidates"]),
    )
    return brief, response
