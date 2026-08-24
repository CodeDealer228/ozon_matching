"""Deterministic fact contract for A2 v3 synthetic hard negatives.

The module is deliberately local-only: it extracts typed facts from the
canonical human-train source rows, builds safe operations for Qwen and validates
candidate rewrites.  It never reads validation examples and never calls an API.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import analyze_synthetic_negative_rules as rules


UNIT_ALIASES = {
    "мм": "mm", "mm": "mm", "см": "cm", "cm": "cm",
    "мл": "ml", "ml": "ml", "л": "l", "l": "l",
    "гр": "g", "г": "g", "g": "g", "кг": "kg", "kg": "kg",
    "шт": "pcs", "pcs": "pcs", "%": "%",
}
MEASURE_RE = re.compile(
    r"(?<![0-9a-zа-я])"
    r"(\d+(?:[.,]\d+)?(?:\s*[-–—]\s*\d+(?:[.,]\d+)?)?)"
    r"\s*(мм|mm|см|cm|мл|ml|л|l|гр|г|g|кг|kg|шт|pcs|%)?"
    r"(?![0-9a-zа-я])",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(
    r"(?<![0-9a-zа-я])(?=[0-9a-zа-я-]{3,})(?=[0-9a-zа-я-]*\d)"
    r"(?=[0-9a-zа-я-]*[a-zа-я])[0-9a-zа-я]+(?:-[0-9a-zа-я]+)*(?![0-9a-zа-я])",
    re.IGNORECASE,
)

ANCHOR_WHITELIST = {
    "Обувь": {
        "brand", "article", "product_model", "product_color", "russian_size",
        "manufacturer_size", "generic_size", "foot_length", "insole_length",
    },
    "Ювелирные изделия": {
        "brand", "article", "product_model", "product_color", "stone_color",
        "item_size", "ring_size", "metal_fineness", "main_insert", "stone_count",
    },
    "Одежда": {
        "brand", "article", "product_model", "product_color", "russian_size",
        "manufacturer_size", "generic_size", "jeans_size", "height_size", "package_count",
    },
    "Автотовары": {
        "brand", "article", "oem_number", "product_model", "product_color",
        "package_count", "compatibility", "car_make", "car_model", "car_generation",
        "car_year", "volume", "diameter", "length", "width", "height", "power",
        "tire_width", "tire_profile", "tire_diameter",
    },
}

IDENTITY_SENSITIVE = set().union(*ANCHOR_WHITELIST.values()) | {
    "completeness", "dimensions", "item_weight", "average_weight", "placement", "engine",
}

# The first paid pilot uses only concepts whose value can be rendered without
# inventing grammar or expanding a broad, noisy attribute.  In particular,
# completeness/composition and dimensions are intentionally excluded.
SAFE_INSERT_CONCEPTS = {
    "brand", "product_color", "gender", "target_audience", "season", "seasonality",
    "item_type", "metal_fineness", "jewelry_material", "main_insert",
}


def stable_hash(*values: object) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_surface(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace("ё", "е")
    text = re.sub(r"[–—]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_numeric_mentions(value: object) -> set[str]:
    """Return unit-aware numbers without losing leading zeroes or ranges."""
    text = normalize_surface(value)
    result: set[str] = set()
    for match in MEASURE_RE.finditer(text):
        number = re.sub(r"\s*[-–—]\s*", "-", match.group(1)).replace(",", ".")
        raw_unit = (match.group(2) or "").lower()
        unit = UNIT_ALIASES.get(raw_unit, raw_unit)
        result.add(f"{number}{unit}")
    return result


def canonical_identifiers(value: object) -> set[str]:
    text = normalize_surface(value)
    return {match.group(0).lower() for match in IDENTIFIER_RE.finditer(text)}


def _raw_display(entry: dict[str, Any], canonical: str) -> str:
    concept = str(entry.get("concept", ""))
    if concept in rules.IDENTIFIER_CONCEPTS:
        return canonical
    candidates: list[str] = []
    for raw in entry.get("raw", []):
        if canonical in rules.canonicalize_value(str(entry.get("concept", "")), raw):
            candidates.append(str(raw).strip())
    if candidates:
        return min(candidates, key=lambda value: (len(value), value.casefold()))
    return canonical


def _fact_id(side: str, concept: str, canonical: str) -> str:
    return f"{side}.{concept}.{stable_hash(concept, canonical)[:12]}"


@lru_cache(maxsize=None)
def _extract_side_cached(category: str, attributes_raw: str, name: str, side: str) -> tuple[dict[str, Any], ...]:
    if category not in rules.SEMANTIC_ALIASES:
        return ()
    semantics = rules.extract_semantics(category, rules.parse_attributes(attributes_raw))
    facts: list[dict[str, Any]] = []
    for concept, entry in sorted(semantics.items()):
        enriched = dict(entry)
        enriched["concept"] = concept
        values = set(entry["values"])
        for canonical in sorted(values):
            representation = rules.find_name_representation(
                concept, list(entry["raw"]), values, str(name)
            )
            visible = representation is not None
            # Existing helper misses unitless surface forms such as name
            # ``170-176`` vs attribute ``170-176 см``.  Treat an equal numeric
            # core as already visible so it never becomes a paid operation.
            if not visible and concept in rules.DIMENSION_CONCEPTS:
                name_numbers = canonical_numeric_mentions(name)
                canonical_number = re.sub(r"(?:mm|cm|ml|l|g|kg|pcs|%)$", "", canonical)
                visible = any(
                    re.sub(r"(?:mm|cm|ml|l|g|kg|pcs|%)$", "", value) == canonical_number
                    for value in name_numbers
                )
            facts.append({
                "fact_id": _fact_id(side, concept, canonical),
                "side": side,
                "concept": concept,
                "canonical_value": canonical,
                "display_value": _raw_display(enriched, canonical),
                "source": f"attributes_{side}",
                "source_keys": sorted(str(value) for value in entry["keys"]),
                "visible_in_name": visible,
                "name_representation": representation[1] if representation else None,
                "confidence": "high",
                "identity_sensitivity": "high" if concept in IDENTITY_SENSITIVE else "medium",
            })
    return tuple(facts)


def _extract_side(category: str, attributes_raw: object, name: str, side: str) -> list[dict[str, Any]]:
    # Items participate in multiple train pairs.  Attribute parsing and
    # semantic extraction dominate a full-corpus scan, so cache immutable
    # per-item results and return only a shallow list wrapper to callers.
    return list(_extract_side_cached(category, str(attributes_raw or ""), str(name), side))


def _by_concept(facts: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for fact in facts:
        result.setdefault(str(fact["concept"]), {})[str(fact["canonical_value"])] = fact
    return result


def _mask_forbidden(name1: str, forbidden: list[dict[str, Any]]) -> str:
    masked = str(name1)
    replacements: list[tuple[str, str]] = []
    for fact in forbidden:
        display = str(fact.get("name_representation") or "").strip()
        if len(display) >= 2:
            replacements.append((display, f"<{fact['concept']}_A_ONLY>"))
    for source, replacement in sorted(replacements, key=lambda value: len(value[0]), reverse=True):
        masked = re.sub(
            rf"(?<![0-9a-zа-я]){re.escape(source)}(?![0-9a-zа-я])",
            replacement,
            masked,
            flags=re.IGNORECASE,
        )
    return masked


def build_fact_contract(row: dict[str, Any]) -> dict[str, Any]:
    category = str(row["category"])
    a_facts = _extract_side(category, row.get("attributes1", ""), str(row["name1"]), "a")
    b_facts = _extract_side(category, row.get("attributes2", ""), str(row["name2"]), "b")
    a_by = _by_concept(a_facts)
    b_by = _by_concept(b_facts)

    shared: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []

    for concept in sorted(set(a_by) | set(b_by)):
        a_values = set(a_by.get(concept, {}))
        b_values = set(b_by.get(concept, {}))
        for canonical in sorted(a_values & b_values):
            a_fact = a_by[concept][canonical]
            b_fact = b_by[concept][canonical]
            shared.append({
                "fact_id": f"shared.{concept}.{stable_hash(concept, canonical)[:12]}",
                "concept": concept,
                "canonical_value": canonical,
                "display_value": b_fact["display_value"],
                "surface_value_from_a": a_fact.get("name_representation"),
                "visible_in_name_a": a_fact["visible_in_name"],
                "visible_in_name_b": b_fact["visible_in_name"],
                "confidence": "high",
            })
        for canonical in sorted(a_values - b_values):
            fact = dict(a_by[concept][canonical])
            fact["fact_id"] = f"forbidden.{concept}.{stable_hash(concept, canonical)[:12]}"
            forbidden.append(fact)
        if concept in ANCHOR_WHITELIST.get(category, set()) and a_values and b_values and not (a_values & b_values):
            anchors.append({
                "anchor_id": f"anchor.{concept}.{stable_hash(sorted(a_values), sorted(b_values))[:12]}",
                "concept": concept,
                "a_values": sorted(a_values),
                "b_values": sorted(b_values),
                "confidence": "high",
            })

    for fact in b_facts:
        if fact["visible_in_name"] and fact["concept"] in IDENTITY_SENSITIVE:
            protected.append(fact)

    protected_numeric = sorted(canonical_numeric_mentions(row["name2"]))
    protected_identifiers = sorted(canonical_identifiers(row["name2"]))
    forbidden_numeric = sorted(
        canonical_numeric_mentions(row["name1"])
        - canonical_numeric_mentions(row["name2"])
        - canonical_numeric_mentions(row.get("attributes2", ""))
    )
    forbidden_identifiers = sorted(
        canonical_identifiers(row["name1"])
        - canonical_identifiers(row["name2"])
        - canonical_identifiers(row.get("attributes2", ""))
    )

    operations: list[dict[str, Any]] = []
    operation_keys: set[tuple[str, str]] = set()
    for fact in shared:
        surface_value = str(fact.get("surface_value_from_a") or "").strip()
        operation_key = (str(fact["concept"]), normalize_surface(surface_value))
        if (
            fact["concept"] in SAFE_INSERT_CONCEPTS
            and fact["visible_in_name_a"]
            and not fact["visible_in_name_b"]
            and len(surface_value) >= 2
            and operation_key not in operation_keys
        ):
            operation_keys.add(operation_key)
            operations.append({
                "operation_id": f"insert.{fact['fact_id']}",
                "type": "insert_shared_fact",
                "fact_id": fact["fact_id"],
                "concept": fact["concept"],
                # Render only the exact surface already present in A.  The
                # fact's canonical value is independently confirmed by B,
                # avoiding broad raw attributes such as a whole color list.
                "display_value": surface_value,
            })
    if operations and anchors:
        tier, reason = "A", "safe_operation_and_reliable_negative_anchor"
    elif operations:
        tier, reason = "B", "safe_operation_without_reliable_negative_anchor"
    else:
        tier, reason = "C", "no_safe_nontrivial_operation"

    contract = {
        "contract_version": 3,
        "source_pair_sha256": str(row["source_pair_sha256"]),
        "category": category,
        "tier": tier,
        "selection_reason": reason,
        "a_safe_template": _mask_forbidden(str(row["name1"]), forbidden),
        "name2": str(row["name2"]),
        "safe_shared_facts": shared,
        "negative_anchors": anchors,
        "protected_b_facts": protected,
        "protected_numeric_mentions": protected_numeric,
        "protected_identifiers": protected_identifiers,
        "forbidden_a_only_facts": forbidden,
        "forbidden_numeric_mentions": forbidden_numeric,
        "forbidden_identifiers": forbidden_identifiers,
        "allowed_operations": operations,
    }
    contract["contract_sha256"] = stable_hash(contract)
    return contract


def validate_candidate(
    contract: dict[str, Any],
    candidate: str,
    used_fact_ids: Iterable[str],
    applied_operation_ids: Iterable[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    candidate = str(candidate).strip()
    if not candidate:
        reasons.append("empty_rewrite")
    if len(candidate) > 500:
        reasons.append("rewrite_too_long")
    if rules.normalize_text(candidate) == rules.normalize_text(contract["name2"]):
        reasons.append("trivial_normalization")

    used = set(map(str, used_fact_ids))
    applied = set(map(str, applied_operation_ids))
    known_facts = {str(value["fact_id"]) for value in contract["safe_shared_facts"]}
    known_ops = {str(value["operation_id"]) for value in contract["allowed_operations"]}
    if not used.issubset(known_facts):
        reasons.append("unknown_used_fact_id")
    if not applied or not applied.issubset(known_ops):
        reasons.append("unknown_or_missing_operation_id")

    candidate_numbers = canonical_numeric_mentions(candidate)
    missing_numbers = sorted(set(contract["protected_numeric_mentions"]) - candidate_numbers)
    copied_numbers = sorted(set(contract["forbidden_numeric_mentions"]) & candidate_numbers)
    if missing_numbers:
        reasons.append("lost_numeric_fact")
    if copied_numbers:
        reasons.append("copied_numeric_fact_from_item1")

    candidate_identifiers = canonical_identifiers(candidate)
    missing_ids = sorted(set(contract["protected_identifiers"]) - candidate_identifiers)
    copied_ids = sorted(set(contract["forbidden_identifiers"]) & candidate_identifiers)
    if missing_ids:
        reasons.append("lost_identifier")
    if copied_ids:
        reasons.append("copied_identifier_from_item1")

    normalized_candidate = normalize_surface(candidate)
    missing_protected_facts: list[str] = []
    for fact in contract.get("protected_b_facts", []):
        representation = normalize_surface(fact.get("name_representation") or "")
        # Numbers, ranges and digit-bearing model/article tokens use the
        # dedicated canonical checks above.  Generic semantic normalization
        # can split decimal commas (17,5 -> 17 + 5), so checking both would
        # create false rejects.
        if fact.get("concept") in rules.DIMENSION_CONCEPTS or any(char.isdigit() for char in representation):
            continue
        if representation and not re.search(
            rf"(?<![0-9a-zа-я]){re.escape(representation)}(?![0-9a-zа-я])",
            normalized_candidate,
        ):
            missing_protected_facts.append(str(fact["fact_id"]))
    if missing_protected_facts:
        reasons.append("lost_protected_b_fact")

    copied_forbidden_facts: list[str] = []
    original_normalized = normalize_surface(contract["name2"])
    for fact in contract.get("forbidden_a_only_facts", []):
        representation = normalize_surface(fact.get("name_representation") or "")
        if not representation or representation in original_normalized:
            continue
        pattern = rf"(?<![0-9a-zа-я]){re.escape(representation)}(?![0-9a-zа-я])"
        if re.search(pattern, normalized_candidate):
            copied_forbidden_facts.append(str(fact["fact_id"]))
    if copied_forbidden_facts:
        reasons.append("copied_forbidden_a_only_fact")

    diagnostics = {
        "missing_numeric_mentions": missing_numbers,
        "copied_numeric_mentions": copied_numbers,
        "missing_identifiers": missing_ids,
        "copied_identifiers": copied_ids,
        "candidate_numeric_mentions": sorted(candidate_numbers),
        "missing_protected_fact_ids": missing_protected_facts,
        "copied_forbidden_fact_ids": copied_forbidden_facts,
    }
    return not reasons, sorted(set(reasons)), diagnostics
