"""Small hard-safety contract for Method C generations."""

from __future__ import annotations

import copy
import json
import re
from typing import Any


KEY_PATTERNS = {
    "c_auto_part_number": re.compile(r"артикул|партномер|код|oem|номер", re.I),
    "c_jewelry_base_metal": re.compile(
        r"материал|металл|проб|цвет|драгоцен", re.I
    ),
    "c_auto_platform": re.compile(
        r"марка\s*(?:тс|авто)|модель\s*(?:тс|авто)|совместим|подходит|партномер|артикул",
        re.I,
    ),
}
MATERIAL_MARKERS = {
    "gold": ("gold", "золот"),
    "silver": ("silver", "серебр"),
    "platinum": ("platin", "платин"),
    "palladium": ("pallad", "паллад"),
}


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).lower().replace("ё", "е")).strip()


def compact(value: object) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", normalize(value))


def parse_json_object(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        raise ValueError(f"{label}_is_not_json_object")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}_is_not_json_object")
    return parsed


def markers_for_value(concept: str, value: object) -> tuple[str, ...]:
    normalized = normalize(value)
    if concept == "jewelry_material":
        return MATERIAL_MARKERS.get(normalized, (compact(value),))
    if concept == "metal_fineness":
        return (normalized,)
    marker = compact(value)
    return (marker,) if marker else ()


def marker_visible(marker: str, text: str) -> bool:
    if not marker:
        return False
    if re.fullmatch(r"\d+", marker):
        return bool(re.search(rf"(?<!\d){re.escape(marker)}(?!\d)", text))
    return marker in compact(text) or marker in normalize(text)


def values_visible(concept: str, values: list[object], text: str) -> bool:
    return any(
        marker_visible(marker, text)
        for value in values
        for marker in markers_for_value(concept, value)
    )


def target_key_allowed(rule_id: str, key: str) -> bool:
    pattern = KEY_PATTERNS.get(rule_id)
    return bool(pattern and pattern.search(normalize(key)))


def original_value_contains_target(
    original_value: object, source_values: dict[str, list[object]]
) -> bool:
    text = json.dumps(original_value, ensure_ascii=False)
    return any(values_visible(concept, values, text) for concept, values in source_values.items())


def target_numeric_tokens(source_values: dict[str, list[object]]) -> set[str]:
    return {
        token
        for values in source_values.values()
        for value in values
        for token in re.findall(r"\d+", normalize(value))
    }


def validate_response(
    plan: dict[str, object], response: dict[str, object]
) -> tuple[bool, list[str], dict[str, object]]:
    """Validate one parsed provider response and apply its attribute patch."""
    reasons: list[str] = []
    status = response.get("status")
    if status == "skip":
        reason = response.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reasons.append("skip_without_reason")
        return False, reasons, {"model_status": "skip", "skip_reason": reason}
    if status != "ok":
        return False, ["invalid_status"], {"model_status": status}

    synthetic_name = response.get("synthetic_name")
    if not isinstance(synthetic_name, str) or not synthetic_name.strip():
        reasons.append("missing_synthetic_name")
        synthetic_name = ""
    elif not 5 <= len(synthetic_name.strip()) <= 500:
        reasons.append("synthetic_name_length")

    updates = response.get("attribute_updates")
    if not isinstance(updates, dict) or not updates:
        reasons.append("missing_attribute_updates")
        updates = {}
    if len(updates) > 8:
        reasons.append("too_many_attribute_updates")

    try:
        source_attributes = parse_json_object(plan["source_attributes"], "source_attributes")
        source_values = parse_json_object(plan["source_values"], "source_values")
        replacement_values = parse_json_object(
            plan["replacement_values"], "replacement_values"
        )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return False, [str(exc)], {"model_status": "ok"}

    rule_id = str(plan["rule_id"])
    patched = copy.deepcopy(source_attributes)
    changed_keys: list[str] = []
    for raw_key, new_value in updates.items():
        key = str(raw_key)
        if key not in source_attributes:
            reasons.append("attribute_update_added_key")
            continue
        if source_attributes[key] == new_value:
            reasons.append("attribute_update_is_noop")
            continue
        if not target_key_allowed(rule_id, key) and not original_value_contains_target(
            source_attributes[key], source_values
        ):
            reasons.append("attribute_update_outside_target_scope")
            continue
        patched[key] = new_value
        changed_keys.append(key)

    if not changed_keys:
        reasons.append("no_valid_attribute_updates")

    final_text = synthetic_name + "\n" + json.dumps(patched, ensure_ascii=False)
    source_text = str(plan.get("source_name", "")) + "\n" + json.dumps(
        source_attributes, ensure_ascii=False
    )
    missing_replacements: list[str] = []
    old_targets_remaining: list[str] = []
    for concept, values in replacement_values.items():
        if not values_visible(str(concept), list(values), final_text):
            missing_replacements.append(str(concept))
    for concept, values in source_values.items():
        if values_visible(str(concept), list(values), source_text) and values_visible(
            str(concept), list(values), final_text
        ):
            old_targets_remaining.append(str(concept))
    if missing_replacements:
        reasons.append("replacement_target_missing")
    if old_targets_remaining:
        reasons.append("old_target_remains")

    protected_numbers = set(re.findall(r"\d+", normalize(plan.get("source_name", ""))))
    protected_numbers -= target_numeric_tokens(source_values)
    final_numbers = set(re.findall(r"\d+", normalize(final_text)))
    lost_protected_numbers = sorted(protected_numbers - final_numbers)
    if lost_protected_numbers:
        reasons.append("lost_protected_numeric_fact")

    diagnostics = {
        "model_status": "ok",
        "changed_attribute_keys": sorted(changed_keys),
        "missing_replacement_concepts": sorted(missing_replacements),
        "old_target_concepts_remaining": sorted(old_targets_remaining),
        "lost_protected_numeric_tokens": lost_protected_numbers,
        "synthetic_name": synthetic_name.strip(),
        "synthetic_attributes": json.dumps(
            patched, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }
    return not reasons, sorted(set(reasons)), diagnostics
