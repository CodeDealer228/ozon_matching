"""Data-driven research of safe synthetic-negative mutations.

This script intentionally uses only ``split == "train"`` from the canonical
human split.  It does not generate synthetic data and it does not train a
model.  Outputs:

    artifacts/synthetic_negative_rules/report.md
    artifacts/synthetic_negative_rules/rules.json

The aliases below are deliberately conservative and were selected after
inspecting frequent raw keys in the four target categories.  Broad fuzzy key
matching is avoided because similarly named fields often represent different
semantics (for example, product colour vs stone colour).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_CATEGORIES = (
    "Обувь",
    "Ювелирные изделия",
    "Одежда",
    "Автотовары",
)
BASELINE_AP = {
    "Обувь": 0.5138302524933993,
    "Ювелирные изделия": 0.5301269915701406,
    "Одежда": 0.5433739347723242,
    "Автотовары": 0.6942934741407881,
}


def _keys(*values: str) -> tuple[str, ...]:
    return values


# Exact aliases only.  Closely related but non-equivalent concepts are kept
# separate: product_color/stone_color, product_model/car_model,
# russian_size/manufacturer_size, and so on.
SEMANTIC_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "Обувь": {
        "brand": _keys("бренд", "бренд в одежде и обуви", "торговая марка"),
        "article": _keys("артикул", "артикул производителя"),
        "product_model": _keys(
            "модель", "модель обуви", "модель туфель", "модель ботинок", "наименование модели"
        ),
        "product_color": _keys(
            "цвет товара", "цвет", "название цвета", "цвет производителя", "основной цвет"
        ),
        "gender": _keys("пол", "пол ребенка"),
        "target_audience": _keys("целевая аудитория"),
        "russian_size": _keys(
            "российский размер", "российский размер (обуви)", "размер ru", "размер российский",
            "рос. размер", "размер обуви"
        ),
        "manufacturer_size": _keys("размер производителя"),
        "generic_size": _keys("размер"),
        "foot_length": _keys("длина стопы, см", "длина стопы, в сантиметрах", "длина стопы, в см"),
        "insole_length": _keys("длина стельки, см", "длина стельки, в сантиметрах", "длина стельки"),
        "shoe_fullness": _keys("полнота", "полнота обуви (eur)", "полнота ru/uk", "полнота обуви"),
        "upper_material": _keys("материал верха", "материал верха обуви"),
        "sole_material": _keys("материал подошвы", "материал подошвы обуви"),
        "insole_material": _keys("материал стельки"),
        "lining_material": _keys(
            "внутренний материал", "материал подкладки обуви", "материал подкладки"
        ),
        "general_material": _keys("материал", "материал (состав)", "состав"),
        "completeness": _keys("комплектация", "комплектность"),
        "season": _keys("сезон"),
        "collection": _keys("коллекция"),
    },
    "Ювелирные изделия": {
        "brand": _keys("бренд", "торговая марка"),
        "article": _keys("артикул", "артикул производителя", "артикул поставщика"),
        "product_model": _keys(
            "модель", "модель кольца", "модель сережек", "модель браслета", "модель пирсинга",
            "наименование модели"
        ),
        "product_color": _keys("цвет товара", "цвет", "название цвета", "основной цвет"),
        "stone_color": _keys("цвет вставки"),
        "metal_color": _keys("цвет материала"),
        "gender": _keys("пол"),
        "item_size": _keys("размер изделия", "размер украшения", "размер"),
        "ring_size": _keys("размер кольца", "российский размер"),
        "dimensions": _keys("размеры, мм", "размер (дхшхв), мм", "размер (вхшхг), мм"),
        "average_weight": _keys(
            "средний вес изделия, гр", "средний вес изделия", "средний вес изделия, г",
            "средний вес", "средний вес изделия, в граммах", "средний вес, г"
        ),
        "item_weight": _keys("вес товара, г", "вес изделия", "вес, г", "вес, в граммах"),
        "metal_fineness": _keys("проба"),
        "jewelry_material": _keys(
            "основной материал украшения", "материал изделия", "основной материал", "вид металла"
        ),
        "jewelry_composition": _keys(
            "состав ювелирного изделия", "состав бижутерии", "состав"
        ),
        "main_insert": _keys("вставка"),
        "stone_count": _keys("количество камней"),
        "completeness": _keys("комплектация", "состав комплекта", "подробная комплектация"),
        "collection": _keys("коллекция"),
        "item_type": _keys("тип", "вид"),
    },
    "Одежда": {
        "brand": _keys("бренд", "бренд в одежде и обуви", "торговая марка"),
        "article": _keys("артикул", "артикул производителя", "артикул sku", "номер артикула"),
        "product_model": _keys(
            "модель", "модель джинсов", "модель брюк", "модель юбки", "модель пальто",
            "модель куртки", "модель купальника", "модель бюстгальтера", "наименование модели"
        ),
        "product_color": _keys(
            "цвет товара", "цвет", "название цвета", "цвет производителя", "основной цвет"
        ),
        "gender": _keys("пол"),
        "russian_size": _keys(
            "российский размер", "размер ru", "размер российский", "рос. размер", "размер одежды"
        ),
        "manufacturer_size": _keys("размер производителя", "размер на бирке", "размер поставщика"),
        "generic_size": _keys("размер"),
        "jeans_size": _keys("размер джинсов"),
        "height_size": _keys("рост", "размер и ростовка", "размер/рост"),
        "material_composition": _keys(
            "материал", "состав материала", "состав", "материал (состав)", "состав от партнера"
        ),
        "lining_material": _keys("материал подкладки", "материал подклада/внутренней отделки"),
        "package_count": _keys("количество в упаковке"),
        "completeness": _keys("комплектация", "состав комплекта"),
        "season": _keys("сезон"),
        "collection": _keys("коллекция"),
        "item_type": _keys("тип", "вид одежды"),
    },
    "Автотовары": {
        "brand": _keys("бренд", "торговая марка"),
        "article": _keys("артикул", "артикул производителя", "партномер (артикул производителя)"),
        "oem_number": _keys("oem-номер", "оем номер", "oem номер", "oem-номер производителя"),
        "alternative_articles": _keys("альтернативные артикулы товара"),
        "product_model": _keys("модель", "наименование модели"),
        "product_color": _keys(
            "цвет товара", "цвет", "название цвета", "цвет производителя", "основной цвет"
        ),
        "general_material": _keys("материал", "материал изделия", "основной материал", "состав материала"),
        "package_count": _keys(
            "количество, штук", "количество в упаковке, шт", "количество в упаковке",
            "количество предметов в упаковке"
        ),
        "completeness": _keys(
            "комплектация", "подробная комплектация", "состав комплекта", "комплект поставки"
        ),
        "compatibility": _keys("совместимость", "совместимые автомобили"),
        "car_make": _keys("марка автомобиля", "марка тс", "марка грузового автомобиля"),
        "car_model": _keys("модель автомобиля", "модель тс"),
        "car_generation": _keys("поколение автомобиля"),
        "car_year": _keys("год выпуска автомобиля", "года выпуска", "год выпуска а/м"),
        "engine": _keys("двигатель", "тип двигателя", "вид двигателя"),
        "placement": _keys("место установки", "расположение детали"),
        "volume": _keys("объем, мл", "объем, л", "объем"),
        "diameter": _keys("диаметр", "диаметр, мм", "диаметр, см", "диаметр, дюймы", "диаметр, дюйм"),
        "length": _keys("длина", "длина, мм", "длина, см", "длина, в миллиметрах", "длина, в сантиметрах"),
        "width": _keys("ширина", "ширина, мм", "ширина, см", "ширина, в миллиметрах", "ширина, в сантиметрах"),
        "height": _keys("высота", "высота, мм", "высота, см", "высота, в миллиметрах", "высота, в сантиметрах"),
        "power": _keys("мощность", "мощность, вт", "мощность, в ваттах"),
        "tire_width": _keys("ширина профиля, мм", "ширина профиля", "ширина профиля, в миллиметрах"),
        "tire_profile": _keys("высота профиля, %", "высота профиля", "высота профиля, в процентах"),
        "tire_diameter": _keys("диаметр шины", "диаметр автомобильного диска", "диаметр диска"),
        "item_type": _keys("тип"),
        "seasonality": _keys("сезонность", "сезон"),
    },
}


COLOR_GROUPS = {
    "black": ("черный", "черная", "черное", "черные", "чёрный", "чёрная", "чёрное", "чёрные", "black"),
    "white": ("белый", "белая", "белое", "белые", "white"),
    "red": ("красный", "красная", "красное", "красные", "red"),
    "blue": ("синий", "синяя", "синее", "синие", "blue"),
    "green": ("зеленый", "зеленая", "зеленое", "зеленые", "зелёный", "зелёная", "green"),
    "grey": ("серый", "серая", "серое", "серые", "gray", "grey"),
    "brown": ("коричневый", "коричневая", "коричневое", "коричневые", "brown"),
    "beige": ("бежевый", "бежевая", "бежевое", "бежевые", "beige"),
    "gold": ("золотой", "золотая", "золотое", "золотые", "gold"),
    "silver": ("серебристый", "серебряный", "серебристая", "silver"),
    "pink": ("розовый", "розовая", "розовое", "розовые", "pink"),
    "yellow": ("желтый", "желтая", "желтое", "желтые", "жёлтый", "жёлтая", "yellow"),
    "orange": ("оранжевый", "оранжевая", "оранжевое", "оранжевые", "orange"),
    "purple": ("фиолетовый", "фиолетовая", "фиолетовое", "фиолетовые", "purple"),
}
COLOR_ALIAS_TO_CANONICAL = {
    alias.replace("ё", "е"): canonical
    for canonical, aliases in COLOR_GROUPS.items()
    for alias in aliases
}

DIMENSION_CONCEPTS = {
    "russian_size", "manufacturer_size", "generic_size", "jeans_size", "height_size",
    "ring_size", "item_size", "dimensions", "foot_length", "insole_length", "shoe_fullness",
    "average_weight", "item_weight", "package_count", "stone_count", "volume", "diameter",
    "length", "width", "height", "power", "tire_width", "tire_profile", "tire_diameter",
    "car_year", "metal_fineness",
}
IDENTIFIER_CONCEPTS = {"article", "oem_number", "alternative_articles"}
COLOR_CONCEPTS = {"product_color", "stone_color", "metal_color"}
MATERIAL_PATTERNS = (
    ("faux_leather", r"\b(?:искусственн\w*\s+кож\w*|экокож\w*)\b"),
    ("suede", r"\b(?:замш\w*|suede)\b"),
    ("nubuck", r"\b(?:нубук\w*|nubuck)\b"),
    ("leather", r"\b(?:натуральн\w*\s+кож\w*|кож\w*|leather)\b"),
    ("textile", r"\b(?:текстил\w*|ткан\w*|textile)\b"),
    ("rubber", r"\b(?:резин\w*|каучук\w*|rubber)\b"),
    ("tep", r"\b(?:тэп|термоэластопласт\w*)\b"),
    ("eva", r"\b(?:эва|eva|этиленвинилацетат\w*)\b"),
    ("pvc", r"\b(?:пвх|pvc|поливинилхлорид\w*)\b"),
    ("polyurethane", r"\b(?:полиуретан\w*|polyurethane)\b"),
    ("polyester", r"\b(?:полиэстер\w*|polyester)\b"),
    ("cotton", r"\b(?:хлоп\w*|cotton)\b"),
    ("wool", r"\b(?:шерст\w*|wool)\b"),
    ("silver", r"\b(?:серебр\w*|silver)\b"),
    ("gold", r"\b(?:золот\w*|gold)\b"),
    ("steel", r"\b(?:стал\w*|steel)\b"),
    ("aluminium", r"\b(?:алюмини\w*|aluminium|aluminum)\b"),
)

CLASSIFICATION_THRESHOLDS = {
    "safe_min_positive_known": 20,
    "safe_min_negative_known": 40,
    "safe_max_positive_diff_rate": 0.03,
    "safe_min_near_identical_support": 8,
    "safe_min_near_identical_negative_rate": 0.90,
    "unsafe_min_positive_known": 20,
    "unsafe_min_positive_diff_count": 8,
    "unsafe_min_positive_diff_rate": 0.12,
    "unsafe_min_near_identical_support": 8,
    "unsafe_min_near_identical_positive_rate": 0.25,
}


def normalize_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = re.sub(r"[_–—]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace("ё", "е")
    replacements = {
        r"\bгигабайт(?:а|ов)?\b|\bгб\b": " gb ",
        r"\bgb\b": " gb ",
        r"\bмиллиметр(?:а|ов)?\b|\bмм\b": " mm ",
        r"\bсантиметр(?:а|ов)?\b|\bсм\b": " cm ",
        r"\bкилограмм(?:а|ов)?\b|\bкг\b": " kg ",
        r"\bграмм(?:а|ов)?\b|\bгр\b": " g ",
        r"\bмиллилитр(?:а|ов)?\b|\bмл\b": " ml ",
        r"\bлитр(?:а|ов)?\b|\bл\b": " l ",
        r"\bштук(?:а|и)?\b|\bшт\b": " pcs ",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^0-9a-zа-я%+.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def flatten_raw_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for key, nested in value.items():
            result.extend(flatten_raw_value(key))
            result.extend(flatten_raw_value(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for nested in value:
            result.extend(flatten_raw_value(nested))
        return result
    text = str(value).strip()
    return [text] if text and text.lower() not in {"none", "null", "nan", "не указано"} else []


def canonicalize_value(concept: str, raw: object) -> set[str]:
    result: set[str] = set()
    for source in flatten_raw_value(raw):
        normalized = normalize_text(source)
        if not normalized:
            continue
        if concept in COLOR_CONCEPTS:
            words = normalized.split()
            colors = {COLOR_ALIAS_TO_CANONICAL[word] for word in words if word in COLOR_ALIAS_TO_CANONICAL}
            result.update(colors or {normalized})
            continue
        if concept in IDENTIFIER_CONCEPTS:
            for part in re.split(r"[;,|/\n]+", normalized):
                compact = re.sub(r"[^0-9a-zа-я]+", "", part)
                if len(compact) >= 2:
                    result.add(compact)
                # Seller exports often prefix a real vendor code with a brand,
                # separated by an underscore/space: ``febi_17790``.  Retain
                # digit-bearing tokens as additional exact representations;
                # pure brand tokens are intentionally ignored.
                for token in re.findall(r"[0-9a-zа-я]+", part):
                    if len(token) >= 3 and any(char.isdigit() for char in token):
                        result.add(token)
            continue
        if concept in DIMENSION_CONCEPTS:
            measures = re.findall(r"\d+(?:[.,]\d+)?\s*(?:gb|mm|cm|kg|g|ml|l|pcs|%|ru|eur|eu)?", normalized)
            measures = [re.sub(r"\s+", "", value.replace(",", ".")) for value in measures]
            if measures:
                result.update(measures)
                continue
        if concept.endswith("material") or "composition" in concept:
            material_tags = set()
            for canonical, pattern in MATERIAL_PATTERNS:
                if canonical == "leather" and "faux_leather" in material_tags:
                    continue
                if re.search(pattern, normalized):
                    material_tags.add(canonical)
            if material_tags:
                result.update(material_tags)
                continue
        if concept == "main_insert":
            if re.search(r"\bбез\s+встав(?:ки|ок)\b", normalized):
                result.add("no_insert")
                continue
        tokens = normalized.split()
        if concept.endswith("material") or "composition" in concept:
            normalized = " ".join(sorted(tokens))
        result.add(normalized)
    return result


def parse_attributes(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@lru_cache(maxsize=None)
def build_alias_index(category: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for concept, aliases in SEMANTIC_ALIASES[category].items():
        for alias in aliases:
            key = normalize_key(alias)
            if key in index and index[key] != concept:
                raise RuntimeError(f"Alias collision for {category}: {key} -> {index[key]} / {concept}")
            index[key] = concept
    return index


def extract_semantics(category: str, attributes: dict[str, object]) -> dict[str, dict[str, object]]:
    alias_index = build_alias_index(category)
    extracted: dict[str, dict[str, object]] = {}
    for raw_key, raw_value in attributes.items():
        normalized_key = normalize_key(raw_key)
        concept = alias_index.get(normalized_key)
        if concept is None:
            continue
        values = canonicalize_value(concept, raw_value)
        if not values:
            continue
        entry = extracted.setdefault(concept, {"values": set(), "raw": [], "keys": set()})
        entry["values"].update(values)
        entry["raw"].extend(flatten_raw_value(raw_value))
        entry["keys"].add(str(raw_key).strip())
    return extracted


def name_similarity(name1: str, name2: str) -> tuple[float, float, float]:
    left = normalize_text(name1)
    right = normalize_text(name2)
    sequence = SequenceMatcher(None, left, right, autojunk=False).ratio() if left or right else 0.0
    tokens1 = set(left.split())
    tokens2 = set(right.split())
    union = tokens1 | tokens2
    jaccard = len(tokens1 & tokens2) / len(union) if union else 0.0
    combined = max(sequence, jaccard)
    return combined, sequence, jaccard


def concept_diff(left: dict[str, object], right: dict[str, object]) -> bool:
    return not bool(set(left["values"]) & set(right["values"]))


def find_name_representation(
    concept: str,
    raw_values: list[str],
    canonical_values: set[str],
    name: str,
) -> tuple[str, str] | None:
    normalized_name = normalize_text(name)
    compact_name = normalized_name.replace(" ", "")
    candidates: list[tuple[str, str]] = []
    minimum_length = 1 if concept in DIMENSION_CONCEPTS else 2 if concept in IDENTIFIER_CONCEPTS else 3
    for raw in raw_values:
        candidate = normalize_text(raw)
        if len(candidate) >= minimum_length:
            raw_canonical_values = canonicalize_value(concept, raw)
            for canonical in raw_canonical_values:
                candidates.append((canonical, candidate))
                candidates.append((canonical, candidate.replace(" ", "")))
    if concept in COLOR_CONCEPTS:
        for canonical in canonical_values:
            for alias in COLOR_GROUPS.get(canonical, ()):
                candidates.append((canonical, normalize_text(alias)))
    for canonical, candidate in sorted(set(candidates), key=lambda value: len(value[1]), reverse=True):
        if not candidate:
            continue
        if " " in candidate and candidate in normalized_name:
            return canonical, candidate
        if " " not in candidate:
            if re.search(rf"(?<![0-9a-zа-я]){re.escape(candidate)}(?![0-9a-zа-я])", normalized_name):
                return canonical, candidate
            if len(candidate) >= 4 and candidate in compact_name:
                return canonical, candidate
    return None


def empty_evidence() -> dict[str, object]:
    return {
        "positive_known": 0,
        "negative_known": 0,
        "positive_diff": 0,
        "negative_diff": 0,
        "near_positive": 0,
        "near_negative": 0,
        "positive_examples": [],
        "negative_examples": [],
        "near_identical_examples": [],
    }


def pair_example(
    row: object,
    item1: dict[str, object],
    item2: dict[str, object],
    concept: str,
    similarity: tuple[float, float, float],
    other_differences: list[str],
) -> dict[str, object]:
    left = item1["semantics"][concept]
    right = item2["semantics"][concept]
    return {
        "id1": int(row.id1),
        "id2": int(row.id2),
        "label": int(row.label),
        "name1": item1["name"],
        "name2": item2["name"],
        "value1": sorted(left["values"]),
        "value2": sorted(right["values"]),
        "raw_keys1": sorted(left["keys"]),
        "raw_keys2": sorted(right["keys"]),
        "name_similarity": round(similarity[0], 6),
        "sequence_similarity": round(similarity[1], 6),
        "token_jaccard": round(similarity[2], 6),
        "other_semantic_differences": other_differences,
    }


def add_example(target: list[dict[str, object]], value: dict[str, object], limit: int = 4) -> None:
    if len(target) < limit:
        target.append(value)


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def classify_rule(evidence: dict[str, object]) -> tuple[str, str]:
    t = CLASSIFICATION_THRESHOLDS
    positive_known = int(evidence["positive_known"])
    negative_known = int(evidence["negative_known"])
    positive_diff = int(evidence["positive_diff"])
    near_positive = int(evidence["near_positive"])
    near_negative = int(evidence["near_negative"])
    positive_diff_rate = safe_div(positive_diff, positive_known)
    near_support = near_positive + near_negative
    near_negative_rate = safe_div(near_negative, near_support)
    near_positive_rate = safe_div(near_positive, near_support)

    safe = (
        positive_known >= t["safe_min_positive_known"]
        and negative_known >= t["safe_min_negative_known"]
        and positive_diff_rate is not None
        and positive_diff_rate <= t["safe_max_positive_diff_rate"]
        and near_support >= t["safe_min_near_identical_support"]
        and near_negative_rate is not None
        and near_negative_rate >= t["safe_min_near_identical_negative_rate"]
    )
    if safe:
        return (
            "safe_negative",
            "Редкие различия в human-positive и не менее 90% human-negative среди строгих near-identical примеров.",
        )

    unsafe_by_positive = (
        positive_known >= t["unsafe_min_positive_known"]
        and positive_diff >= t["unsafe_min_positive_diff_count"]
        and positive_diff_rate is not None
        and positive_diff_rate >= t["unsafe_min_positive_diff_rate"]
    )
    unsafe_by_near = (
        near_support >= t["unsafe_min_near_identical_support"]
        and near_positive_rate is not None
        and near_positive_rate >= t["unsafe_min_near_identical_positive_rate"]
    )
    if unsafe_by_positive or unsafe_by_near:
        reasons = []
        if unsafe_by_positive:
            reasons.append("характеристика регулярно различается в human-positive")
        if unsafe_by_near:
            reasons.append("near-identical evidence содержит существенную долю positive")
        return "unsafe_variant", "; ".join(reasons) + "."

    return "uncertain", "Недостаточный support или неоднозначная картина; консервативно не использовать для генерации."


def json_ready_rate(value: float | None) -> float | None:
    return round(value, 8) if value is not None and math.isfinite(value) else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md(value: object, limit: int = 180) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def rate_text(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def analyze(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    started = time.time()
    print(f"[READ] canonical split: {args.split_path}", flush=True)
    # The filter is intentional: validation rows never enter the dataframe.
    pairs = pd.read_parquet(
        args.split_path,
        columns=["id1", "id2", "label", "split"],
        filters=[("split", "=", "train")],
    )
    if set(pairs["split"].dropna().unique()) != {"train"}:
        raise RuntimeError("Canonical split filter returned non-train rows")
    if not set(pairs["label"].dropna().unique()).issubset({0, 1}):
        raise RuntimeError("Human train labels are not binary")
    train_rows_before_validation_guard = len(pairs)

    # Validation labels and item content are never read.  Validation IDs are
    # used only as a deny-list, because the canonical split is not fully
    # item-disjoint.  This prevents shared items from influencing rules or
    # appearing in examples/source candidates.
    validation_refs = pd.read_parquet(
        args.split_path,
        columns=["id1", "id2"],
        filters=[("split", "=", "val")],
    )
    validation_item_ids = set(validation_refs["id1"]) | set(validation_refs["id2"])
    train_item_ids_before_guard = set(pairs["id1"]) | set(pairs["id2"])
    overlapping_item_ids = train_item_ids_before_guard & validation_item_ids
    validation_guard_mask = ~pairs["id1"].isin(validation_item_ids) & ~pairs["id2"].isin(validation_item_ids)
    pairs = pairs[validation_guard_mask].copy()
    excluded_train_rows = train_rows_before_validation_guard - len(pairs)

    print(f"[READ] human items: {args.items_path}", flush=True)
    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    if items["id"].duplicated().any():
        raise RuntimeError("items_human contains duplicate item IDs")
    # Filter by canonical train IDs before any attribute/name analysis.  This is
    # what prevents validation-only items from influencing rules indirectly.
    train_item_ids = set(pairs["id1"]) | set(pairs["id2"])
    items = items[
        items["id"].isin(train_item_ids)
        & items["category"].isin(TARGET_CATEGORIES)
    ].copy()

    category_by_id = items.set_index("id")["category"]
    pairs["category1"] = pairs["id1"].map(category_by_id)
    pairs["category2"] = pairs["id2"].map(category_by_id)
    target_pairs = pairs[
        pairs["category1"].isin(TARGET_CATEGORIES)
        & pairs["category1"].eq(pairs["category2"])
    ].copy()
    missing_target_item_ids = sorted(
        (set(target_pairs["id1"]) | set(target_pairs["id2"])) - set(items["id"])
    )
    if missing_target_item_ids:
        raise RuntimeError(f"Missing target item IDs in items_human: {len(missing_target_item_ids)}")

    print(f"[PARSE] target items={len(items):,} target train pairs={len(target_pairs):,}", flush=True)
    item_records: dict[int, dict[str, object]] = {}
    observed_keys: dict[str, dict[str, Counter]] = {
        category: {concept: Counter() for concept in SEMANTIC_ALIASES[category]}
        for category in TARGET_CATEGORIES
    }
    item_concept_counts: dict[str, Counter] = {category: Counter() for category in TARGET_CATEGORIES}
    name_presence_counts: dict[str, Counter] = {category: Counter() for category in TARGET_CATEGORIES}
    representation_counts: dict[str, dict[str, Counter]] = {
        category: {concept: Counter() for concept in SEMANTIC_ALIASES[category]}
        for category in TARGET_CATEGORIES
    }
    bad_attribute_json = Counter()

    for row in items.itertuples(index=False):
        attributes = parse_attributes(row.attributes)
        if row.attributes and not attributes:
            bad_attribute_json[row.category] += 1
        semantics = extract_semantics(row.category, attributes)
        item_records[int(row.id)] = {
            "name": str(row.name or ""),
            "category": row.category,
            "semantics": semantics,
        }
        for concept, entry in semantics.items():
            item_concept_counts[row.category][concept] += 1
            observed_keys[row.category][concept].update(entry["keys"])
            representation = find_name_representation(
                concept, entry["raw"], entry["values"], str(row.name or "")
            )
            if representation is not None:
                name_presence_counts[row.category][concept] += 1
                canonical, matched_text = representation
                representation_counts[row.category][concept][(canonical, matched_text)] += 1

    evidence: dict[str, dict[str, dict[str, object]]] = {
        category: {concept: empty_evidence() for concept in SEMANTIC_ALIASES[category]}
        for category in TARGET_CATEGORIES
    }
    category_pair_counts: dict[str, Counter] = {category: Counter() for category in TARGET_CATEGORIES}

    print("[ANALYZE] human train pairs", flush=True)
    for category in TARGET_CATEGORIES:
        subset = target_pairs[target_pairs["category1"].eq(category)]
        print(f"  {category}: {len(subset):,}", flush=True)
        concepts = tuple(SEMANTIC_ALIASES[category])
        for row in subset.itertuples(index=False):
            label = int(row.label)
            category_pair_counts[category][label] += 1
            item1 = item_records[int(row.id1)]
            item2 = item_records[int(row.id2)]
            semantics1 = item1["semantics"]
            semantics2 = item2["semantics"]
            common = [concept for concept in concepts if concept in semantics1 and concept in semantics2]
            differences = [concept for concept in common if concept_diff(semantics1[concept], semantics2[concept])]
            similarity = name_similarity(item1["name"], item2["name"])
            for concept in common:
                current = evidence[category][concept]
                current["positive_known" if label == 1 else "negative_known"] += 1
                differs = concept in differences
                if not differs:
                    continue
                current["positive_diff" if label == 1 else "negative_diff"] += 1
                others = [value for value in differences if value != concept]
                example = pair_example(row, item1, item2, concept, similarity, others)
                add_example(
                    current["positive_examples" if label == 1 else "negative_examples"],
                    example,
                )
                equal_other = len(common) - len(differences)
                near_identical = similarity[0] >= args.near_name_similarity and len(others) <= 1 and equal_other >= 1
                if near_identical:
                    current["near_positive" if label == 1 else "near_negative"] += 1
                    add_example(current["near_identical_examples"], example, limit=6)

    result: dict[str, object] = {
        "_meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "research_scope": "canonical human split; split == train only",
            "split_path": str(Path(args.split_path).resolve()),
            "split_sha256": file_sha256(Path(args.split_path)),
            "items_path": str(Path(args.items_path).resolve()),
            "items_sha256": file_sha256(Path(args.items_path)),
            "target_categories": list(TARGET_CATEGORIES),
            "canonical_train_rows_before_validation_item_exclusion": int(train_rows_before_validation_guard),
            "canonical_train_rows_all_categories": int(len(pairs)),
            "validation_item_ids_used_as_denylist_only": int(len(validation_item_ids)),
            "train_validation_item_id_overlap": int(len(overlapping_item_ids)),
            "train_rows_excluded_by_validation_item_guard": int(excluded_train_rows),
            "target_train_rows": int(len(target_pairs)),
            "target_item_rows": int(len(items)),
            "near_identical_definition": {
                "name_similarity": f">= {args.near_name_similarity}",
                "other_known_semantic_differences": "<= 1",
                "equal_other_known_semantic_attributes": ">= 1",
                "name_similarity_formula": "max(SequenceMatcher ratio, token-set Jaccard)",
            },
            "classification_thresholds": CLASSIFICATION_THRESHOLDS,
            "bad_attribute_json": dict(bad_attribute_json),
            "elapsed_seconds": None,
        }
    }

    for category in TARGET_CATEGORIES:
        category_rules: dict[str, object] = {}
        for concept in SEMANTIC_ALIASES[category]:
            raw = evidence[category][concept]
            positive_known = int(raw["positive_known"])
            negative_known = int(raw["negative_known"])
            positive_diff = int(raw["positive_diff"])
            negative_diff = int(raw["negative_diff"])
            near_positive = int(raw["near_positive"])
            near_negative = int(raw["near_negative"])
            near_support = near_positive + near_negative
            status, reason = classify_rule(raw)
            top_representations: dict[str, list[str]] = defaultdict(list)
            for (canonical, representation), _count in representation_counts[category][concept].most_common(30):
                if representation not in top_representations[canonical] and len(top_representations[canonical]) < 5:
                    top_representations[canonical].append(representation)
                if len(top_representations) >= 12:
                    break
            item_support = int(item_concept_counts[category][concept])
            name_presence = int(name_presence_counts[category][concept])
            category_rules[concept] = {
                "status": status,
                "decision_reason": reason,
                "attribute_keys": sorted(observed_keys[category][concept]),
                "configured_aliases": list(SEMANTIC_ALIASES[category][concept]),
                "may_appear_in_name": bool(item_support and name_presence / item_support >= 0.05),
                "known_value_representations": dict(top_representations),
                "name_evidence": {
                    "item_support": item_support,
                    "value_detected_in_name": name_presence,
                    "value_detected_in_name_rate": json_ready_rate(safe_div(name_presence, item_support)),
                },
                "evidence": {
                    "support": positive_known + negative_known,
                    "positive_known": positive_known,
                    "negative_known": negative_known,
                    "positive_diff": positive_diff,
                    "negative_diff": negative_diff,
                    "positive_diff_rate": json_ready_rate(safe_div(positive_diff, positive_known)),
                    "negative_diff_rate": json_ready_rate(safe_div(negative_diff, negative_known)),
                    "near_identical_support": near_support,
                    "near_identical_positive": near_positive,
                    "near_identical_negative": near_negative,
                    "near_identical_negative_rate": json_ready_rate(safe_div(near_negative, near_support)),
                },
                "examples": {
                    "positive_attribute_diff": raw["positive_examples"],
                    "negative_attribute_diff": raw["negative_examples"],
                    "near_identical_attribute_diff": raw["near_identical_examples"],
                },
            }
        result[category] = category_rules

    result["_meta"]["category_train_pair_counts"] = {
        category: {
            "positive": int(category_pair_counts[category][1]),
            "negative": int(category_pair_counts[category][0]),
            "total": int(sum(category_pair_counts[category].values())),
        }
        for category in TARGET_CATEGORIES
    }
    result["_meta"]["elapsed_seconds"] = round(time.time() - started, 3)
    report = build_report(result)
    return result, report


def build_report(result: dict[str, object]) -> str:
    meta = result["_meta"]
    lines = [
        "# Исследование правил для synthetic hard negatives",
        "",
        "> Исследование выполнено только на `split == \"train\"` canonical human split. "
        "Human validation не использовался для выбора aliases, правил, порогов или примеров.",
        "",
        "## Данные и методика",
        "",
        f"- Canonical human train до защиты: **{meta['canonical_train_rows_before_validation_item_exclusion']:,}** пар.",
        f"- Исключено train-пар с validation item ID: **{meta['train_rows_excluded_by_validation_item_guard']:,}**; "
        f"пересекающихся item ID: **{meta['train_validation_item_id_overlap']:,}**.",
        f"- Human train после защиты, все категории: **{meta['canonical_train_rows_all_categories']:,}** пар.",
        f"- Четыре целевые категории: **{meta['target_train_rows']:,}** пар.",
        f"- Товаров целевых категорий в `items_human`: **{meta['target_item_rows']:,}**.",
        "- Значения сравнивались после нормализации регистра, `ё/е`, единиц измерения и пунктуации.",
        "- Aliases объединялись только по точным, заранее проверенным ключам; близкие, но разные поля оставлены раздельно.",
        "- Near-identical: сходство названий не ниже заданного порога, не более одного дополнительного "
        "semantic-различия и хотя бы одна совпадающая известная характеристика.",
        "- Validation использован только как deny-list item ID; его labels и item content не читались для исследования.",
        "- `safe_negative` назначался только при низкой доле различий в positive и сильном near-identical negative evidence.",
        "- При слабом support или противоречии выставлялся `uncertain`.",
        "",
        "### Объём human train",
        "",
        "| Категория | Baseline AP | Positive | Negative | Всего |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in TARGET_CATEGORIES:
        counts = meta["category_train_pair_counts"][category]
        lines.append(
            f"| {category} | {BASELINE_AP[category]:.6f} | {counts['positive']:,} | "
            f"{counts['negative']:,} | {counts['total']:,} |"
        )

    lines.extend(["", "## Таблица правил", ""])
    for category in TARGET_CATEGORIES:
        lines.extend([
            f"### {category}",
            "",
            "| Semantic attribute | Статус | Known + / − | Diff + / − | P(diff\\|+) | P(diff\\|−) | Near + / − | В name |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for concept, rule in result[category].items():
            evidence = rule["evidence"]
            name_evidence = rule["name_evidence"]
            lines.append(
                f"| `{concept}` | **{rule['status']}** | {evidence['positive_known']:,} / {evidence['negative_known']:,} | "
                f"{evidence['positive_diff']:,} / {evidence['negative_diff']:,} | "
                f"{rate_text(evidence['positive_diff_rate'])} | {rate_text(evidence['negative_diff_rate'])} | "
                f"{evidence['near_identical_positive']:,} / {evidence['near_identical_negative']:,} | "
                f"{rate_text(name_evidence['value_detected_in_name_rate'])} |"
            )
        lines.append("")

    lines.extend(["## Группы решений", ""])
    for status in ("safe_negative", "unsafe_variant", "uncertain"):
        lines.append(f"### `{status}`")
        lines.append("")
        any_rule = False
        for category in TARGET_CATEGORIES:
            concepts = [concept for concept, rule in result[category].items() if rule["status"] == status]
            if concepts:
                any_rule = True
                lines.append(f"- **{category}:** " + ", ".join(f"`{value}`" for value in concepts))
        if not any_rule:
            lines.append("- Нет правил, прошедших консервативные критерии.")
        lines.append("")

    lines.extend([
        "## Near-identical evidence",
        "",
        "Ниже приведены реальные пары только из human train. Показываются характеристики с наибольшим "
        "near-identical support; примеры не являются синтетическими.",
        "",
    ])
    for category in TARGET_CATEGORIES:
        ranked = sorted(
            result[category].items(),
            key=lambda item: item[1]["evidence"]["near_identical_support"],
            reverse=True,
        )[:6]
        lines.extend([f"### {category}", ""])
        for concept, rule in ranked:
            examples = rule["examples"]["near_identical_attribute_diff"]
            if not examples:
                continue
            evidence = rule["evidence"]
            lines.append(
                f"#### `{concept}` — {rule['status']}; near +/− = "
                f"{evidence['near_identical_positive']}/{evidence['near_identical_negative']}"
            )
            lines.append("")
            for example in examples[:3]:
                lines.append(
                    f"- label={example['label']}, sim={example['name_similarity']:.3f}, "
                    f"ID {example['id1']} ↔ {example['id2']}; "
                    f"`{md(example['value1'], 80)}` → `{md(example['value2'], 80)}`; "
                    f"other diffs={md(example['other_semantic_differences'], 90)}"
                )
                lines.append(f"  - A: {md(example['name1'])}")
                lines.append(f"  - B: {md(example['name2'])}")
            lines.append("")

    lines.extend([
        "## Name ↔ attributes",
        "",
        "Доля ниже — среди товаров, где structured attribute известен. Это эвристическое обнаружение "
        "нормализованного значения в `name`, поэтому небольшие числа не доказывают отсутствия связи.",
        "",
        "| Категория | Semantic attribute | Item support | Найдено в name | Доля | Примеры representations |",
        "|---|---|---:|---:|---:|---|",
    ])
    name_rows = []
    for category in TARGET_CATEGORIES:
        for concept, rule in result[category].items():
            evidence = rule["name_evidence"]
            if evidence["item_support"] <= 0:
                continue
            name_rows.append((
                evidence["value_detected_in_name_rate"] or 0.0,
                category,
                concept,
                rule,
            ))
    for _rate, category, concept, rule in sorted(name_rows, reverse=True)[:40]:
        evidence = rule["name_evidence"]
        reps = []
        for canonical, values in list(rule["known_value_representations"].items())[:3]:
            reps.append(f"{canonical}: {', '.join(values[:3])}")
        lines.append(
            f"| {category} | `{concept}` | {evidence['item_support']:,} | "
            f"{evidence['value_detected_in_name']:,} | {rate_text(evidence['value_detected_in_name_rate'])} | "
            f"{md('; '.join(reps), 160)} |"
        )

    lines.extend([
        "",
        "## Неожиданные и рискованные результаты",
        "",
    ])
    risky = []
    for category in TARGET_CATEGORIES:
        for concept, rule in result[category].items():
            evidence = rule["evidence"]
            if evidence["positive_diff_rate"] is not None and evidence["positive_diff"] >= 5:
                risky.append((evidence["positive_diff_rate"], category, concept, rule))
    for positive_rate, category, concept, rule in sorted(risky, reverse=True)[:20]:
        evidence = rule["evidence"]
        lines.append(
            f"- **{category} / `{concept}`:** различается в {rate_text(positive_rate)} human-positive "
            f"({evidence['positive_diff']}/{evidence['positive_known']}); статус `{rule['status']}`."
        )
    if not risky:
        lines.append("- Не найдено характеристик с пятью и более различиями в human-positive.")

    lines.extend([
        "",
        "## Ограничения исследования",
        "",
        "- Наличие human label не доказывает причинность: даже near-identical фильтр остаётся эвристикой.",
        "- Пропуски и неодинаковое заполнение structured attributes уменьшают support.",
        "- Значения aliases считаются совпадающими при пересечении нормализованных representations; это консервативно, "
        "но не является универсальным entity linking.",
        "- `uncertain` нельзя передавать генератору как критическую характеристику без дополнительной проверки.",
        "- Этот отчёт не использует и не оценивает human validation.",
        "",
        f"Время анализа: **{meta['elapsed_seconds']:.1f} с**.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-path",
        default=str(REPO_ROOT / "data" / "splits" / "matches_split_seed42.parquet"),
    )
    parser.add_argument(
        "--items-path",
        default=str(REPO_ROOT / "data" / "items_human.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "synthetic_negative_rules"),
    )
    parser.add_argument("--near-name-similarity", type=float, default=0.82)
    args = parser.parse_args()

    if not Path(args.split_path).is_file():
        raise FileNotFoundError(args.split_path)
    if not Path(args.items_path).is_file():
        raise FileNotFoundError(args.items_path)

    result, report = analyze(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rules_path = out_dir / "rules.json"
    report_path = out_dir / "report.md"
    rules_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"[DONE] {rules_path}", flush=True)
    print(f"[DONE] {report_path}", flush=True)


if __name__ == "__main__":
    main()
