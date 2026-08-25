"""Sample items and print title facts confirmed by attribute values.

Requires: pip install pymorphy3
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from analyze_synthetic_negative_rules import normalize_text

try:
    from pymorphy3 import MorphAnalyzer
except ImportError as exc:
    raise RuntimeError("Install pymorphy3: pip install pymorphy3") from exc


def flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [text for nested in value.values() for text in flatten_values(nested)]
    if isinstance(value, list):
        return [text for nested in value for text in flatten_values(nested)]
    return [] if value is None else [str(value)]


morph = MorphAnalyzer()


@lru_cache(maxsize=100_000)
def lemma(token: str) -> str:
    return morph.parse(token)[0].normal_form if re.fullmatch(r"[а-я]+", token) else token


def lemma_set(text: object) -> set[str]:
    normalized = normalize_text(text)
    return {lemma(token) for token in re.findall(r"[0-9a-zа-я]+", normalized)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-attributes-chars", type=int, default=1000)
    args = parser.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "category", "name", "attributes"])
    sample = items[
        items["name"].notna()
        & items["attributes"].notna()
        & items["attributes"].str.len().le(args.max_attributes_chars)
    ].sample(args.count, random_state=args.seed)
    for row in sample.itertuples(index=False):
        try:
            attributes = json.loads(row.attributes)
        except json.JSONDecodeError:
            continue
        name_tokens = lemma_set(row.name)
        attribute_tokens = lemma_set(" ".join(flatten_values(attributes)))
        print(
            f"\n{'=' * 80}\nid={row.id} | {row.category}\n"
            f"Название: {row.name}\nАтрибуты: {row.attributes}\n"
            f"Леммы названия: {sorted(name_tokens)}\n"
            f"Леммы значений attributes: {sorted(attribute_tokens)}\n"
            f"Пересечение: {sorted(name_tokens & attribute_tokens)}"
        )


if __name__ == "__main__":
    main()
