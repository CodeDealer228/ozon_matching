"""Sample human pairs and print intersections of item fact lemmas.

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
    parser.add_argument("--pairs", type=Path, default=Path("data/matches.parquet"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-attributes-chars", type=int, default=1000)
    args = parser.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "category", "name", "attributes"]).set_index("id")
    pairs = pd.read_parquet(args.pairs, columns=["id1", "id2", "target"]).sample(frac=1, random_state=args.seed)
    shown = 0
    for pair in pairs.itertuples(index=False):
        if pair.id1 not in items.index or pair.id2 not in items.index:
            continue
        left, right = items.loc[pair.id1], items.loc[pair.id2]
        if any(not isinstance(value, str) or len(value) > args.max_attributes_chars for value in (left["attributes"], right["attributes"])):
            continue
        try:
            left_values = " ".join(flatten_values(json.loads(left["attributes"])))
            right_values = " ".join(flatten_values(json.loads(right["attributes"])))
        except json.JSONDecodeError:
            continue
        left_tokens = lemma_set(f"{left['name']} {left_values}")
        right_tokens = lemma_set(f"{right['name']} {right_values}")
        print(
            f"\n{'=' * 80}\ntarget={pair.target}\n"
            f"Item 1: id={pair.id1} | {left['category']}\nНазвание: {left['name']}\nАтрибуты: {left['attributes']}\n"
            f"\nItem 2: id={pair.id2} | {right['category']}\nНазвание: {right['name']}\nАтрибуты: {right['attributes']}\n"
            f"\nПересечение лемм ({len(left_tokens & right_tokens)}): {sorted(left_tokens & right_tokens)}"
        )
        shown += 1
        if shown == args.count:
            break
    if shown != args.count:
        raise RuntimeError(f"Found only {shown} valid pairs")


if __name__ == "__main__":
    main()
