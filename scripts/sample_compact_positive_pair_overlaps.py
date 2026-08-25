"""Print positive pair token overlaps when both item payloads fit a character budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sample_name_attribute_overlap import flatten_values, token_set


def pair_char_count(left: object, right: object) -> int:
    return sum(len(str(item[key])) for item in (left, right) for key in ("name", "attributes"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--pairs", type=Path, default=Path("data/matches.parquet"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pair-chars", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "category", "name", "attributes"]).set_index("id")
    pairs = pd.read_parquet(args.pairs, columns=["id1", "id2", "target"])
    pairs = pairs[pairs["target"].eq(1)].sample(frac=1, random_state=args.seed)
    reports: list[str] = []
    for pair in pairs.itertuples(index=False):
        if pair.id1 not in items.index or pair.id2 not in items.index:
            continue
        left, right = items.loc[pair.id1], items.loc[pair.id2]
        pair_chars = pair_char_count(left, right)
        if pair_chars > args.max_pair_chars:
            continue
        try:
            left_values = " ".join(flatten_values(json.loads(left["attributes"])))
            right_values = " ".join(flatten_values(json.loads(right["attributes"])))
        except json.JSONDecodeError:
            continue
        shared = sorted(token_set(f"{left['name']} {left_values}") & token_set(f"{right['name']} {right_values}"))
        reports.append(
            f"\n{'=' * 80}\ntarget={pair.target} | pair_chars={pair_chars}\n"
            f"id_1={pair.id1} | category_1={left['category']}\nname_1: {left['name']}\nattr_1: {left['attributes']}\n"
            f"\nid_2={pair.id2} | category_2={right['category']}\nname_2: {right['name']}\nattr_2: {right['attributes']}\n"
            f"\nПересечение токенов ({len(shared)}): {shared}"
        )
        if len(reports) == args.count:
            break
    if len(reports) != args.count:
        raise RuntimeError(f"Found only {len(reports)} pairs within {args.max_pair_chars} characters")
    report = "\n".join(reports)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
