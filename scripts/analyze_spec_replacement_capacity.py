"""Count strict non-identity replacement sources before a paid generation run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from filter_token_replacement_training_candidates import (
    ALLOWED_SPEC_ROLES,
    mutation_rejection,
)
from generate_token_replacement_counterfactuals import read_jsonl
from generate_token_replacement_fast_v2 import VALUE_RE, infer_source_role


def shared_name_tokens(left: str, right: str) -> list[str]:
    right_tokens = {value.casefold() for value in VALUE_RE.findall(right)}
    output = []
    seen: set[str] = set()
    for value in VALUE_RE.findall(left):
        folded = value.casefold()
        if folded in right_tokens and folded not in seen:
            seen.add(folded)
            output.append(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pack", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/splits/matches_item_disjoint_70_30_seed42.parquet"),
    )
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    args = parser.parse_args()

    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id")
    if args.source_pack:
        rows = read_jsonl(args.source_pack)
    else:
        pairs = pd.read_parquet(
            args.pairs, columns=["id1", "id2", "target", "category", "split"]
        )
        pairs = pairs[pairs["split"].eq("train") & pairs["target"].eq(1)]
        rows = []
        for pair in pairs.itertuples(index=False):
            if pair.id1 not in items.index or pair.id2 not in items.index:
                continue
            left = items.loc[int(pair.id1)]
            right = items.loc[int(pair.id2)]
            rows.append({
                "id1": int(pair.id1),
                "id2": int(pair.id2),
                "category": str(pair.category),
                "offered_tokens": shared_name_tokens(
                    str(left["name"]), str(right["name"])
                ),
            })
    categories: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    tokens = 0
    for row in rows:
        left = items.loc[int(row["id1"])]
        right = items.loc[int(row["id2"])]
        valid = []
        for token_value in row["offered_tokens"]:
            token = str(token_value)
            role = infer_source_role(left, right, token)
            if role not in ALLOWED_SPEC_ROLES:
                continue
            _, rejection = mutation_rejection(left, right, token, token)
            if rejection is None:
                valid.append((token, role))
                roles[str(role)] += 1
        if valid:
            categories[str(row["category"])] += 1
            tokens += len(valid)
    print(json.dumps({
        "source_rows": len(rows),
        "strict_spec_sources": sum(categories.values()),
        "strict_spec_tokens": tokens,
        "role_counts": dict(sorted(roles.items())),
        "category_counts": dict(sorted(categories.items())),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
