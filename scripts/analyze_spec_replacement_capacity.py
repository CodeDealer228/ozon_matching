"""Count strict non-identity replacement sources before a paid generation run."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from filter_token_replacement_training_candidates import (
    ALLOWED_SPEC_ROLES,
    mutation_rejection,
)
from generate_token_replacement_counterfactuals import parse_attributes, read_jsonl, replace_text
from generate_token_replacement_fast_v2 import (
    NUMERIC_ROLES,
    VALUE_RE,
    infer_source_role,
    token_roles,
)


LETTER_SIZES = {"xxs", "xs", "s", "m", "l", "xl", "xxl", "xxxl"}


def strict_spec_token(role: str, token: str) -> bool:
    if role not in NUMERIC_ROLES:
        return True
    if any(char.isdigit() for char in token):
        return True
    return role == "size" and token.casefold() in LETTER_SIZES


DIMENSION_ROLES = {"length", "width", "height", "diameter", "size"}
TITLE_CUES = (
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:мл|литр|oz\b|унци)", re.I), {"volume"}),
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:mah|мач)", re.I), {"volume"}),
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:кг|г|грамм)", re.I), {"weight"}),
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:шт|штук|пар|линз|лист|цвет|детал|элемент|led|ламп)", re.I), {"count"}),
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:мм|см|метр|м\b)", re.I), DIMENSION_ROLES),
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:вт|w\b)", re.I), {"power"}),
    (re.compile(r"^(?:\s|[-–—,:;(])*(?:dpi)", re.I), {"resolution"}),
)

ROLE_MEASUREMENT_PATTERNS = {
    "volume": re.compile(r"\d+(?:[.,]\d+)?\s*(?:мл|л\b|литр|oz\b|унци)", re.I),
    "weight": re.compile(r"\d+(?:[.,]\d+)?\s*(?:кг|г\b|грамм)", re.I),
    "count": re.compile(r"\d+(?:[.,]\d+)?\s*(?:шт|штук|пар|линз|лист|цвет|детал|элемент|led|ламп)", re.I),
    "power": re.compile(r"\d+(?:[.,]\d+)?\s*(?:вт|w\b)", re.I),
}


def safe_title_occurrence(name: str, token: str, role: str) -> bool:
    escaped = re.escape(token)
    if any(char.isdigit() for char in token):
        pattern = re.compile(rf"(?<!\d){escaped}(?!\d)", re.I)
    else:
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.I)
    matches = list(pattern.finditer(name))
    if len(matches) != 1:
        return False
    match = matches[0]
    before = name[max(0, match.start() - 2):match.start()]
    after = name[match.end():match.end() + 18]
    if before and before[-1:] in ".," and len(before) > 1 and before[-2].isdigit():
        return False
    if after[:1] in ".," and len(after) > 1 and after[1].isdigit():
        return False
    if replace_text(name, token, "__probe__")[1] != 1:
        return False
    measurement_pattern = ROLE_MEASUREMENT_PATTERNS.get(role)
    if measurement_pattern is not None and len(measurement_pattern.findall(name)) != 1:
        return False
    for cue, allowed_roles in TITLE_CUES:
        if cue.search(after):
            return role in allowed_roles
    return True


def strict_source_role(left: pd.Series, right: pd.Series, token: str) -> str | None:
    role = infer_source_role(left, right, token)
    if role not in ALLOWED_SPEC_ROLES or not strict_spec_token(str(role), token):
        return None
    if role == "power":
        return None
    left_roles = token_roles(parse_attributes(left["attributes"]), token)
    right_roles = token_roles(parse_attributes(right["attributes"]), token)
    if left_roles & right_roles != {role}:
        return None
    if not safe_title_occurrence(str(left["name"]), token, str(role)):
        return None
    if not safe_title_occurrence(str(right["name"]), token, str(role)):
        return None
    return str(role)


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
            role = strict_source_role(left, right, token)
            if role is None:
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
