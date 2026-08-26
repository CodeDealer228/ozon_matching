"""Build a local, human-annotation template from a dry-run request preview."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from generate_token_replacement_counterfactuals import parse_attributes, read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("artifacts/token_replacement_counterfactual/few_shot_sources/request_preview.jsonl"),
    )
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/token_replacement_counterfactual/few_shot_annotation_template_v1.json"),
    )
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    if args.per_category <= 0:
        raise ValueError("per_category_must_be_positive")

    preview = read_jsonl(args.preview)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in preview:
        category = str(row["category"])
        if len(grouped[category]) < args.per_category:
            grouped[category].append(row)
    counts = Counter({category: len(rows) for category, rows in grouped.items()})
    if len(grouped) != 20 or set(counts.values()) != {args.per_category}:
        raise RuntimeError(f"Expected {args.per_category} rows for 20 categories, got {dict(counts)}")

    item_ids = {
        row[key]
        for rows in grouped.values()
        for row in rows
        for key in ("id1", "id2")
    }
    items = pd.read_parquet(
        args.items,
        columns=["id", "category", "name", "attributes"],
        filters=[("id", "in", list(item_ids))],
    ).set_index("id", drop=False)
    if not item_ids.issubset(set(items.index)):
        raise RuntimeError("items_missing_for_few_shot_template")

    categories: dict[str, list[dict[str, object]]] = {}
    for category in sorted(grouped):
        categories[category] = []
        for row in grouped[category]:
            left, right = items.loc[row["id1"]], items.loc[row["id2"]]
            categories[category].append({
                "source_reference_local_only": {
                    "id1": row["id1"],
                    "id2": row["id2"],
                    "source_pair_sha256": row["source_pair_sha256"],
                },
                "input": {
                    "category": category,
                    "item_1": {
                        "name": str(left["name"]),
                        "attributes": parse_attributes(left["attributes"]),
                    },
                    "item_2": {
                        "name": str(right["name"]),
                        "attributes": parse_attributes(right["attributes"]),
                    },
                    "offered_tokens": row["offered_tokens"],
                },
                "decision": {
                    "status": "__FILL_REPLACE_OR_SKIP__",
                    "replacements": [],
                    "skip_reason": "",
                },
                "reviewer_note_local_only": "",
            })
    output = {
        "schema_version": 1,
        "status": "annotation_required",
        "instructions": [
            "Set decision.status to replace or skip.",
            "For replace, choose source_token only from input.offered_tokens.",
            "replacement_token must be one token of the same semantic type, without spaces.",
            "For skip, keep replacements empty and write a non-empty skip_reason.",
            "source_reference_local_only and reviewer_note_local_only are never sent to the provider.",
        ],
        "source_preview_sha256": sha256_file(args.preview),
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "categories": len(categories),
        "examples": sum(map(len, categories.values())),
        "sha256": sha256_file(args.output),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
