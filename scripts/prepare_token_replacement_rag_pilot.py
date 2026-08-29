"""Select hard positive sources and retrieve relevant human-train RAG examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"[\w-]{2,}", flags=re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_tokens(value: object) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(str(value))}


def public_item(item: pd.Series) -> dict[str, object]:
    return {
        "name": str(item["name"]),
        "attributes": json.loads(str(item["attributes"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/splits/matches_item_disjoint_70_30_seed42.parquet"),
    )
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=600)
    parser.add_argument("--minimum-per-category", type=int, default=350)
    parser.add_argument("--source-candidates-per-category", type=int, default=1200)
    parser.add_argument("--positive-examples", type=int, default=2)
    parser.add_argument("--negative-examples", type=int, default=3)
    args = parser.parse_args()
    if min(
        args.per_category,
        args.minimum_per_category,
        args.source_candidates_per_category,
        args.positive_examples,
        args.negative_examples,
    ) <= 0:
        raise ValueError("counts_must_be_positive")

    items = pd.read_parquet(
        args.items, columns=["id", "category", "name", "attributes"]
    ).set_index("id", drop=False)
    all_pairs = pd.read_parquet(
        args.pairs, columns=["id1", "id2", "target", "category", "split"]
    )
    validation = all_pairs[~all_pairs["split"].eq("train")]
    validation_ids = set(validation["id1"]) | set(validation["id2"])
    train = all_pairs[all_pairs["split"].eq("train")].reset_index(drop=True)
    train_id1 = train["id1"].to_numpy()
    train_id2 = train["id2"].to_numpy()
    train_target = train["target"].to_numpy()

    signatures: list[set[str]] = []
    postings: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    category_sizes: Counter[str] = Counter()
    for index, pair in enumerate(train.itertuples(index=False)):
        if pair.id1 not in items.index or pair.id2 not in items.index:
            signatures.append(set())
            continue
        signature = text_tokens(items.at[pair.id1, "name"]) | text_tokens(
            items.at[pair.id2, "name"]
        )
        signatures.append(signature)
        category = str(pair.category)
        category_sizes[category] += 1
        for token in signature:
            postings[category][token].append(index)

    previews = [
        json.loads(line)
        for line in args.preview.read_text(encoding="utf-8").splitlines()
        if line
    ]
    limited: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in previews:
        category = str(row["category"])
        if len(limited[category]) < args.source_candidates_per_category:
            limited[category].append(row)

    ranked_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for category, source_rows in limited.items():
        category_postings = postings[category]
        category_size = max(category_sizes[category], 1)
        for source in source_rows:
            id1, id2 = source["id1"], source["id2"]
            if id1 in validation_ids or id2 in validation_ids:
                raise RuntimeError("validation_source_detected")
            if id1 not in items.index or id2 not in items.index:
                continue
            query = text_tokens(items.at[id1, "name"]) | text_tokens(items.at[id2, "name"])
            useful_tokens = sorted(
                (
                    token
                    for token in query
                    if token in category_postings
                    and len(category_postings[token]) <= max(5000, category_size // 3)
                ),
                key=lambda token: len(category_postings[token]),
            )[:8]
            candidate_indices: set[int] = set()
            for token in useful_tokens:
                remaining = 4000 - len(candidate_indices)
                if remaining <= 0:
                    break
                candidate_indices.update(category_postings[token][:remaining])
            scored: dict[int, float] = {}
            source_ids = {id1, id2}
            for pair_index in candidate_indices:
                if train_id1[pair_index] in source_ids or train_id2[pair_index] in source_ids:
                    continue
                signature = signatures[pair_index]
                shared = query & signature
                if not shared:
                    continue
                idf_sum = sum(
                    math.log((1 + category_size) / (1 + len(category_postings[token]))) + 1
                    for token in shared
                )
                scored[pair_index] = idf_sum / math.sqrt(max(len(query) * len(signature), 1))

            positive = sorted(
                (index for index in scored if float(train_target[index]) == 1.0),
                key=scored.get,
                reverse=True,
            )[: args.positive_examples]
            negative = sorted(
                (index for index in scored if float(train_target[index]) == 0.0),
                key=scored.get,
                reverse=True,
            )[: args.negative_examples]
            if len(positive) < args.positive_examples or len(negative) < args.negative_examples:
                continue
            rag_examples: list[dict[str, object]] = []
            for relation, indices in (("same_product", positive), ("different_product", negative)):
                for pair_index in indices:
                    rag_examples.append({
                        "relation": relation,
                        "category": category,
                        "item_1": public_item(items.loc[train_id1[pair_index]]),
                        "item_2": public_item(items.loc[train_id2[pair_index]]),
                        "retrieval_score": scored[pair_index],
                    })
            ranked_by_category[category].append({
                **source,
                "rag_examples": rag_examples,
                "hard_negative_score": scored[negative[0]],
                "nearest_positive_score": scored[positive[0]],
            })

    selected: list[dict[str, object]] = []
    for category in sorted(ranked_by_category):
        ranked = sorted(
            ranked_by_category[category],
            key=lambda row: (row["hard_negative_score"], row["nearest_positive_score"]),
            reverse=True,
        )
        if len(ranked) < args.minimum_per_category:
            raise RuntimeError(f"insufficient_rag_sources:{category}:{len(ranked)}")
        selected.extend(ranked[: min(args.per_category, len(ranked))])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rag_source_pack.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    counts = Counter(str(row["category"]) for row in selected)
    output_ids = {row["id1"] for row in selected} | {row["id2"] for row in selected}
    manifest = {
        "status": "complete",
        "rows": len(selected),
        "categories": len(counts),
        "category_counts": dict(sorted(counts.items())),
        "rag_examples_per_source": args.positive_examples + args.negative_examples,
        "rag_positive_per_source": args.positive_examples,
        "rag_negative_per_source": args.negative_examples,
        "source_pair_duplicates": len(selected)
        - len({row["source_pair_sha256"] for row in selected}),
        "validation_item_overlap": len(output_ids & validation_ids),
        "ids_are_local_only": True,
        "provider_rag_relation_sent": True,
        "source_pairs_sha256": sha256_file(args.pairs),
        "source_preview_sha256": sha256_file(args.preview),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
    }
    (args.output_dir / "rag_source_pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
