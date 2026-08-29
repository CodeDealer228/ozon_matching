"""Build one category-balanced local source pack for baseline vs aggressive arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count_must_be_positive")

    rows = [
        json.loads(line)
        for line in args.preview.read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        source_hash = str(row["source_pair_sha256"])
        if source_hash in seen:
            continue
        seen.add(source_hash)
        by_category[str(row["category"])].append(row)

    selected: list[dict[str, object]] = []
    offsets: dict[str, int] = defaultdict(int)
    categories = sorted(by_category)
    while len(selected) < args.count:
        added = False
        for category in categories:
            offset = offsets[category]
            if offset >= len(by_category[category]):
                continue
            selected.append(by_category[category][offset])
            offsets[category] += 1
            added = True
            if len(selected) == args.count:
                break
        if not added:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"only_{len(selected)}_sources_available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "comparison_source_pack.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    counts = Counter(str(row["category"]) for row in selected)
    manifest = {
        "status": "complete",
        "source": "baseline_all_eligible_dry_run_preview",
        "source_preview_sha256": sha256_file(args.preview),
        "rows": len(selected),
        "categories": len(counts),
        "category_counts": dict(sorted(counts.items())),
        "source_pair_duplicates": len(selected) - len({row["source_pair_sha256"] for row in selected}),
        "ids_are_local_only": True,
        "validation_is_not_a_source": True,
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
    }
    (args.output_dir / "comparison_source_pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
