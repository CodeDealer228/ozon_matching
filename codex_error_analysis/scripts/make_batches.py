from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from common import ROOT, load_config, read_jsonl, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    source = ROOT / "artifacts" / "prepared" / "audit_sample.jsonl"
    if not source.exists():
        raise SystemExit("Run prepare_audit.py first.")

    rows = read_jsonl(source)

    # Stable shuffle so each batch has a mixture of model/error/severity groups.
    rng = random.Random(int(cfg["random_seed"]) + 101)
    rng.shuffle(rows)

    batch_root = ROOT / "audit_batches"
    existing = sorted(p for p in batch_root.glob("batch_*") if p.is_dir())
    if existing and not args.force:
        raise SystemExit(
            "audit_batches already contains batches. Use --force only if you intentionally "
            "want to recreate them BEFORE classification."
        )
    for p in existing:
        shutil.rmtree(p)

    batch_size = int(cfg["batch_size"])
    manifest_rows = []

    for batch_idx, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start:start + batch_size]
        d = batch_root / f"batch_{batch_idx:03d}"
        d.mkdir(parents=True, exist_ok=True)

        input_p = d / "input.jsonl"
        with input_p.open("w", encoding="utf-8") as f:
            for rec in batch:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        (d / "INSTRUCTIONS.txt").write_text(
            "Read ../../TAXONOMY_RU.md and ../../schemas/classification.schema.json.\n"
            "Classify every line of input.jsonl and save output.jsonl here.\n"
            "Do not edit input.jsonl.\n",
            encoding="utf-8",
        )

        groups = {}
        for r in batch:
            key = f"{r['focal_model']}_{r['error_type']}_{r['severity_band']}"
            groups[key] = groups.get(key, 0) + 1

        manifest_rows.append({
            "batch": d.name,
            "n": len(batch),
            "input_sha256": sha256_file(input_p),
            "groups_json": json.dumps(groups, ensure_ascii=False, sort_keys=True),
        })

    manifest_p = batch_root / "batch_manifest.csv"
    with manifest_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "n", "input_sha256", "groups_json"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Created {len(manifest_rows)} batches, batch_size={batch_size}.")
    print("Manifest:", manifest_p)


if __name__ == "__main__":
    main()
