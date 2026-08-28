"""Revalidate A2 rewrites against their immutable human-train item2 facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generate_a2_pilot_qwen import REPO_ROOT, file_sha256, load_helpers, validate_rewrite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "source_pools" / "a2_source_pool.parquet",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "a2_pilot_v2" / "a2_pilot_results.parquet",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.results.with_name("a2_pilot_validation.parquet")
    pool = pd.read_parquet(args.source_pool).set_index("source_pair_sha256")
    results = pd.read_parquet(args.results)
    helpers = load_helpers()
    rows = []
    for result in results.itertuples(index=False):
        if result.request_status != "complete" or not getattr(result, "rewritten_name2", None):
            continue
        source = pool.loc[result.source_pair_sha256]
        accepted, reasons, diagnostics = validate_rewrite(source, result.rewritten_name2, helpers)
        rows.append({
            "source_pair_sha256": result.source_pair_sha256,
            "accepted": accepted,
            "reasons": json.dumps(reasons, ensure_ascii=False),
            "diagnostics": json.dumps(diagnostics, ensure_ascii=False),
        })
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(json.dumps({
        "checked": len(frame),
        "accepted": int(frame["accepted"].sum()) if len(frame) else 0,
        "output": str(output),
        "source_pool_sha256": file_sha256(args.source_pool),
        "results_sha256": file_sha256(args.results),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
