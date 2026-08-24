"""Print A2 user prompts, Qwen replies, and current validator decisions."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from generate_a2_pilot_qwen import REPO_ROOT, build_messages, load_helpers, validate_rewrite

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, default=REPO_ROOT / "artifacts/synthetic_v2/source_pools/a2_source_pool.parquet")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "artifacts/synthetic_v2/a2_pilot_30/raw_responses.jsonl")
    parser.add_argument("--few-shot", type=Path, default=REPO_ROOT / "synthetic_research/a2_few_shot_v1.json")
    args = parser.parse_args()

    rows = pd.read_parquet(args.source_pool).set_index("source_pair_sha256")
    few_shot = json.loads(args.few_shot.read_text(encoding="utf-8"))
    helpers = load_helpers()
    for record in map(json.loads, args.raw.read_text(encoding="utf-8").splitlines()):
        if record.get("request_status") != "complete":
            continue
        row = rows.loc[record["source_pair_sha256"]]
        accepted, reasons, _ = validate_rewrite(row, record["rewritten_name2"], helpers)
        user_prompt = build_messages(few_shot, row, explicit_cache=False)[-1]["content"]
        print(
            f"\n{'=' * 100}\nINPUT (без system prompt):\n{user_prompt}"
            f"\n\nATTRIBUTES A (Qwen этого не видел):\n{row.attributes1}"
            f"\n\nQWEN:\n{record['raw_content']}"
            f"\n\nVALIDATOR: accepted={accepted}; reasons={reasons}"
        )


if __name__ == "__main__":
    main()
