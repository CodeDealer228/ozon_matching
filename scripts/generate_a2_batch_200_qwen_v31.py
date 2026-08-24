"""Run the disjoint 200-source A2 v3.1 continuation batch."""

from __future__ import annotations

import sys

import generate_a2_pilot_qwen_v3 as engine
import generate_a2_pilot_qwen_v31 as v31


PROFILE = "a2_qwen_contract_v31_scale_200_02"
EXPECTED_SOURCE_SHA256 = (
    "77b9828a0e811ff41c4c4f68e3602b5a0a6d65575106a994084bc0edbe9655b6"
)
EXPECTED_PLAN_SHA256 = (
    "0ed0b2a75009b40dfcaa1ecd9d8564dec7b114a813003a94d2e7dc444177f6f8"
)
EXPECTED_FEW_SHOT_SHA256 = (
    "1f6ed4386bdbb9b0b3fd789036993778f7d21fdeeb372abc3334b8634e705057"
)


def add_default_argument(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend((flag, value))


def main() -> None:
    repo = engine.REPO_ROOT
    add_default_argument(
        "--source-pool",
        str(repo / "artifacts/synthetic_v2/a2_v31_batch_200_02_plan/source_pool.parquet"),
    )
    add_default_argument(
        "--plan",
        str(repo / "artifacts/synthetic_v2/a2_v31_batch_200_02_plan/generation_plan.jsonl"),
    )
    add_default_argument("--few-shot", str(repo / "synthetic_research/a2_few_shot_v31.json"))
    add_default_argument(
        "--output-dir", str(repo / "artifacts/synthetic_v2/a2_v31_batch_200_02")
    )

    engine.PROFILE = PROFILE
    engine.EXPECTED_SOURCE_SHA256 = EXPECTED_SOURCE_SHA256
    engine.EXPECTED_PLAN_SHA256 = EXPECTED_PLAN_SHA256
    engine.EXPECTED_FEW_SHOT_SHA256 = EXPECTED_FEW_SHOT_SHA256
    engine.EXPECTED_PILOT_ROWS = 200
    engine.CANDIDATE_COUNT_LABEL = "exactly_2_same_facts_different_placement"
    engine.MAX_COMPLETION_TOKENS = 320
    engine.build_messages = v31.build_messages
    engine.parse_response = v31.parse_response
    engine.main()


if __name__ == "__main__":
    main()
