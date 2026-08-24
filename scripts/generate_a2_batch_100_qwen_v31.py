"""Run the first 100-source A2 v3.1 scale batch."""

from __future__ import annotations

import sys

import generate_a2_pilot_qwen_v3 as engine
import generate_a2_pilot_qwen_v31 as v31


PROFILE = "a2_qwen_contract_v31_scale_100_01"
EXPECTED_SOURCE_SHA256 = (
    "c7b8e8b02648dcb880a567bb6fb3be6a7e42740b42f7ad385543fc01724a1f0f"
)
EXPECTED_PLAN_SHA256 = (
    "8de3ea52eae53b5c1be76c4a4b7863afa93004dc7f98b23c3e70f75e5f902da2"
)
EXPECTED_FEW_SHOT_SHA256 = (
    "1f6ed4386bdbb9b0b3fd789036993778f7d21fdeeb372abc3334b8634e705057"
)
EXPECTED_ROWS = 100


def add_default_argument(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend((flag, value))


def main() -> None:
    repo = engine.REPO_ROOT
    add_default_argument(
        "--source-pool",
        str(repo / "artifacts/synthetic_v2/a2_v31_batch_100_01_plan/source_pool.parquet"),
    )
    add_default_argument(
        "--plan",
        str(repo / "artifacts/synthetic_v2/a2_v31_batch_100_01_plan/generation_plan.jsonl"),
    )
    add_default_argument("--few-shot", str(repo / "synthetic_research/a2_few_shot_v31.json"))
    add_default_argument(
        "--output-dir", str(repo / "artifacts/synthetic_v2/a2_v31_batch_100_01")
    )

    engine.PROFILE = PROFILE
    engine.EXPECTED_SOURCE_SHA256 = EXPECTED_SOURCE_SHA256
    engine.EXPECTED_PLAN_SHA256 = EXPECTED_PLAN_SHA256
    engine.EXPECTED_FEW_SHOT_SHA256 = EXPECTED_FEW_SHOT_SHA256
    engine.EXPECTED_PILOT_ROWS = EXPECTED_ROWS
    engine.CANDIDATE_COUNT_LABEL = "exactly_2_same_facts_different_placement"
    engine.MAX_COMPLETION_TOKENS = 320
    engine.build_messages = v31.build_messages
    engine.parse_response = v31.parse_response
    engine.main()


if __name__ == "__main__":
    main()
