from types import SimpleNamespace

from validate_grounded_generation_dual import parse


def test_parse_dual_validator() -> None:
    message = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(
        name="validate_grounded_bundles",
        arguments='{"decisions":[{"slot":0,"status":"keep","reason":"ok"}]}',
    ))])
    assert parse(message, [{}]) == [{"status": "keep", "reason": "ok"}]


if __name__ == "__main__":
    test_parse_dual_validator()
