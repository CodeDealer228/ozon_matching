import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import validate_grounded_generation_dual as module
from validate_grounded_generation_dual import load_candidates, parse


def test_parse_dual_validator() -> None:
    message = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(
        name="validate_grounded_bundles",
        arguments='{"decisions":[{"slot":0,"status":"keep","reason":"ok"}]}',
    ))])
    assert parse(message, [{}]) == [{"status": "keep", "reason": "ok"}]


def test_load_candidates_recovers_from_journal_when_preview_is_empty(tmp_path: Path) -> None:
    preview = tmp_path / "preview.json"
    preview.write_text("[]\n", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps({
        "request_status": "complete",
        "results": [{
            "category": "test",
            "operation_sha256": "stale-operation",
            "source_role": "brand",
            "decision": {"status": "choose", "bundle_id": "bundle-1"},
            "selected_bundle": {
                "bundle_id": "bundle-1",
                "mappings": [{"source_span": "a", "target_span": "b"}],
                "preview_name_1": "b one",
                "preview_name_2": "b two",
                "target_card": {"name": "b", "attributes": {}},
            },
        }],
    }) + "\n", encoding="utf-8")
    pairs = pd.DataFrame([{
        "bundle_id": "bundle-1",
        "operation_sha256": "current-operation",
        "source_id1": 1,
        "source_id2": 2,
        "source_role": "brand",
    }])
    items = pd.DataFrame([
        {"id": 1, "name": "a one", "attributes": {}},
        {"id": 2, "name": "a two", "attributes": {}},
    ])
    original = module.pd.read_parquet
    module.pd.read_parquet = lambda path, columns=None: (
        pairs[columns] if str(path).endswith("pairs.parquet") else items[columns]
    )
    try:
        rows = load_candidates(
            preview, journal, tmp_path / "pairs.parquet", tmp_path / "items.parquet"
        )
    finally:
        module.pd.read_parquet = original
    assert len(rows) == 1
    assert rows[0]["operation_sha256"] == "current-operation"
    assert rows[0]["item_1"]["name"] == "a one"


if __name__ == "__main__":
    test_parse_dual_validator()
