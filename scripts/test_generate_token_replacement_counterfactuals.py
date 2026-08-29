import json
import tempfile
from pathlib import Path

import pandas as pd

from generate_token_replacement_counterfactuals import (
    decision_tool,
    load_few_shots,
    materialize,
    mutate_item,
    offered_tokens,
    replace_text,
    validate_decision,
)


left = pd.Series({
    "name": "lynxauto фильтр sg1909",
    "category": "Автотовары",
    "attributes": json.dumps({"бренд": "lynxauto", "артикул": "sg1909"}),
})
right = pd.Series({
    "name": "фильтр LYNXAUTO sg1909",
    "category": "Автотовары",
    "attributes": json.dumps({"бренд": "lynxauto", "артикул": "sg1909"}),
})

tokens = offered_tokens(left, right, 12)
assert "lynxauto" in tokens and "sg1909" in tokens
assert decision_tool(tokens, 2)["function"]["parameters"]["properties"]["replacements"]["items"]["properties"]["source_token"]["enum"] == tokens
assert validate_decision(
    {
        "status": "replace",
        "replacements": [{"source_token": "lynxauto", "replacement_token": "Masuma"}],
        "skip_reason": "",
    },
    tokens,
    2,
)["status"] == "replace"
assert validate_decision(
    {"status": "skip", "replacements": [], "skip_reason": "нет безопасной замены"},
    tokens,
    2,
)["status"] == "skip"

try:
    validate_decision(
        {
            "status": "replace",
            "replacements": [{"source_token": "не предлагали", "replacement_token": "x"}],
            "skip_reason": "",
        },
        tokens,
        2,
    )
except ValueError as exc:
    assert str(exc) == "source_token_not_offered"
else:
    raise AssertionError("unoffered source token was accepted")

assert replace_text("16 160 x16", "16", "14") == ("14 160 x16", 1)
mutated, changes = mutate_item(left, "lynxauto", "Masuma", -1)
assert changes == 2
assert mutated["name"].startswith("Masuma")
assert json.loads(mutated["attributes"])["бренд"] == "Masuma"

items = pd.DataFrame([
    {"id": 1, **left.to_dict()},
    {"id": 2, **right.to_dict()},
]).set_index("id", drop=False)
synthetic_items, synthetic_pairs = materialize([
    {
        "id1": 1,
        "id2": 2,
        "source_pair_sha256": "a" * 64,
        "request_status": "complete",
        "decision": {
            "status": "replace",
            "replacements": [
                {"source_token": "lynxauto", "replacement_token": "Masuma"},
                {"source_token": "sg1909", "replacement_token": "w71295"},
            ],
            "skip_reason": "",
        },
    }
], items)
assert len(synthetic_items) == 4
assert len(synthetic_pairs) == 4
assert synthetic_pairs["target"].value_counts().to_dict() == {1: 2, 0: 2}
assert set(synthetic_pairs["variant"]) <= {"both_replaced", "left_only", "right_only"}


with tempfile.TemporaryDirectory() as tmp_dir:
    path = Path(tmp_dir) / "few_shots.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "categories": {
            "Автотовары": [{
                "source_reference_local_only": {"id1": 1, "id2": 2},
                "input": {
                    "category": "Автотовары",
                    "item_1": {"name": "a", "attributes": {}},
                    "item_2": {"name": "b", "attributes": {}},
                    "offered_tokens": ["lynxauto"],
                },
                "decision": {
                    "status": "replace",
                    "replacements": [{
                        "source_token": "lynxauto",
                        "replacement_token": "Masuma",
                    }],
                    "skip_reason": "",
                },
                "reviewer_note_local_only": "private note",
            }],
        },
    }, ensure_ascii=False), encoding="utf-8")
    loaded = load_few_shots(path, 2)
    assert loaded["Автотовары"][0]["decision"]["status"] == "replace"
    assert set(loaded["Автотовары"][0]) == {"input", "decision"}
