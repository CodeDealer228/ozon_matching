import json
from types import SimpleNamespace

import pandas as pd

from generate_token_replacement_fast_v2 import (
    BankEntry,
    ReplacementBank,
    balanced_sources,
    batch_tool,
    infer_source_role,
    make_batches,
    parse_batch_message,
    role_values,
    validate_batch_decisions,
)


rows = [
    {
        "candidate_operations": [
            {"operation_id": "op0", "source_token": "lynxauto", "replacement_token": "masuma"},
            {"operation_id": "op1", "source_token": "sg1909", "replacement_token": "w71295"},
        ]
    },
    {
        "candidate_operations": [
            {"operation_id": "op0", "source_token": "500", "replacement_token": "750"},
        ]
    },
]
tool = batch_tool(rows, 2)
schema = tool["function"]["parameters"]["properties"]["decisions"]
assert schema["minItems"] == 2
assert schema["items"]["oneOf"][0]["properties"]["slot"]["const"] == 0

value = {
    "decisions": [
        {"slot": 0, "status": "replace", "operation_ids": ["op0", "op1"], "skip_reason": ""},
        {"slot": 1, "status": "skip", "operation_ids": [], "skip_reason": "uncertain"},
    ]
}
decisions = validate_batch_decisions(value, rows, 2)
assert decisions[0]["replacements"][1]["replacement_token"] == "w71295"
assert decisions[1]["status"] == "skip"
message = SimpleNamespace(
    tool_calls=[SimpleNamespace(function=SimpleNamespace(name="submit_batched_replacement_decisions", arguments=json.dumps(value)))]
)
assert parse_batch_message(message, rows, 2) == decisions

tolerant = {
    "decisions": [
        {
            "slot": "0",
            "status": "replace",
            "operation_ids": ["op0", "op1"],
            "skip_reason": "",
            "extra_explanation": "ignored",
        },
        {"slot": 1, "status": "skip", "operation_ids": [], "skip_reason": ""},
    ]
}
tolerant_decisions = validate_batch_decisions(tolerant, rows, 2)
assert tolerant_decisions[0]["replacements"][0]["source_token"] == "lynxauto"
assert tolerant_decisions[1]["skip_reason"] == "model_skip"

source_rows = [
    {"category": "A", "source_pair_sha256": "a"},
    {"category": "A", "source_pair_sha256": "b"},
    {"category": "B", "source_pair_sha256": "c"},
    {"category": "B", "source_pair_sha256": "d"},
]
assert len(balanced_sources(source_rows, {"a"}, 2)) == 2
batches = make_batches(source_rows, 1)
assert len(batches) == 4
assert [batch[0]["category"] for batch in batches[:2]] == ["A", "B"]

bank = ReplacementBank.__new__(ReplacementBank)
bank.entries = {
    ("A", "article", "model"): [
        BankEntry("w71295", "model", "article", frozenset({"фильтр", "масляный"}), "x", "фильтр масляный w71295"),
        BankEntry("ab123", "model", "article", frozenset({"другое"}), "y", "другой товар ab123"),
    ]
}
bank.postings = {
    ("A", "article", "model", "фильтр"): [0],
    ("A", "article", "model", "масляный"): [0],
}
bank.frequencies = {
    ("A", "article", "model", "w71295"): 3,
    ("A", "article", "model", "ab123"): 1,
}
candidate = bank.candidates(
    "A", "sg1909", "article", {"фильтр", "масляный"}, {"sg1909"}, 1
)
assert candidate[0]["replacement_token"] == "w71295"

left = pd.Series({
    "name": "набор 100штук",
    "attributes": json.dumps({"количество": "100"}, ensure_ascii=False),
})
right = pd.Series({
    "name": "набор 100штук",
    "attributes": json.dumps({"количество предметов": "100"}, ensure_ascii=False),
})
assert infer_source_role(left, right, "100штук") == "count"
assert role_values(
    {"name": "набор 50 штук", "attributes": {"количество": "50"}},
    "count",
    "100штук",
) == ["50штук"]
