import json
from types import SimpleNamespace

import pandas as pd

from generate_token_replacement_fast_v2 import (
    BankEntry,
    ReplacementBank,
    balanced_sources,
    batch_tool,
    make_batches,
    parse_batch_message,
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
    ("A", "model"): [
        BankEntry("w71295", "model", frozenset({"фильтр", "масляный"}), "x"),
        BankEntry("ab123", "model", frozenset({"другое"}), "y"),
    ]
}
bank.postings = {
    ("A", "model", "фильтр"): [0],
    ("A", "model", "масляный"): [0],
}
bank.frequencies = {("A", "model", "w71295"): 3, ("A", "model", "ab123"): 1}
candidate = bank.candidates("A", "sg1909", {"фильтр", "масляный"}, {"sg1909"}, 1)
assert candidate[0]["replacement_token"] == "w71295"
