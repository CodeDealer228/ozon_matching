from generate_token_replacement_aggressive_rag import (
    VALIDATOR_SYSTEM_PROMPT,
    candidate_tool,
    compact_rag_item,
    semantic_type,
)


operations = [
    {"source_token": "lynxauto", "replacement_token": "Masuma"},
    {"source_token": "sg1909", "replacement_token": "w71295"},
]
tool = candidate_tool(operations, 2)
alternatives = tool["function"]["parameters"]["properties"]["replacements"]["items"]["oneOf"]
assert len(alternatives) == 2
assert alternatives[0]["properties"]["source_token"]["const"] == "lynxauto"
assert alternatives[1]["properties"]["replacement_token"]["const"] == "w71295"
assert semantic_type("sg1909") == "model"
assert semantic_type("xiaomi") == "brand"
assert semantic_type("500") == "size_or_quantity"
assert "ОБЯЗАНА менять идентичность" in VALIDATOR_SYSTEM_PROMPT
assert "не причина для отклонения" in VALIDATOR_SYSTEM_PROMPT
assert "negative-пару разных товаров" in VALIDATOR_SYSTEM_PROMPT
assert "positive-пару" in VALIDATOR_SYSTEM_PROMPT

item = {"name": "товар", "attributes": {"a": "x" * 30, "b": "y" * 30}}
compact = compact_rag_item(item, max_attribute_chars=45)
assert compact["name"] == "товар"
assert list(compact["attributes"]) == ["a"]
