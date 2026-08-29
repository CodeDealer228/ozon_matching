import json

import pandas as pd

from generate_token_replacement_grounded_v1 import target_tokens


def test_target_tokens_use_only_explicit_identity_fields() -> None:
    target = pd.Series({
        "name": "xiaomi redmi note 13 8/256",
        "attributes": json.dumps({
            "бренд": "xiaomi",
            "модель": "redmi note 13",
            "объем памяти": "256 гб",
            "цвет": "черный",
        }, ensure_ascii=False),
    })
    assert target_tokens(target, "brand", "samsung") == ["xiaomi"]
    assert target_tokens(target, "article", "galaxy a14") == ["redmi note 13"]


if __name__ == "__main__":
    test_target_tokens_use_only_explicit_identity_fields()
