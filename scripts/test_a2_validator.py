from types import SimpleNamespace

from generate_a2_pilot_qwen import load_helpers, request_completion, validate_rewrite


def main() -> None:
    helpers = load_helpers()
    row = SimpleNamespace(
        name1="красные ботинки бренд b",
        name2="ботинки красные",
        attributes2='{"бренд":"b","цвет":"красный"}',
    )
    assert validate_rewrite(row, "красные ботинки b", helpers)[0]
    assert "unsupported_added_fact" in validate_rewrite(row, "красные ботинки adidas", helpers)[1]
    assert "lost_name_fact" in validate_rewrite(row, "ботинки", helpers)[1]

    seen: dict[str, object] = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"name2":"ботинки"}'))],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 7}),
            id="request-1",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    content, usage, transport = request_completion(client, [{"role": "user", "content": "x"}], 42, 300)
    assert content == '{"name2":"ботинки"}' and usage["total_tokens"] == 7
    assert transport["provider_request_id"] == "request-1"
    assert "top_p" not in seen and seen["extra_body"] == {"enable_thinking": False}


if __name__ == "__main__":
    main()
