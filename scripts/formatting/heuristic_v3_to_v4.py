"""Measure how much of canonical v4 tokenization can be recovered from v3 ids.

The migration function itself consumes only v3 token ids and the FRIDA
tokenizer. Raw item fields are used by the experiment only to build the true
v4 target and identify rows whose longer v4 fields were already truncated by
v3 and therefore cannot be recovered from the cache.
"""
import json
import re

import polars as pl
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

MODEL = "ai-forever/FRIDA"
N_SAMPLE = 5_000
OLD_NAME_CHARS, OLD_CATEGORY_CHARS, OLD_ATTR_CHARS = 140, 32, 529
OLD_ITEM_CHARS = 731
NEW_NAME_CHARS, NEW_ATTR_CHARS = 160, 640
MAX_LENGTH = 256


def clean_field(value):
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace(" = ", "=")


def demotion_rank(key, value):
    value = value if isinstance(value, str) else str(value)
    rank = int(len(value) > 200)
    if "https" in value.lower():
        rank = 2
    if "oem" in value.lower() or "oem" in str(key).lower():
        rank = 3
    return rank


def reordered_attributes(raw):
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    ranked = sorted(
        ((demotion_rank(key, value), key, value) for key, value in data.items()),
        key=lambda row: row[0],
    )
    return {key: value for _rank, key, value in ranked}


def old_text(name, category, attrs):
    attrs_json = json.dumps(attrs, ensure_ascii=False)[:OLD_ATTR_CHARS]
    body = (
        f"Name: {name[:OLD_NAME_CHARS]} "
        f"Category: {category[:OLD_CATEGORY_CHARS]} Attributes: {attrs_json}"
    )[:OLD_ITEM_CHARS]
    return "paraphrase: " + body


def new_text(name, attrs):
    lines = [
        f"{clean_field(key)} = {clean_field(value)}"
        for key, value in attrs.items()
        if clean_field(key) or clean_field(value)
    ]
    attrs_text = "\n".join(lines)[:NEW_ATTR_CHARS]
    name = clean_field(name)[:NEW_NAME_CHARS]
    return f"{name}\n{attrs_text}" if attrs_text else name


def token_bytes_and_offsets(tokenizer, ids, byte_decoder):
    core = ids[1:-1]
    pieces = []
    for token_id in core:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        try:
            pieces.append(bytes(byte_decoder[char] for char in token))
        except KeyError:
            return None
    raw = b"".join(pieces)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    offsets = []
    position = 0
    for piece in pieces:
        offsets.append((position, position + len(piece)))
        position += len(piece)
    return core, offsets, text


def byte_position(text, char_position):
    return len(text[:char_position].encode("utf-8"))


def ids_inside(core, offsets, start, end):
    return [token_id for token_id, (left, right) in zip(core, offsets)
            if left >= start and right <= end]


def migrate_old_ids(old_ids, tokenizer, byte_decoder):
    decoded = token_bytes_and_offsets(tokenizer, old_ids, byte_decoder)
    if decoded is None:
        return None
    core, offsets, text = decoded
    prefix = "paraphrase: Name: "
    category_marker = " Category: "
    attributes_marker = " Attributes: "
    if not text.startswith(prefix):
        return None
    attributes_at = text.rfind(attributes_marker)
    category_at = text.rfind(category_marker, len(prefix), attributes_at)
    if category_at < 0 or attributes_at < 0:
        return None

    old_name = text[len(prefix):category_at]
    attrs_json = text[attributes_at + len(attributes_marker):]
    try:
        attrs = json.loads(attrs_json)
    except Exception:
        return None
    if not isinstance(attrs, dict):
        return None

    newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    equals_ids = tokenizer(" =", add_special_tokens=False)["input_ids"]
    space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
    copied = retokenized = 0

    clean_name = clean_field(old_name)
    if clean_name != old_name:
        name_ids = tokenizer(clean_name, add_special_tokens=False)["input_ids"]
        retokenized += len(name_ids)
    else:
        match = re.search(r"\s", old_name)
        cut = match.start() if match else len(old_name)
        rest = ids_inside(
            core, offsets,
            byte_position(text, len(prefix) + cut),
            byte_position(text, category_at),
        )
        if tokenizer.decode(rest, clean_up_tokenization_spaces=False) != old_name[cut:]:
            return None
        first = tokenizer(old_name[:cut], add_special_tokens=False)["input_ids"]
        name_ids = first + rest
        retokenized += len(first)
        copied += len(rest)

    lines = []
    rendered_lines = []
    search_from = 1
    for key, value in attrs.items():
        clean_key, clean_value = clean_field(key), clean_field(value)
        if not (clean_key or clean_value):
            continue
        key_json = json.dumps(key, ensure_ascii=False)
        value_json = json.dumps(value, ensure_ascii=False)
        key_at = attrs_json.find(key_json, search_from)
        value_at = attrs_json.find(value_json, key_at + len(key_json)) if key_at >= 0 else -1
        if key_at < 0 or value_at < 0:
            return None

        if isinstance(key, str) and clean_key == key and key_json[1:-1] == key:
            start = attributes_at + len(attributes_marker) + key_at + 1
            end = attributes_at + len(attributes_marker) + key_at + len(key_json) - 1
            key_ids = ids_inside(
                core, offsets, byte_position(text, start), byte_position(text, end),
            )
            if tokenizer.decode(key_ids, clean_up_tokenization_spaces=False) != key:
                return None
            copied += len(key_ids)
        else:
            key_ids = tokenizer(clean_key, add_special_tokens=False)["input_ids"]
            retokenized += len(key_ids)

        if isinstance(value, str) and clean_value == value and value_json[1:-1] == value:
            start = attributes_at + len(attributes_marker) + value_at + 1
            end = attributes_at + len(attributes_marker) + value_at + len(value_json) - 1
            value_ids = ids_inside(
                core, offsets, byte_position(text, start), byte_position(text, end),
            )
            if tokenizer.decode(value_ids, clean_up_tokenization_spaces=False) != value:
                return None
            if value:
                match = re.search(r"\s", value)
                cut = match.start() if match else len(value)
                old_first = tokenizer(value[:cut], add_special_tokens=False)["input_ids"]
                if value_ids[:len(old_first)] != old_first:
                    return None
                new_first = tokenizer(" " + value[:cut], add_special_tokens=False)["input_ids"]
                rendered_value_ids = new_first + value_ids[len(old_first):]
                retokenized += len(new_first)
                copied += len(value_ids) - len(old_first)
            else:
                rendered_value_ids = space_ids
                retokenized += len(space_ids)
        else:
            rendered_value_ids = tokenizer(
                " " + clean_value, add_special_tokens=False,
            )["input_ids"]
            retokenized += len(rendered_value_ids)

        lines.append(key_ids + equals_ids + rendered_value_ids)
        rendered_lines.append(f"{clean_key} = {clean_value}")
        retokenized += len(equals_ids)
        search_from = value_at + len(value_json)

    attrs_text = "\n".join(rendered_lines)
    if len(attrs_text) > NEW_ATTR_CHARS:
        return None

    migrated = [tokenizer.bos_token_id] + name_ids
    if lines:
        migrated += newline_ids
        for index, line in enumerate(lines):
            if index:
                migrated += newline_ids
            migrated += line
    migrated += [tokenizer.eos_token_id]
    if len(migrated) > MAX_LENGTH:
        migrated = migrated[:MAX_LENGTH - 1] + [tokenizer.eos_token_id]
    recovered_text = clean_name + ("\n" + attrs_text if attrs_text else "")
    return migrated, recovered_text, copied, retokenized


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    byte_decoder = {unicode_char: byte for byte, unicode_char in bytes_to_unicode().items()}
    items = pl.read_parquet(
        "data/items_human.parquet", columns=["name", "category", "attributes"],
    ).sample(n=N_SAMPLE, seed=42)

    counts = {"all": 0, "recoverable": 0, "migrated": 0, "text_exact": 0, "ids_exact": 0}
    copied = retokenized = 0
    for name, category, attributes in items.iter_rows():
        name, category = name or "", category or ""
        attrs = reordered_attributes(attributes or "")
        old = old_text(name, category, attrs)
        new = new_text(name, attrs)
        old_ids = tokenizer(
            old, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH,
        )["input_ids"]
        new_ids = tokenizer(
            new, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH,
        )["input_ids"]
        counts["all"] += 1

        full_attrs_json = json.dumps(attrs, ensure_ascii=False)
        recoverable = (
            len(name) <= OLD_NAME_CHARS
            and len(full_attrs_json) <= OLD_ATTR_CHARS
            and len(tokenizer(old, add_special_tokens=True)["input_ids"]) <= MAX_LENGTH
        )
        counts["recoverable"] += int(recoverable)
        migrated = migrate_old_ids(old_ids, tokenizer, byte_decoder)
        if migrated is None:
            continue
        counts["migrated"] += 1
        heuristic_ids, recovered_text, copied_now, retokenized_now = migrated
        copied += copied_now
        retokenized += retokenized_now
        counts["text_exact"] += int(recovered_text == new)
        counts["ids_exact"] += int(recoverable and heuristic_ids == new_ids)

    result = {
        "sample": N_SAMPLE,
        "counts": counts,
        "recoverable_pct": 100 * counts["recoverable"] / counts["all"],
        "canonical_ids_exact_pct_all": 100 * counts["ids_exact"] / counts["all"],
        "canonical_ids_exact_pct_recoverable": (
            100 * counts["ids_exact"] / max(counts["recoverable"], 1)
        ),
        "copied_content_token_share_pct": 100 * copied / max(copied + retokenized, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
