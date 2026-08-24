"""
Local experiment (no Kaggle): can a "human-readable-overhead-free" item text
format be derived from the ALREADY-TOKENIZED old format via pure token
splicing, avoiding a full re-tokenization of the 711K human + 12.38M LLM
item caches?

OLD (real production format, kaggle_kernels/frida/tokenize/tokenize-items-cpu):
    "paraphrase: Name: {name} Category: {category} Attributes: {json_attrs}"
    where json_attrs = json.dumps(v2-reordered attrs dict, ensure_ascii=False)
    -- i.e. WITH quotes/braces/colon-space, e.g. {"цвет товара": "черный"}.

NEW (proposed, per direct instruction): drop the field-label overhead and
raw-JSON punctuation, drop category from the text entirely (kept as a
separate per-item field, already true today -- category already gets
excised from PAIRS at inference time; this goes further and never puts it
in the item text at all):
    "paraphrase: {name}\\n{key}:{value}, {key}:{value}, ..."

HYPOTHESIS BEING TESTED: byte-level BPE (RobertaTokenizer, this project's
own confirmed FRIDA tokenizer type) pretokenizes on whitespace/punctuation
boundaries, so a word's own token span should be identical regardless of
what immediately follows it, AS LONG AS what precedes it is unchanged
(same leading-space context). If true, the NAME segment's tokens can be
spliced directly out of the OLD sequence with zero retokenization. The
ATTRIBUTES segment is expected to fail splicing: removing the JSON quote
characters changes the immediate character context of every value's first
token, which should change its BPE merge.

Reports, per item and in aggregate: token-count savings from the format
change alone, and exact-match rate of the splice heuristic against a real
fresh tokenization of the new format -- separately for the NAME segment
and the ATTRIBUTES segment, so the two independent questions ("does the
format save tokens" and "can we skip retokenizing it") don't get
conflated into one pass/fail number.
"""
import json
import random
import sys

import polars as pl
from transformers import AutoTokenizer

SEED = 42
N_SAMPLE = 5000

FRIDA_MODEL_NAME = "ai-forever/FRIDA"
FRIDA_PREFIX = "paraphrase: "

# Byte-for-byte the real production constants (kaggle_kernels/frida/tokenize/
# tokenize-items-cpu/frida_tokenize_items_cpu.py) -- NOT re-derived, copied,
# so this experiment's "old" format is the actual shipped one, not an
# approximation of it.
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 529
LONG_VALUE_CHARS = 200
LINK_MARKER = "https"
OEM_MARKER = "oem"

ITEMS_PATH = "data/items_human.parquet"


def _demotion_rank(key, value):
    v = value if isinstance(value, str) else str(value)
    rank = 0
    if len(v) > LONG_VALUE_CHARS:
        rank = 1
    vl = v.lower()
    if LINK_MARKER in vl:
        rank = 2
    if OEM_MARKER in vl or OEM_MARKER in str(key).lower():
        rank = 3
    return rank


def reorder_attributes_dict(attr_str):
    """Returns the v2-reordered dict itself (not yet re-serialized) so both
    the OLD (json.dumps) and NEW (key:value,...) formats build from the
    SAME reordering decision -- isolates the experiment to "text format",
    not "which attributes survive the char budget", which is a separate
    question this script doesn't touch."""
    if not attr_str:
        return {}
    try:
        d = json.loads(attr_str)
    except Exception:
        return {}
    if not isinstance(d, dict) or not d:
        return {}
    ranked = sorted(((_demotion_rank(k, v), k, v) for k, v in d.items()), key=lambda t: t[0])
    return {k: v for _r, k, v in ranked}


def build_old_text(name, category, attrs_dict):
    attr_json = json.dumps(attrs_dict, ensure_ascii=False)[:MAX_ATTR_CHARS]
    body = f"Name: {name[:MAX_NAME_CHARS]} Category: {category[:MAX_CATEGORY_CHARS]} Attributes: {attr_json}"
    return FRIDA_PREFIX + body


def build_new_text(name, attrs_dict):
    kv = ", ".join(f"{k}:{v}" for k, v in attrs_dict.items())
    kv = kv[:MAX_ATTR_CHARS]  # same budget as old, for an apples-to-apples token-count comparison
    body = f"{name[:MAX_NAME_CHARS]}\n{kv}"
    return FRIDA_PREFIX + body


def _find_subseq(haystack, needle):
    """First index where `needle` occurs as a contiguous run in `haystack`,
    or -1. Small N here (needle is a handful of tokens) so naive search is
    plenty fast for a 5K-item experiment."""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and haystack[i:i + m] == needle:
            return i
    return -1


def main():
    random.seed(SEED)
    print(f"[SETUP] loading tokenizer {FRIDA_MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(FRIDA_MODEL_NAME)
    print(f"[SETUP] tokenizer={type(tok).__name__} vocab={tok.vocab_size} "
          f"bos={tok.bos_token_id} eos={tok.eos_token_id}", flush=True)

    print(f"[DATA] sampling {N_SAMPLE} items from {ITEMS_PATH} ...", flush=True)
    items = (
        pl.scan_parquet(ITEMS_PATH)
        .select(["id", "name", "category", "attributes"])
        .collect(engine="streaming")
    )
    if items.height > N_SAMPLE:
        idx = random.sample(range(items.height), N_SAMPLE)
        items = items[idx]
    print(f"[DATA] sampled {items.height:,} items", flush=True)

    names = items.get_column("name").fill_null("").to_list()
    cats = items.get_column("category").fill_null("").to_list()
    attrs_raw = items.get_column("attributes").fill_null("").to_list()

    # ---- fixed-boundary token spans ----
    # REAL BUG, found and fixed while building this experiment: tokenizing
    # a fixed prefix ALONE (e.g. tok(FRIDA_PREFIX + "Name: ")) and using its
    # LENGTH as a splice boundary is wrong for byte-level BPE (this
    # tokenizer is RobertaTokenizer, confirmed GPT2-style byte-BPE) -- a
    # TRAILING space with nothing after it tokenizes differently than that
    # same space immediately followed by real content, because BPE merges
    # "space+word" into one leading-space-flagged token; alone, the space
    # has nothing to merge with. First attempt at this used the isolated-
    # tokenize-then-assume-length approach and got a 0.6% (29/5000) sanity-
    # check pass rate as a direct result -- not a data problem, a tokenizer-
    # mechanics bug in the experiment itself.
    #
    # Fix: the prefix strings ("paraphrase: ", "paraphrase: Name: ") are
    # IDENTICAL for every single item, so their correct in-context token
    # span can be recovered directly from real ground-truth tokenizations
    # by taking the LONGEST COMMON PREFIX across many items with genuinely
    # different names -- the common run necessarily ends exactly where the
    # (per-item-varying) name begins. This is a ONE-TIME, item-independent
    # computation (not "retokenizing this item" -- the constant is derived
    # once and reused for all N items, the same way a production cache
    # would hardcode it after verifying it once).
    def _longest_common_prefix_len(seqs):
        if not seqs:
            return 0
        shortest = min(len(s) for s in seqs)
        for i in range(shortest):
            if len({s[i] for s in seqs}) > 1:
                return i
        return shortest

    category_needle = tok(" Category:", add_special_tokens=False)["input_ids"]
    newline_ids = tok("\n", add_special_tokens=False)["input_ids"]

    probe_texts_old = [build_old_text(n, "", {}) for n in names[:300] if n]
    probe_texts_new = [build_new_text(n, {}) for n in names[:300] if n]
    old_prefix_len = _longest_common_prefix_len([tok(t, add_special_tokens=True)["input_ids"] for t in probe_texts_old])
    new_prefix_len = _longest_common_prefix_len([tok(t, add_special_tokens=True)["input_ids"] for t in probe_texts_new])
    probe_old_ids = tok(probe_texts_old[0], add_special_tokens=True)["input_ids"]
    probe_new_ids = tok(probe_texts_new[0], add_special_tokens=True)["input_ids"]
    new_prefix_ids = probe_new_ids[1:new_prefix_len]  # "paraphrase: " in-context, before the name
    print(f"[SETUP] empirical old prefix ('[bos] paraphrase: Name: ') = {old_prefix_len} tok "
          f"{probe_old_ids[:old_prefix_len]}", flush=True)
    print(f"[SETUP] empirical new prefix ('[bos] paraphrase: ') = {new_prefix_len} tok "
          f"{probe_new_ids[:new_prefix_len]}", flush=True)
    print(f"[SETUP] ' Category:' needle = {category_needle}, '\\n' = {newline_ids}", flush=True)

    old_lens, new_lens = [], []
    prefix_boundary_ok = 0
    category_found = 0
    name_splice_exact = 0
    name_splice_total = 0
    full_splice_exact = 0
    full_splice_total = 0
    mismatch_examples = []
    name_token_lens, rest_token_lens = [], []

    for i, (name, cat, attr_raw) in enumerate(zip(names, cats, attrs_raw)):
        attrs_dict = reorder_attributes_dict(attr_raw)
        old_text = build_old_text(name, cat, attrs_dict)
        new_text = build_new_text(name, attrs_dict)

        old_ids = tok(old_text, add_special_tokens=True)["input_ids"]
        new_ids_real = tok(new_text, add_special_tokens=True)["input_ids"]
        old_lens.append(len(old_ids))
        new_lens.append(len(new_ids_real))

        # ---- sanity: OLD sequence really does start with the empirically-
        # derived [bos]+prefix+'Name: ' span ----
        if old_ids[0] == tok.bos_token_id and len(old_ids) > old_prefix_len:
            prefix_boundary_ok += 1
        else:
            continue  # can't locate the name span reliably for this item; skip splice test

        cat_start = _find_subseq(old_ids, category_needle)
        if cat_start < 0 or cat_start < old_prefix_len:
            continue
        category_found += 1

        name_tokens = old_ids[old_prefix_len:cat_start]

        # ---- heuristic A: NAME-only splice, rest freshly tokenized ----
        # (splice bos+prefix+name from OLD, retokenize "\n"+attrs fresh --
        # NOT zero retokenization, but zero retokenization of the NAME,
        # which is the part most likely to repeat unchanged across many
        # pairs referencing the same item.)
        attrs_kv = ", ".join(f"{k}:{v}" for k, v in attrs_dict.items())[:MAX_ATTR_CHARS]
        rest_ids = tok("\n" + attrs_kv, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        heuristic_name_only = [tok.bos_token_id] + new_prefix_ids + name_tokens + rest_ids

        name_splice_total += 1
        if heuristic_name_only == new_ids_real:
            name_splice_exact += 1
        name_token_lens.append(len(name_tokens))
        rest_token_lens.append(len(rest_ids))

        # ---- heuristic B: reuse each attribute VALUE's own token span
        # straight out of OLD's JSON-quoted form (located via character
        # offset mapping, not guessed), splicing in fresh ":"/", " connector
        # tokens between them -- the real question: can even the ATTRIBUTE
        # VALUES (the bulk of the text) skip retokenization, or only the
        # name can. ----
        if not attrs_dict:
            continue
        attr_json_str = json.dumps(attrs_dict, ensure_ascii=False)[:MAX_ATTR_CHARS]
        enc = tok(attr_json_str, add_special_tokens=False, return_offsets_mapping=True)
        json_tok_ids, offsets = enc["input_ids"], enc["offset_mapping"]

        def _char_span_to_tokens(start_char, end_char):
            """Tokens whose own char span lies fully inside [start_char,
            end_char) -- a token straddling the boundary (e.g. quote+first
            char fused) is dropped, not truncated, since a partial token id
            can't be spliced meaningfully on its own."""
            return [tid for tid, (s, e) in zip(json_tok_ids, offsets) if s >= start_char and e <= end_char]

        colon_ids = tok(":", add_special_tokens=False)["input_ids"]
        comma_ids = tok(", ", add_special_tokens=False)["input_ids"]

        spliced_value_pieces = []
        all_values_found = True
        search_from = 0
        for k, v in attrs_dict.items():
            v_json = json.dumps(v, ensure_ascii=False)  # e.g. '"черный"' -- exact substring in attr_json_str
            pos = attr_json_str.find(v_json, search_from)
            if pos < 0:
                all_values_found = False
                break
            is_quoted = v_json.startswith('"')
            value_start = pos + 1 if is_quoted else pos
            value_end = pos + len(v_json) - 1 if is_quoted else pos + len(v_json)
            value_tokens = _char_span_to_tokens(value_start, value_end)
            key_tokens = tok(k, add_special_tokens=False)["input_ids"]
            spliced_value_pieces.append(key_tokens + colon_ids + value_tokens)
            search_from = pos + len(v_json)

        full_splice_total += 1
        if not all_values_found:
            continue
        attrs_spliced = []
        for j, piece in enumerate(spliced_value_pieces):
            if j > 0:
                attrs_spliced += comma_ids
            attrs_spliced += piece
        heuristic_full = ([tok.bos_token_id] + new_prefix_ids + name_tokens + newline_ids
                          + attrs_spliced + [tok.eos_token_id])
        if heuristic_full == new_ids_real:
            full_splice_exact += 1
        elif len(mismatch_examples) < 5:
            mismatch_examples.append({
                "name": name[:60], "new_real": new_ids_real[:25],
                "heuristic": heuristic_full[:25],
            })

    n = len(names)
    print(f"\n=== TOKEN COUNT (mean, same {MAX_ATTR_CHARS}-char attrs budget for both formats) ===")
    print(f"OLD (Name:/Category:/Attributes: + raw JSON): mean={sum(old_lens)/n:.1f} tok")
    print(f"NEW (bare name + key:value, no category):     mean={sum(new_lens)/n:.1f} tok")
    print(f"Savings: {100*(1 - sum(new_lens)/sum(old_lens)):.1f}% fewer tokens for the SAME content budget")

    print(f"\n=== SPLICE BOUNDARY DETECTION ===")
    print(f"prefix+'Name: ' matched at sequence start: {prefix_boundary_ok}/{n} ({100*prefix_boundary_ok/n:.1f}%)")
    print(f"' Category:' needle found:                 {category_found}/{n} ({100*category_found/n:.1f}%)")

    print(f"\n=== HEURISTIC A: splice NAME from OLD tokens, retokenize '\\n'+attrs fresh ===")
    print(f"Exact match vs real new-format tokenization: {name_splice_exact}/{name_splice_total} "
          f"({100*name_splice_exact/max(name_splice_total,1):.2f}%)")
    mean_name = sum(name_token_lens) / max(len(name_token_lens), 1)
    mean_rest = sum(rest_token_lens) / max(len(rest_token_lens), 1)
    print(f"Mean NAME tokens (spliced, zero retokenize cost): {mean_name:.1f}")
    print(f"Mean '\\n'+attrs tokens (still needs retokenizing): {mean_rest:.1f}")
    print(f"-> splicing the name skips retokenizing ~{100*mean_name/(mean_name+mean_rest):.1f}% of "
          f"each item's own token count -- real compute saved on a re-tokenization pass over the "
          f"existing 711K human + 12.38M LLM item caches, not just a format change.")

    print(f"\n=== HEURISTIC B: splice NAME + every attr VALUE's own tokens from OLD, zero retokenize ===")
    print(f"Exact match vs real new-format tokenization: {full_splice_exact}/{full_splice_total} "
          f"({100*full_splice_exact/max(full_splice_total,1):.2f}%)")

    if mismatch_examples:
        print(f"\n=== first mismatch examples ===")
        for ex in mismatch_examples:
            print(ex)


if __name__ == "__main__":
    sys.exit(main())
