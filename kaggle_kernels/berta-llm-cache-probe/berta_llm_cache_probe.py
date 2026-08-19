"""Cheap CPU-only probe: verify the real content of
maxamsecret/berta-tokenized-items-llm before trusting it for a real
training run. cache_meta.json claims prefix="" (empty) and max_length=256,
but the training script that produced the CURRENT best checkpoint
(berta-lora-sft-human-only-20-epochs) uses PREFIX="paraphrase: " and
ITEM_MAX_LENGTH=128 for the HUMAN-scope cache -- need to know for real
whether the LLM-scope cache actually has the prefix baked in (metadata
field could just be an unfilled/wrong default), and confirm the real
per-item length distribution at max_length=256 before designing a
continue-train script against it.
"""
import os
import json

import numpy as np
from transformers import AutoTokenizer

CACHE_DIR_HINT = "/kaggle/input/datasets/maxamsecret/berta-tokenized-items-llm"


def find_cache_dir():
    if os.path.isdir(CACHE_DIR_HINT) and os.path.exists(os.path.join(CACHE_DIR_HINT, "ids.bin")):
        return CACHE_DIR_HINT
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "ids.bin" in files:
            return root
    raise RuntimeError("cache dir not found")


def main():
    cache_dir = find_cache_dir()
    print(f"[PROBE] cache_dir={cache_dir}")

    with open(os.path.join(cache_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    print(f"[PROBE] cache_meta.json: {meta}")

    n_items = meta["n_items"]
    max_length = meta["max_length"]

    ids = np.memmap(os.path.join(cache_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    input_ids = np.memmap(os.path.join(cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                           shape=(n_items, max_length))
    attn = np.memmap(os.path.join(cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                      shape=(n_items, max_length))

    print(f"[PROBE] real file sizes: ids={os.path.getsize(os.path.join(cache_dir, 'ids.bin')):,} "
          f"input_ids={os.path.getsize(os.path.join(cache_dir, 'input_ids.bin')):,} "
          f"attn={os.path.getsize(os.path.join(cache_dir, 'attention_mask.bin')):,}")
    expected_input_ids_bytes = n_items * max_length * 4
    expected_attn_bytes = n_items * max_length * 1
    print(f"[PROBE] expected (n_items*max_length*dtype): input_ids={expected_input_ids_bytes:,} "
          f"attn={expected_attn_bytes:,}")

    tokenizer = AutoTokenizer.from_pretrained("sergeyzh/BERTA")
    print(f"[PROBE] tokenizer cls={tokenizer.cls_token_id} sep={tokenizer.sep_token_id} "
          f"pad={tokenizer.pad_token_id}")

    lengths = attn.astype(np.int64).sum(axis=1)
    print(f"[PROBE] real length: mean={lengths.mean():.1f} p50={np.percentile(lengths, 50):.0f} "
          f"p90={np.percentile(lengths, 90):.0f} p99={np.percentile(lengths, 99):.0f} "
          f"max={lengths.max()} at_cap(=={max_length})={(lengths == max_length).sum():,} "
          f"({100 * (lengths == max_length).mean():.2f}%)")

    print("\n[PROBE] decoding 10 random real items (full, incl. special tokens):")
    rng = np.random.RandomState(42)
    sample_rows = rng.choice(n_items, size=10, replace=False)
    for r in sample_rows:
        L = int(lengths[r])
        toks = input_ids[r, :L].tolist()
        text = tokenizer.decode(toks)
        starts_with_cls = toks[0] == tokenizer.cls_token_id
        ends_with_sep = toks[-1] == tokenizer.sep_token_id
        print(f"  row={r} id={int(ids[r])} len={L} starts_cls={starts_with_cls} ends_sep={ends_with_sep}")
        print(f"    text[:200]={text[:200]!r}")

    print("\n[PROBE] checking for 'paraphrase' substring in decoded text (prefix check):")
    n_with_prefix = 0
    for r in sample_rows:
        L = int(lengths[r])
        text = tokenizer.decode(input_ids[r, :L].tolist())
        if "paraphrase" in text.lower():
            n_with_prefix += 1
    print(f"[PROBE] {n_with_prefix}/10 sampled items contain 'paraphrase' in decoded text")

    print("\n[PROBE] checking id overlap with matches_llm.parquet scope (sanity, first 5 ids):")
    print(f"[PROBE] first 5 ids in cache: {ids[:5].tolist()}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
