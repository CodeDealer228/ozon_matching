"""
FRIDA item-level token cache, v2 -- rebuilds maxamsecret/frida-tokenized-
items-human (711,304 items, "paraphrase: " prefix, bos/eos baked in,
max_length=256) with a NEW attribute-ordering policy, and emits TWO
pre-tokenized variants per item so the training kernel can apply the
policy's random component at ZERO train-time cost.

===========================================================================
The new item-input policy (per explicit instruction)
===========================================================================
`attributes` is a JSON object of "key":"value" pairs. Applied in order:
  1. with 5% probability, shuffle the attribute order
  2. any attribute whose VALUE is longer than 200 chars -> move to the end
  3. then any attribute containing a link ("https")     -> move to the end
  4. then any attribute containing "oem"                -> move to the end

Rules 2-4 are "move to the end" applied SEQUENTIALLY, so an attribute
matching a later rule ends up after one matching an earlier rule, and an
attribute matching several rules is positioned by the LAST one it matches.
That is exactly a STABLE SORT by "index of the last matching rule"
(0=normal, 1=long, 2=https, 3=oem), which is how it's implemented below --
same result, one pass, and relative order inside each group is preserved.

WHY this policy helps here: PAIR_MAX_LENGTH=128 truncates ~94% of real
pairs (this lineage's own measurement), so what sits at the FRONT of the
attribute list is what the model actually gets to see. Rules 2-4 demote
exactly the fields that are long and low-signal for duplicate detection --
OEM part-number dumps, URLs, giant free-text blobs -- pushing genuinely
discriminative short attributes (size, colour, article) into the surviving
window.

--- One deliberate deviation from the literal instruction, measured ---
The instruction says rule 4 tests the attribute's VALUE for "oem". Measured
on 50,000 real items (551,150 attributes) BEFORE writing this:
    "oem" in KEY:   8,750 attrs (1.588%) -- affects 17.35% of items
    "oem" in VALUE:    34 attrs (0.006%) -- affects  0.06% of items
In this dataset "oem" is essentially always a KEY ("oem-номер", whose value
is a semicolon-separated part-number dump -- precisely the noise this rule
exists to demote). Testing the value alone would make rule 4 a near-no-op
(0.06% of items). So rule 4 here tests KEY **or** VALUE, case-insensitive.
Rules 2 and 3 are value-only exactly as instructed (measured: 0 keys
contain "https", so key-testing there would add nothing anyway).

===========================================================================
Two variants, zero train-time overhead
===========================================================================
Rule 1's 5% shuffle is random per draw, but doing ANY of this at collate
time would mean json.loads + re-serialize + re-tokenize per item per batch
-- this project has already documented that per-pair/per-chunk work of that
kind is what turned an earlier run into a ~9.5h-of-12h bottleneck. So both
possible orderings are pre-tokenized ONCE here:
    variant_base/     rules 2-4 only          (drawn 95% of the time)
    variant_shuffled/ rule 1 + rules 2-4      (drawn  5% of the time)
At train time the collate function rolls a 5% coin and indexes into one
array or the other -- a pointer choice, no compute. The shuffled variant is
one fixed permutation per item rather than a fresh one per draw; at a 5%
rate this is a deliberate, negligible approximation of the policy, traded
for keeping train-time cost at exactly zero.

Byte-compatibility with the existing FRIDA lineage is preserved everywhere
else: same "paraphrase: " prefix, same Name/Category/Attributes layout, same
max_length=256, same add_special_tokens=True. Char budgets are NOT the
CLAUDE.md 128/64/384 formula -- those were rescaled for this kernel's
256-token target, see the constants below.
(bos=1 ... eos=2 per item -- the FRIDA convention, unlike this project's
BERTA caches which store raw content and add specials at collate time).
CPU kernel (no GPU/TPU quota spent): pure tokenization, 711K items.
"""

import os
import sys
import gc
import ctypes
import json
import random
import subprocess
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import polars as pl
from transformers import AutoTokenizer

MODEL_NAME = "ai-forever/FRIDA"

ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

SEED = 42

PREFIX = "paraphrase: "
# Rescaled for ITEM_MAX_LENGTH=256 (the 128/64/384 CLAUDE.md formula was sized
# for MiniLM's 128-token regime and was silently under-covering real content
# here -- category got ~3x more char budget than its real max needs, while
# attributes got truncated below its real p90/p99 before the tokenizer ever
# saw it). Derived from real FRIDA chars/token ratios (see
# kaggle_kernels/frida-char-token-ratio): name p99=50tok*2.83=142c -> +10%
# margin; category real max=8tok*2.34=19c -> +50% margin; attributes gets the
# remainder up to a combined-text ceiling of 256tok*3.12c/tok=799c.
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 156, 28, 564
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 812
ITEM_MAX_LENGTH = 256

LONG_VALUE_CHARS = 200      # rule 2 threshold
LINK_MARKER = "https"       # rule 3, tested on the VALUE (0 keys contain it)
OEM_MARKER = "oem"          # rule 4, tested on KEY or VALUE -- see docstring

TOKENIZE_CHUNK_SIZE = 20_000

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_item_tokens_v2")

t_start = time.perf_counter()


def _log(stage, msg):
    print(f"[{time.perf_counter() - t_start:7.1f}s] {stage}: {msg}", flush=True)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return None


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                _log("DATA", f"hint {hint} didn't resolve -- found {filename} at {root}")
                return os.path.join(root, filename)
    return hint


def _demotion_rank(key, value):
    """Index of the LAST rule that matches -- see module docstring for why
    this single number reproduces the sequential 'move to the end' rules.
    0 = keep in place, 3 = pushed furthest back."""
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


def reorder_attributes(attr_str, shuffle, rng, stats):
    """Returns the re-serialized attributes JSON with the policy applied.
    Falls back to the untouched input on anything not cleanly parseable as a
    non-empty dict -- same defensive convention the rest of this project
    uses for malformed attribute blobs."""
    if not attr_str:
        return attr_str
    try:
        d = json.loads(attr_str)
    except Exception:
        stats["unparseable"] += 1
        return attr_str
    if not isinstance(d, dict) or not d:
        stats["not_a_dict_or_empty"] += 1
        return attr_str

    items = list(d.items())
    if shuffle:
        rng.shuffle(items)

    ranked = []
    for k, v in items:
        r = _demotion_rank(k, v)
        if r:
            stats[f"rank{r}"] += 1
        ranked.append((r, k, v))

    # Stable sort by rank -- preserves relative order inside each group,
    # which is what sequential "move to the end" actually does.
    ranked.sort(key=lambda t: t[0])
    stats["reordered_items"] += 1
    return json.dumps({k: v for _r, k, v in ranked}, ensure_ascii=False)


def build_texts(items_pl, shuffle, rng, stats):
    """Builds the final per-item text strings. Attribute reordering happens
    on the FULL attributes string and the MAX_ATTR_CHARS slice is applied
    AFTER -- slicing first would hand json.loads a truncated (invalid)
    blob, and the whole point of the policy is to decide what survives that
    slice."""
    ids = items_pl.get_column("id").to_numpy().astype(np.int64)
    names = items_pl.get_column("name").to_list()
    cats = items_pl.get_column("category").to_list()
    attrs = items_pl.get_column("attributes").to_list()

    texts = []
    for name, cat, attr in zip(names, cats, attrs):
        name = (name or "")[:MAX_NAME_CHARS]
        cat = (cat or "")[:MAX_CATEGORY_CHARS]
        attr_ordered = reorder_attributes(attr or "", shuffle, rng, stats)[:MAX_ATTR_CHARS]
        texts.append(f"Name: {name} Category: {cat} Attributes: {attr_ordered}"[:MAX_ITEM_CHARS])
    return ids, texts


def tokenize_to_memmap(ids, texts, tokenizer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = len(ids)

    import shutil
    need = n * (8 + ITEM_MAX_LENGTH * 4 + ITEM_MAX_LENGTH)
    free = shutil.disk_usage(out_dir).free
    _log("DISK", f"{out_dir}: free={free / 1e9:.1f}GB need(+15%)={need * 1.15 / 1e9:.1f}GB")
    if free < need * 1.15:
        raise RuntimeError(f"insufficient disk for {out_dir}")

    ids_mm = np.memmap(os.path.join(out_dir, "ids.bin"), dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(os.path.join(out_dir, "input_ids.bin"), dtype=np.int32, mode="w+",
                              shape=(n, ITEM_MAX_LENGTH))
    attn_mm = np.memmap(os.path.join(out_dir, "attention_mask.bin"), dtype=np.int8, mode="w+",
                         shape=(n, ITEM_MAX_LENGTH))
    ids_mm[:] = ids

    written = 0
    next_log = 0
    while written < n:
        cur = min(TOKENIZE_CHUNK_SIZE, n - written)
        chunk = [PREFIX + t for t in texts[written:written + cur]]
        # add_special_tokens defaults to True -- FRIDA cache convention
        # (bos=1 ... eos=2 baked into each item; the training collate
        # concatenates items directly and never re-adds specials).
        enc = tokenizer(chunk, padding="max_length", truncation=True,
                         max_length=ITEM_MAX_LENGTH, return_tensors="np")
        input_ids_mm[written:written + cur] = enc["input_ids"].astype(np.int32)
        attn_mm[written:written + cur] = enc["attention_mask"].astype(np.int8)
        written += cur
        if written >= next_log:
            rss = _rss_gb()
            _log("TOKENIZE", f"{out_dir}: {written:,}/{n:,}" + (f" rss={rss:.1f}GB" if rss else ""))
            next_log += 200_000
        gc.collect()

    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()

    lengths = np.asarray(attn_mm).sum(axis=1)
    _log("TOKENIZE", f"{out_dir}: length mean={lengths.mean():.1f} p50={np.percentile(lengths, 50):.0f} "
                      f"p90={np.percentile(lengths, 90):.0f} max={lengths.max()} "
                      f"at_cap={100 * (lengths == ITEM_MAX_LENGTH).mean():.2f}%")

    # bos/eos sanity check -- the training kernel asserts exactly this, so
    # catching a violation here is much cheaper than discovering it on GPU.
    rng = np.random.RandomState(SEED)
    for r in rng.choice(n, size=min(50, n), replace=False):
        L = int(lengths[r])
        assert input_ids_mm[r, 0] == 1, f"row {r}: expected bos=1, got {input_ids_mm[r, 0]}"
        assert input_ids_mm[r, L - 1] == 2, f"row {r}: expected eos=2 at {L - 1}, got {input_ids_mm[r, L - 1]}"
    _log("TOKENIZE", f"{out_dir}: bos/eos sanity check passed on 50 random rows")

    del ids_mm, input_ids_mm, attn_mm
    gc.collect()
    _malloc_trim()
    return n, lengths


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    _log("CONFIG", f"model={MODEL_NAME} items={items_path} item_max_length={ITEM_MAX_LENGTH} "
                   f"prefix={PREFIX!r} long_value_chars={LONG_VALUE_CHARS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _log("SETUP", f"tokenizer={type(tokenizer).__name__} vocab={tokenizer.vocab_size}")

    items = pl.read_parquet(items_path, columns=["id", "name", "category", "attributes"])
    _log("DATA", f"items loaded: {items.height:,}")

    meta_variants = {}
    for variant, shuffle in (("variant_base", False), ("variant_shuffled", True)):
        t0 = time.perf_counter()
        rng = random.Random(SEED)
        stats = {"unparseable": 0, "not_a_dict_or_empty": 0, "reordered_items": 0,
                 "rank1": 0, "rank2": 0, "rank3": 0}
        ids, texts = build_texts(items, shuffle, rng, stats)
        _log("POLICY", f"{variant}: reordered={stats['reordered_items']:,} "
                        f"unparseable={stats['unparseable']:,} not_dict/empty={stats['not_a_dict_or_empty']:,} | "
                        f"demoted attrs: long={stats['rank1']:,} https={stats['rank2']:,} oem={stats['rank3']:,}")

        n, lengths = tokenize_to_memmap(ids, texts, tokenizer, os.path.join(OUTPUT_DIR, variant))
        meta_variants[variant] = {
            "n_items": n, "shuffled": shuffle, "policy_stats": stats,
            "length_mean": float(lengths.mean()), "length_p90": float(np.percentile(lengths, 90)),
            "at_cap_frac": float((lengths == ITEM_MAX_LENGTH).mean()),
        }
        del ids, texts
        gc.collect()
        _malloc_trim()
        _log("DONE", f"{variant} finished in {time.perf_counter() - t0:.0f}s")

    n_items = meta_variants["variant_base"]["n_items"]
    with open(os.path.join(OUTPUT_DIR, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_items": n_items, "max_length": ITEM_MAX_LENGTH, "model": MODEL_NAME,
            "prefix": PREFIX, "add_special_tokens": True,
            "text_formula": "Name/Category/Attributes, CLAUDE.md char budgets (128/64/384/640)",
            "source": "items_human.parquet, full scope",
            "attribute_policy": {
                "shuffle_probability_at_train_time": 0.05,
                "long_value_chars": LONG_VALUE_CHARS,
                "link_marker": LINK_MARKER,
                "oem_marker": OEM_MARKER,
                "oem_tested_on": "key OR value (case-insensitive) -- measured: 'oem' is in the KEY "
                                  "for 17.35% of items but in the VALUE for only 0.06%",
                "rule_order": "stable sort by last-matching-rule rank (0 normal, 1 long, 2 https, 3 oem)",
            },
            "variants": meta_variants,
            "variant_dirs": ["variant_base", "variant_shuffled"],
        }, f, ensure_ascii=False, indent=2)

    _log("DONE", f"both variants written -> {OUTPUT_DIR}, total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
