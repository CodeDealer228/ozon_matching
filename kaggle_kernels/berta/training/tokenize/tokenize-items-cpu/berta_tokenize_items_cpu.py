"""
BERTA item-level token cache, CPU kernel, human + LLM scope in ONE run.

Sibling of kaggle_kernels/frida-tokenize-items-cpu -- same rationale for
running both scopes in one kernel (human/LLM item ids are NOT a subset
pair, 0 shared ids out of 711,304 vs 12,384,610 per
reports/llm_label_overlap_summary.json; each scope reads its own natural
source file, so nothing is tokenized twice), same v2 attribute-cleanup
policy (deterministic demotion only, no shuffle variant), same char budget
derivation. See that kernel's docstring for the shared rationale in full;
this one only calls out what's BERTA-specific.

BERTA-specific:
- Tokenizer: sergeyzh/BERTA's own BertTokenizer (WordPiece, vocab 55,083)
  -- confirmed via direct measurement to be LESS efficient than FRIDA's
  byte-BPE on attributes (2.500 vs 2.994 chars/token), which is why its
  MAX_ATTR_CHARS differs (442, not FRIDA's 529) despite the same
  ITEM_MAX_LENGTH=256 and the same NAME/CATEGORY targets (140/32 -- those
  are properties of the field-length distribution, not the tokenizer).
- add_special_tokens=False: BERTA is used as a real BERT-style cross-
  encoder ([CLS] item1 [SEP] item2 [SEP], jointly attended, token_type_ids
  distinguishing the two segments) -- CLS/SEP get inserted at PAIR-BUILD
  time (see kaggle_kernels/berta-lora-pair-cache's allocate_pair_budget/
  collate logic), not baked into each item's own cache. Storing specials
  here would leave stray CLS/SEP tokens in the middle of an assembled
  pair. This matches the existing item-cache convention already
  established by berta-tokenize-llm-256 and berta-lora-sft-*.
- ITEM_MAX_LENGTH=256, matching berta-tokenize-llm-256 (the cache behind
  kaggle_kernels/berta-llm-continue-train, which continues the
  berta_tpu_lora_submit lineage -- the real test-passing BERTA
  submission, PAIR_MAX_LENGTH=256). NOT the older 128-token item cache
  used by berta-lora-pair-cache for a different (already-trained,
  val=0.7111) checkpoint's embedding extraction -- that lineage isn't
  being retokenized here.
- category is also cached alongside id/text (category_vocab.json +
  categories.npy), matching berta-tokenize-llm-256's own convention --
  kept here for parity even though nothing in this kernel itself
  consumes it; a downstream training kernel may want per-category
  weighting or stratification without a second items.parquet pass.

Plain CPU kernel (enable_gpu=False, enable_tpu=False), sized for Kaggle's
standard 4-core/30GB offering (per direct instruction) -- NOT the TPU-host
sizing (100/200GB RAM targets) berta-tokenize-llm-256 used for the same
LLM scope. Same acknowledged risk as the FRIDA sibling: the TPU run's own
docstring credits the larger RAM ceiling for the whole pipeline (join +
tokenize), not just tokenization, so 30GB may not be enough for the
12.38M-item LLM-scope semi-join + chunked tokenize -- RAM_TARGET_GB/
RAM_HARD_GB are deliberately conservative (16/24) and every phase logs
real RSS so a failure is visible immediately rather than a silent OOM-kill.

Publishes TWO separate Kaggle datasets under the rays0n account (BERTA's
own established account, kept separate from FRIDA's maxamsecret quota):
  rays0n/berta-item-tokens-v2-human
  rays0n/berta-item-tokens-v2-llm
"""
import os
import sys
import gc
import ctypes
import json
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "transformers==5.15.0", "tokenizers==0.22.2"])
# >=2.2.0, not a bare "kaggle" -- see frida-tokenize-items-cpu's identical
# fix for the real 2.0.2-vs-2.2.4 CLI bug this addresses (same error, same
# root cause, hit on this kernel's own real run too).
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle>=2.2.0"])

import numpy as np
import pandas as pd
import polars as pl
from transformers import AutoTokenizer

BERTA_MODEL_NAME = "sergeyzh/BERTA"
BERTA_PREFIX = "paraphrase: "

MATCHES_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
ITEMS_FULL_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

# Char budget -- see reports/experiment_log.md "Char-budget rule v2".
# NAME/CATEGORY targets are shared with FRIDA (distribution properties,
# not tokenizer-specific); ATTR differs because BERTA's WordPiece tokenizer
# is less efficient on attribute text (2.500 vs FRIDA's 2.994 chars/token).
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 442
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 30  # 644
ITEM_MAX_LENGTH = 256

LONG_VALUE_CHARS = 200
LINK_MARKER = "https"
OEM_MARKER = "oem"

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 100_000
RAM_TARGET_GB = 16.0
RAM_HARD_GB = 24.0

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_item_tokens_cpu")

SCOPES = {
    "human": {
        "dataset_id": "rays0n/berta-item-tokens-v2-human",
        "dataset_title": "BERTA item tokens v2 -- human scope",
    },
    "llm": {
        "dataset_id": "rays0n/berta-item-tokens-v2-llm",
        "dataset_title": "BERTA item tokens v2 -- LLM scope",
    },
}

t_start = time.perf_counter()


def _log(stage, msg):
    print(f"[{time.perf_counter() - t_start:7.1f}s] {stage}: {msg}", flush=True)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _current_rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return None


def _log_real_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        _log("RAM", f"host RAM: total={vm.total / 1e9:.1f}GB available={vm.available / 1e9:.1f}GB")
    except ImportError:
        pass


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                _log("DATA", f"hint {hint} didn't resolve -- found {filename} at {root}")
                return os.path.join(root, filename)
    return hint


def _require_disk_space(path, required_bytes, label, margin=1.15):
    os.makedirs(path, exist_ok=True)
    import shutil
    free = shutil.disk_usage(path).free
    need = int(required_bytes * margin)
    _log("DISK", f"{label}: free={free / 1e9:.1f}GB required(+15%)={need / 1e9:.1f}GB")
    if free < need:
        raise RuntimeError(f"insufficient disk for {label}: need {need / 1e9:.1f}GB, have {free / 1e9:.1f}GB")


# =====================================================================
# v2 attribute-cleanup policy -- identical rules to the FRIDA sibling
# kernel (shared corpus, shared rationale); duplicated rather than
# imported since Kaggle kernels are single self-contained scripts.
# =====================================================================

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


def reorder_attributes(attr_str, stats):
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

    ranked = []
    for k, v in d.items():
        r = _demotion_rank(k, v)
        if r:
            stats[f"rank{r}"] += 1
        ranked.append((r, k, v))
    ranked.sort(key=lambda t: t[0])
    stats["reordered_items"] += 1
    return json.dumps({k: v for _r, k, v in ranked}, ensure_ascii=False)


def build_item_texts(items_lazy, needed_ids_lazy=None):
    lf = items_lazy
    if needed_ids_lazy is not None:
        lf = lf.join(needed_ids_lazy, on="id", how="semi")
    lf = lf.select(["id", "name", "category", "attributes"]).with_columns([
        pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS),
        pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_CATEGORY_CHARS),
        pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False),
    ])
    return lf.collect(engine="streaming")


def tokenize_scope(items_pl, tokenizer, out_dir, extra_meta):
    os.makedirs(out_dir, exist_ok=True)
    n = items_pl.height

    all_ids = items_pl.get_column("id").to_numpy().astype(np.int64)
    all_names = items_pl.get_column("name").to_list()
    all_cats_raw = items_pl.get_column("category").fill_null("UNKNOWN").to_list()
    all_attrs = items_pl.get_column("attributes").to_list()
    del items_pl
    gc.collect()

    bytes_per_row = 8 + ITEM_MAX_LENGTH * 4 + ITEM_MAX_LENGTH * 1
    _require_disk_space(out_dir, n * bytes_per_row, f"{out_dir} token cache")

    ids_mm = np.memmap(os.path.join(out_dir, "ids.bin"), dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(os.path.join(out_dir, "input_ids.bin"), dtype=np.int32, mode="w+",
                              shape=(n, ITEM_MAX_LENGTH))
    attn_mm = np.memmap(os.path.join(out_dir, "attention_mask.bin"), dtype=np.int8, mode="w+",
                         shape=(n, ITEM_MAX_LENGTH))
    ids_mm[:] = all_ids

    category_vocab = sorted(set(all_cats_raw) | {"UNKNOWN"})
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    categories_arr = np.array([category_to_id[c] for c in all_cats_raw], dtype=np.int32)
    np.save(os.path.join(out_dir, "categories.npy"), categories_arr)
    with open(os.path.join(out_dir, "category_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(category_vocab, f, ensure_ascii=False, indent=2)

    policy_stats = {"unparseable": 0, "not_a_dict_or_empty": 0, "reordered_items": 0,
                     "rank1": 0, "rank2": 0, "rank3": 0}

    chunk_size = TOKENIZE_CHUNK_SIZE_START
    written = 0
    next_log_at = 0
    while written < n:
        cur = min(chunk_size, n - written)
        end = written + cur

        chunk_texts = []
        for name, cat, attr in zip(all_names[written:end], all_cats_raw[written:end], all_attrs[written:end]):
            attr_ordered = reorder_attributes(attr or "", policy_stats)[:MAX_ATTR_CHARS]
            text = f"Name: {name or ''} Category: {cat or ''} Attributes: {attr_ordered}"[:MAX_ITEM_CHARS]
            chunk_texts.append(BERTA_PREFIX + text)

        enc = tokenizer(
            chunk_texts, add_special_tokens=False, padding="max_length",
            truncation=True, max_length=ITEM_MAX_LENGTH, return_tensors="np",
        )
        input_ids_mm[written:end] = enc["input_ids"].astype(np.int32)
        attn_mm[written:end] = enc["attention_mask"].astype(np.int8)

        written = end
        gc.collect()
        _malloc_trim()
        rss_gb = _current_rss_gb()
        if rss_gb is not None:
            if rss_gb < RAM_TARGET_GB:
                chunk_size = min(TOKENIZE_CHUNK_SIZE_MAX, int(chunk_size * 1.3))
            elif rss_gb > RAM_HARD_GB:
                chunk_size = max(TOKENIZE_CHUNK_SIZE_MIN, int(chunk_size * 0.5))

        if written >= next_log_at:
            rss_part = f"rss={rss_gb:.2f}GB " if rss_gb is not None else ""
            _log("TOKENIZE", f"{out_dir}: {written:,}/{n:,} ({rss_part}next_chunk={chunk_size:,})")
            next_log_at += 500_000

    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()

    lengths = np.asarray(attn_mm).sum(axis=1)
    _log("TOKENIZE", f"{out_dir}: length mean={lengths.mean():.1f} p50={np.percentile(lengths, 50):.0f} "
                      f"p99={np.percentile(lengths, 99):.0f} max={lengths.max()} "
                      f"at_cap={100 * (lengths == ITEM_MAX_LENGTH).mean():.2f}%")
    _log("POLICY", f"{out_dir}: reordered={policy_stats['reordered_items']:,} "
                    f"unparseable={policy_stats['unparseable']:,} "
                    f"not_dict/empty={policy_stats['not_a_dict_or_empty']:,} | "
                    f"demoted: long={policy_stats['rank1']:,} https={policy_stats['rank2']:,} "
                    f"oem={policy_stats['rank3']:,}")

    del ids_mm, input_ids_mm, attn_mm, all_ids, all_names, all_cats_raw, all_attrs
    gc.collect()
    _malloc_trim()

    meta = {
        "n_items": n, "max_length": ITEM_MAX_LENGTH, "model": BERTA_MODEL_NAME,
        "prefix": BERTA_PREFIX, "add_special_tokens": False,
        "text_formula": f"Name/Category/Attributes, char budget NAME={MAX_NAME_CHARS}/"
                         f"CATEGORY={MAX_CATEGORY_CHARS}/ATTR={MAX_ATTR_CHARS} "
                         "(see reports/experiment_log.md 'Char-budget rule v2')",
        "category_vocab_file": "category_vocab.json", "categories_file": "categories.npy",
        "attribute_policy": {
            "long_value_chars": LONG_VALUE_CHARS, "link_marker": LINK_MARKER, "oem_marker": OEM_MARKER,
            "oem_tested_on": "key OR value (case-insensitive)",
            "shuffle_variant": "NOT included -- deterministic demotion rules only, see module docstring",
            "stats": policy_stats,
        },
        "length_mean": float(lengths.mean()), "length_p99": float(np.percentile(lengths, 99)),
        "at_cap_frac": float((lengths == ITEM_MAX_LENGTH).mean()),
    }
    meta.update(extra_meta)
    with open(os.path.join(out_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _log("DONE", f"{out_dir}: {n:,} items tokenized")


def publish_dataset(out_dir, dataset_id, dataset_title):
    # Fail here, at second zero, not after hours of real tokenization work --
    # the FRIDA sibling kernel hit "The dataset title must be between 6 and
    # 50 characters" AFTER both scopes (711K + 12.38M items, ~3.65h) had
    # already finished, at the very last line. Same title-length bug was
    # present here too (copy-pasted); checked locally before spending it again.
    assert 6 <= len(dataset_title) <= 50, \
        f"dataset_title must be 6-50 chars, got {len(dataset_title)}: {dataset_title!r}"
    with open(os.path.join(out_dir, "dataset-metadata.json"), "w") as f:
        json.dump({"id": dataset_id, "title": dataset_title, "licenses": [{"name": "CC0-1.0"}]}, f)
    # -u/--public: real bug found the hard way -- -t is --keep-tabular (don't
    # convert .bin files to CSV), NOT "public". Without -u, `datasets create`
    # defaults to PRIVATE. Missing this flag is a real, wider pattern across
    # this project's older publish_dataset-style code (frida-tokenize-items,
    # frida-tokenize-items-human/-llm, several berta-*-train kernels all
    # share the same -t-without-u gap) -- not fixed everywhere retroactively
    # (changing an EXISTING dataset's visibility is a separate, deliberate
    # action), but this kernel gets it right going forward.
    _log("PUBLISH", f"creating dataset {dataset_id} (public) ...")
    subprocess.check_call(["kaggle", "datasets", "create", "-p", out_dir, "-t", "-u"])
    _log("PUBLISH", f"published at https://www.kaggle.com/datasets/{dataset_id}")


def main():
    for scope_name, scope in SCOPES.items():
        title = scope["dataset_title"]
        assert 6 <= len(title) <= 50, \
            f"SCOPES[{scope_name!r}]['dataset_title'] must be 6-50 chars, got {len(title)}: {title!r}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _log_real_ram()

    tokenizer = AutoTokenizer.from_pretrained(BERTA_MODEL_NAME)
    _log("SETUP", f"tokenizer={type(tokenizer).__name__} vocab={tokenizer.vocab_size} "
                  f"prefix={BERTA_PREFIX!r} item_max_length={ITEM_MAX_LENGTH}")

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    items_full_path = _find_dataset_file(ITEMS_FULL_PATH_HINT, "items.parquet")
    matches_llm_path = _find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet")

    # Phase 1: human scope (items_human.parquet is already pre-filtered).
    human_items = build_item_texts(pl.scan_parquet(items_human_path))
    _log("DATA", f"human-scope items: {human_items.height:,}")
    tokenize_scope(human_items, tokenizer, os.path.join(OUTPUT_DIR, "human_scope"), {
        "source": "items_human.parquet, full scope",
    })
    del human_items
    gc.collect()
    _malloc_trim()
    _log_real_ram()

    # Phase 2: LLM scope (semi-join against the full catalog).
    llm_pairs = (
        pl.scan_parquet(matches_llm_path)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .select(["id1", "id2"]).collect(engine="streaming")
    )
    needed_ids = pd.unique(np.concatenate([
        llm_pairs.get_column("id1").to_numpy(), llm_pairs.get_column("id2").to_numpy()
    ]))
    del llm_pairs
    gc.collect()
    _log("DATA", f"LLM-scope unique item ids: {len(needed_ids):,}")

    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    llm_items = build_item_texts(pl.scan_parquet(items_full_path), needed_ids_lazy=needed_ids_lazy)
    _log("DATA", f"LLM-scope items resolved: {llm_items.height:,} / {len(needed_ids):,} needed")
    del needed_ids, needed_ids_lazy
    gc.collect()
    _malloc_trim()
    _log_real_ram()

    tokenize_scope(llm_items, tokenizer, os.path.join(OUTPUT_DIR, "llm_scope"), {
        "source": "items.parquet semi-joined to every id referenced by matches_llm.parquet (no sampling)",
    })
    del llm_items
    gc.collect()
    _malloc_trim()

    publish_dataset(os.path.join(OUTPUT_DIR, "human_scope"), SCOPES["human"]["dataset_id"],
                     SCOPES["human"]["dataset_title"])
    publish_dataset(os.path.join(OUTPUT_DIR, "llm_scope"), SCOPES["llm"]["dataset_id"],
                     SCOPES["llm"]["dataset_title"])

    _log("DONE", f"all scopes tokenized + published, total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
