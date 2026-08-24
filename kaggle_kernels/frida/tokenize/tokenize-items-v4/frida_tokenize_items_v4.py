"""FRIDA v4 item cache: one CPU pass over items.parquet, two outputs.

Output:
  /kaggle/working/frida_item_tokens_v4/human/
  /kaggle/working/frida_item_tokens_v4/llm/

Human and LLM ids are disjoint, but their union is the full items.parquet.
Every full item is therefore formatted and tokenized exactly once, then routed
to human/ when its id occurs in items_human.parquet, otherwise to llm/.
"""
import os

# Must be set before importing transformers/tokenizers.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import gc
import json
import shutil
import subprocess
import sys
import time

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.15.0", "tokenizers==0.22.2",
])

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer, __version__ as transformers_version
import tokenizers

MODEL_NAME = "ai-forever/FRIDA"
MAX_NAME_CHARS = 160
MAX_ATTR_CHARS = 640
ITEM_MAX_LENGTH = 256
BATCH_ROWS = 20_000

EXPECTED_HUMAN = 711_304
EXPECTED_LLM = 12_384_610
EXPECTED_FULL = EXPECTED_HUMAN + EXPECTED_LLM

LONG_VALUE_CHARS = 200
LINK_MARKER = "https"
OEM_MARKER = "oem"

INPUT_ROOT = "/kaggle/input"
WORKING_ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_ROOT = os.path.join(WORKING_ROOT, "frida_item_tokens_v4")
HUMAN_DIR = os.path.join(OUTPUT_ROOT, "human")
LLM_DIR = os.path.join(OUTPUT_ROOT, "llm")

t0 = time.perf_counter()


def log(stage, message):
    print(f"[{time.perf_counter() - t0:8.1f}s] {stage}: {message}", flush=True)


def find_input(filename):
    preferred = os.path.join(
        INPUT_ROOT, "datasets", "maxamsecret", "ozon-ecup-2026-matching", filename,
    )
    if os.path.isfile(preferred):
        return preferred
    for root, _dirs, files in os.walk(INPUT_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(filename)


def clean_field(value):
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text.replace(" = ", "=")


def demotion_rank(key, value):
    value = value if isinstance(value, str) else str(value)
    rank = int(len(value) > LONG_VALUE_CHARS)
    value_lower = value.lower()
    if LINK_MARKER in value_lower:
        rank = 2
    if OEM_MARKER in value_lower or OEM_MARKER in str(key).lower():
        rank = 3
    return rank


def format_item(name, attributes):
    name = clean_field(name)[:MAX_NAME_CHARS]
    try:
        attrs = json.loads(attributes) if attributes else {}
    except Exception:
        attrs = {}
    if not isinstance(attrs, dict):
        attrs = {}

    ranked = sorted(
        ((demotion_rank(key, value), key, value) for key, value in attrs.items()),
        key=lambda row: row[0],
    )
    lines = []
    for _rank, key, value in ranked:
        key = clean_field(key)
        value = clean_field(value)
        if key or value:
            lines.append(f"{key} = {value}")
    attrs_text = "\n".join(lines)[:MAX_ATTR_CHARS]
    return f"{name}\n{attrs_text}" if attrs_text else name


def create_cache(path, n_rows):
    os.makedirs(path, exist_ok=True)
    return {
        "ids": np.memmap(os.path.join(path, "ids.bin"), np.int64, "w+", shape=(n_rows,)),
        "input_ids": np.memmap(
            os.path.join(path, "input_ids.bin"), np.int32, "w+",
            shape=(n_rows, ITEM_MAX_LENGTH),
        ),
        "attention_mask": np.memmap(
            os.path.join(path, "attention_mask.bin"), np.int8, "w+",
            shape=(n_rows, ITEM_MAX_LENGTH),
        ),
        "length_hist": np.zeros(ITEM_MAX_LENGTH + 1, dtype=np.int64),
        "written": 0,
    }


def write_rows(cache, ids, input_ids, attention_mask):
    n = len(ids)
    start = cache["written"]
    end = start + n
    cache["ids"][start:end] = ids
    cache["input_ids"][start:end] = input_ids
    cache["attention_mask"][start:end] = attention_mask
    lengths = attention_mask.sum(axis=1).astype(np.int64)
    cache["length_hist"] += np.bincount(lengths, minlength=ITEM_MAX_LENGTH + 1)
    cache["written"] = end


def percentile_from_hist(hist, q):
    target = int(np.ceil(hist.sum() * q / 100))
    return int(np.searchsorted(np.cumsum(hist), target, side="left"))


def finalize_cache(cache, path, scope):
    for key in ("ids", "input_ids", "attention_mask"):
        cache[key].flush()
    hist = cache["length_hist"]
    n = int(hist.sum())
    meta = {
        "format_version": 4,
        "scope": scope,
        "n_items": n,
        "max_length": ITEM_MAX_LENGTH,
        "model": MODEL_NAME,
        "prefix": "",
        "add_special_tokens": True,
        "text_formula": "{clean name}\\n{clean key} = {clean value}\\n...",
        "normalization": "collapse whitespace; strip; internal ' = ' -> '='",
        "category_in_text": False,
        "max_name_chars": MAX_NAME_CHARS,
        "max_attr_chars": MAX_ATTR_CHARS,
        "length_mean": float(np.dot(np.arange(len(hist)), hist) / max(n, 1)),
        "length_p50": percentile_from_hist(hist, 50),
        "length_p99": percentile_from_hist(hist, 99),
        "at_cap_frac": float(hist[ITEM_MAX_LENGTH] / max(n, 1)),
    }
    with open(os.path.join(path, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("DONE", f"{scope}: {n:,} rows, mean={meta['length_mean']:.2f}, "
                f"p99={meta['length_p99']}, at_cap={100 * meta['at_cap_frac']:.2f}%")
    return meta


def main():
    items_path = find_input("items.parquet")
    human_path = find_input("items_human.parquet")
    full_rows = pq.ParquetFile(items_path).metadata.num_rows
    human_table = pq.read_table(human_path, columns=["id"])
    human_ids_array = human_table.column("id").to_numpy(zero_copy_only=False).astype(np.int64)
    human_ids = set(map(int, human_ids_array))
    del human_table, human_ids_array

    assert full_rows == EXPECTED_FULL, f"items.parquet rows={full_rows:,}, expected={EXPECTED_FULL:,}"
    assert len(human_ids) == EXPECTED_HUMAN, \
        f"human ids={len(human_ids):,}, expected={EXPECTED_HUMAN:,}"

    bytes_per_row = 8 + ITEM_MAX_LENGTH * 5
    required = int((EXPECTED_FULL * bytes_per_row) * 1.15)
    free = shutil.disk_usage(WORKING_ROOT).free
    log("PREFLIGHT", f"output required(+15%)={required / 1e9:.2f}GB, free={free / 1e9:.2f}GB")
    if free < required:
        raise RuntimeError(f"insufficient disk: need {required / 1e9:.2f}GB, have {free / 1e9:.2f}GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sample = format_item(" Название  товара\n", '{"цвет товара":"черный","бренд":"Samsung"}')
    assert sample == "Название товара\nцвет товара = черный\nбренд = Samsung"
    log("PREFLIGHT", f"transformers={transformers_version}, tokenizers={tokenizers.__version__}, "
                     f"tokenizer={type(tokenizer).__name__}, sample={sample!r}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_path = os.path.join(OUTPUT_ROOT, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"status": "running", "full_rows": full_rows}, f)

    caches = {
        "human": create_cache(HUMAN_DIR, EXPECTED_HUMAN),
        "llm": create_cache(LLM_DIR, EXPECTED_LLM),
    }
    seen_human = set()
    processed = 0
    parquet = pq.ParquetFile(items_path)
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=["id", "name", "attributes"]):
        ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        names = batch.column(1).to_pylist()
        attributes = batch.column(2).to_pylist()
        texts = [format_item(name, attr) for name, attr in zip(names, attributes)]
        encoded = tokenizer(
            texts, add_special_tokens=True, padding="max_length", truncation=True,
            max_length=ITEM_MAX_LENGTH, return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int32)
        attention_mask = encoded["attention_mask"].astype(np.int8)
        human_mask = np.fromiter((int(item_id) in human_ids for item_id in ids), bool, len(ids))
        if human_mask.any():
            seen_human.update(map(int, ids[human_mask]))
            write_rows(caches["human"], ids[human_mask], input_ids[human_mask], attention_mask[human_mask])
        if (~human_mask).any():
            write_rows(caches["llm"], ids[~human_mask], input_ids[~human_mask], attention_mask[~human_mask])

        processed += len(ids)
        if processed % 500_000 < len(ids):
            log("TOKENIZE", f"{processed:,}/{full_rows:,}; human={caches['human']['written']:,}; "
                            f"llm={caches['llm']['written']:,}")
        del batch, names, attributes, texts, encoded, input_ids, attention_mask, human_mask
        gc.collect()

    assert caches["human"]["written"] == EXPECTED_HUMAN
    assert caches["llm"]["written"] == EXPECTED_LLM
    assert seen_human == human_ids, f"missing human ids: {len(human_ids - seen_human):,}"

    human_meta = finalize_cache(caches["human"], HUMAN_DIR, "human")
    llm_meta = finalize_cache(caches["llm"], LLM_DIR, "llm")
    summary = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - t0,
        "single_full_items_pass": True,
        "human": human_meta,
        "llm": llm_meta,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("DONE", f"all outputs complete in {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
