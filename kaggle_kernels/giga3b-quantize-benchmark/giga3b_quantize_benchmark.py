"""
Feasibility probe for ai-sage/Giga-Embeddings-instruct (3.45B decoder bi-encoder)
as a replacement for the current LoRA cross-encoder.

Answers, in one run, the four questions that gate the whole 3B direction:
  1. CORRECTNESS  - does `trust_remote_code + sentence-transformers` load and
     encode this model at all, in full precision and (critically) int8?
  2. QUALITY      - raw (un-fine-tuned) cosine macro PR-AUC per category on a
     stratified subsample of our canonical val split. If a raw 3B bi-encoder is
     already near / above the 0.5282 LoRA cross-encoder, fine-tuning it is worth
     it; if it's ~0.15, it isn't.
  3. SPEED        - encode throughput (items/s, tokens/s) + peak VRAM, projected
     to the ~500-680K unique items of the hidden private phase, so we know if
     it can beat the 13-min private limit on the stand's H100.
  4. PACKAGING    - save the int8-quantized weights (~3.5GB, fits the 5GB zip)
     and report their on-disk size, so the submission can ship them.

Grounded in the model card / config.json (checked 2026-08-13):
  - 3.45B params stored as F32 -> ~13.8GB download; hidden 2048; MIT license.
  - ST pipeline is Transformer -> Pooling(mean) -> Normalize (modules.json),
    max_seq_length 4096. Card pins transformers==4.51.0, sentence-transformers
    ==5.1.1 -- the custom `trust_remote_code` code is written against those, so
    they are pinned here rather than trusting whatever the Kaggle image ships.
  - flash-attn is NOT a hard dependency: modeling_gigarembed.py resolves the
    attention backend through HF's ALL_ATTENTION_FUNCTIONS with an eager
    fallback, so sdpa works on pre-Ampere GPUs (flash-attn needs sm_80+).

GPU here is whatever Kaggle grants (T4 = sm_75, P100 = sm_60). Neither has
bf16 tensor cores, so the full-precision pass runs fp16 and only switches to
bf16 when the device actually supports it. Correctness / quality / int8 save
are GPU-agnostic; throughput is additionally scaled to an H100 estimate by
peak-FLOPS ratio (clearly labelled as indicative).

VAL_PAIRS_LIMIT keeps this a *probe*: the full 73,131-pair val split would mean
~146K item encodes x2 passes (~4h of a scarce weekly GPU quota) to answer a
question a stratified subsample answers just as well.
"""

import os
import sys
import gc
import json
import time
import shutil
import subprocess

import numpy as np
import polars as pl

MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
SEED = 42

DATA_DIR = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching"
# items_human.parquet (214MB) covers every id in matches.parquet, so the 4.1GB
# 13.4M-row items.parquet is never needed for a val-only probe.
ITEMS_HUMAN_PATH = os.path.join(DATA_DIR, "items_human.parquet")
ITEMS_FULL_PATH = os.path.join(DATA_DIR, "items.parquet")
MATCHES_PATH = os.path.join(DATA_DIR, "matches.parquet")

WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INT8_DIR = os.path.join(WORKING, "giga_int8")
RESULTS_PATH = os.path.join(WORKING, "giga3b_probe_results.json")

MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64

VAL_PAIRS_LIMIT = 8_000       # stratified (category x label) subsample of val
MAX_SEQ_LEN = 512             # texts cap at 606 chars (~250 tok); 4096 wastes memory
BATCH_START = 32              # probed downward on OOM before the real run
BATCH_MIN = 4
WARMUP_TEXTS = 128            # cheap load+encode correctness check / batch probe
PRIVATE_ITEMS_ESTIMATE = 680_000   # worst-case unique items in private phase

# GPU peak TFLOPS at the dtype each device actually runs (dense, no sparsity).
GPU_PEAK_TFLOPS = {"Tesla T4": 65.0, "Tesla P100": 19.0, "A100": 312.0,
                   "H100": 990.0, "L4": 121.0}
H100_PEAK = 990.0

results = {}


def save_results():
    """Written after every stage so a mid-run failure still yields data."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def sh(*cmd):
    print("[SH]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


# The card's pinned versions: the custom modeling code is written against them.
sh(sys.executable, "-m", "pip", "install", "-q",
   "transformers==4.51.0", "sentence-transformers==5.1.1",
   "bitsandbytes", "accelerate")

import torch
import transformers
import sentence_transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
full_dtype = torch.bfloat16 if bf16_ok else torch.float16
DTYPE_TAG = "bf16" if bf16_ok else "fp16"

print(f"[INFO] torch={torch.__version__} transformers={transformers.__version__} "
      f"sentence_transformers={sentence_transformers.__version__}", flush=True)
print(f"[INFO] GPU={gpu_name} sm_{cap[0]}{cap[1]} bf16_supported={bf16_ok} "
      f"-> full-precision pass uses {DTYPE_TAG}", flush=True)

results["env"] = {"gpu": gpu_name, "sm": f"{cap[0]}.{cap[1]}", "bf16_supported": bf16_ok,
                  "full_dtype": DTYPE_TAG, "torch": torch.__version__,
                  "transformers": transformers.__version__,
                  "sentence_transformers": sentence_transformers.__version__}
save_results()


# ---------------------------------------------------------------- val split
def build_val():
    """Canonical split (CLAUDE.md): filter null ids, label = target >= 0.5, then
    a seed-42 stratified 80/20 -- in that order."""
    m = pl.read_parquet(MATCHES_PATH)
    m = (m.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
           .with_columns([
               pl.col("target").fill_null(0).cast(pl.Float32),
               (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
           ])
           .select(["id1", "id2", "label"]))
    md = m.to_pandas()
    labels = md["label"].to_numpy().astype(np.int64)
    idx = np.arange(len(labels))
    _, val_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=labels)
    return md.iloc[val_idx].reset_index(drop=True)


val = build_val()
print(f"[DATA] full canonical val pairs: {len(val):,}", flush=True)

needed = pl.concat([pl.Series(val["id1"]), pl.Series(val["id2"])]).unique().to_list()

items_path = ITEMS_HUMAN_PATH if os.path.exists(ITEMS_HUMAN_PATH) else ITEMS_FULL_PATH
print(f"[DATA] items source: {items_path}", flush=True)

# Semi-join down to the ids we actually need BEFORE any string/truncation op.
items = (pl.scan_parquet(items_path)
         .select(["id", "name", "category", "attributes"])
         .join(pl.DataFrame({"id": needed}).lazy(), on="id", how="inner")
         .collect(engine="streaming"))
items = items.with_columns(
    (pl.lit("Name: ")
     + pl.col("name").fill_null("").cast(pl.Utf8).str.slice(0, MAX_NAME_CHARS)
     + pl.lit(" Category: ")
     + pl.col("category").fill_null("").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS)
     + pl.lit(" Attributes: ")
     + pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS)
     ).str.slice(0, MAX_ITEM_CHARS).alias("text"))

id_to_text = dict(zip(items.get_column("id").to_list(), items.get_column("text").to_list()))
id_to_cat = dict(zip(items.get_column("id").to_list(), items.get_column("category").to_list()))
print(f"[DATA] item texts built: {len(id_to_text):,} (of {len(needed):,} needed)", flush=True)


# ------------------------------------------------- stratified val subsample
val["cat"] = [id_to_cat.get(i) or "UNK" for i in val["id1"]]
full_val_pairs = len(val)

if VAL_PAIRS_LIMIT is not None and len(val) > VAL_PAIRS_LIMIT:
    frac = VAL_PAIRS_LIMIT / len(val)
    parts = []
    for _key, g in val.groupby(["cat", "label"], sort=True):
        n = max(1, int(round(len(g) * frac)))
        parts.append(g.sample(n=min(n, len(g)), random_state=SEED))
    import pandas as pd
    val = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

pos_rate = float(val["label"].mean())
print(f"[DATA] probe val pairs: {len(val):,} | positives {pos_rate:.3%} | "
      f"categories {val['cat'].nunique()}", flush=True)

uniq_ids = sorted(set(val["id1"]).union(set(val["id2"])))
missing = [i for i in uniq_ids if i not in id_to_text]
uniq_ids = [i for i in uniq_ids if i in id_to_text]
uniq_texts = [id_to_text[i] for i in uniq_ids]
avg_chars = float(np.mean([len(t) for t in uniq_texts]))
print(f"[DATA] unique items to encode: {len(uniq_texts):,} | avg {avg_chars:.1f} chars | "
      f"ids missing from items: {len(missing)}", flush=True)

results["data"] = {"full_val_pairs": full_val_pairs, "probe_val_pairs": len(val),
                   "positive_rate": pos_rate, "n_categories": int(val["cat"].nunique()),
                   "unique_items": len(uniq_texts), "avg_chars": avg_chars,
                   "missing_ids": len(missing), "items_source": items_path}
save_results()


# ---------------------------------------------------------------- metric
def macro_pr_auc(y_true, y_prob, cats):
    """Macro-averaged per-category average_precision_score (CLAUDE.md metric):
    skip categories without both classes, plain mean over the rest."""
    y_true, y_prob, cats = np.asarray(y_true), np.asarray(y_prob), np.asarray(cats)
    aps = []
    for c in np.unique(cats):
        m = cats == c
        if len(np.unique(y_true[m])) == 2:
            aps.append(average_precision_score(y_true[m], y_prob[m]))
    overall = average_precision_score(y_true, y_prob)
    return (float(np.mean(aps)) if aps else float("nan")), float(overall), len(aps)


def score_val(id_to_vec, dim):
    zero = np.zeros(dim, dtype=np.float32)
    v1 = np.asarray([id_to_vec.get(i, zero) for i in val["id1"]], dtype=np.float32)
    v2 = np.asarray([id_to_vec.get(i, zero) for i in val["id2"]], dtype=np.float32)
    cos = np.sum(v1 * v2, axis=1)          # vectors are L2-normalized by ST
    return macro_pr_auc(val["label"].to_numpy(), cos, val["cat"].to_numpy())


# A signal-free encoder does NOT score 0 on this metric -- it scores about the
# positive rate. Measure that floor now so the verdict is readable without
# guessing what "bad" looks like on this particular subsample.
_rng = np.random.default_rng(SEED)
_rand = _rng.normal(size=(len(uniq_ids), 2048)).astype(np.float32)
_rand /= np.linalg.norm(_rand, axis=1, keepdims=True)
RANDOM_FLOOR, _, _ = score_val({i: _rand[k] for k, i in enumerate(uniq_ids)}, 2048)
del _rand
results["data"]["random_encoder_floor_macro_pr_auc"] = RANDOM_FLOOR
print(f"[DATA] random-encoder floor macro_pr_auc = {RANDOM_FLOOR:.4f} "
      f"(anything at this level means NO signal)", flush=True)
save_results()


# ---------------------------------------------------------------- model IO
def load_model(quantized):
    """Load through sentence-transformers, trying attention backends in order of
    preference. flash-attn only exists on sm_80+, so it is not even attempted on
    older cards; sdpa is the realistic path, eager the last resort."""
    backends = []
    if cap[0] >= 8:
        try:
            import flash_attn  # noqa: F401
            backends.append("flash_attention_2")
        except Exception:
            pass
    backends += ["sdpa", "eager"]

    last_err = None
    for attn in backends:
        model_kwargs = {"trust_remote_code": True, "attn_implementation": attn}
        if quantized:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = full_dtype
        try:
            print(f"[LOAD] quantized={quantized} attn={attn} ...", flush=True)
            t0 = time.time()
            m = SentenceTransformer(MODEL_NAME, trust_remote_code=True,
                                    model_kwargs=model_kwargs,
                                    config_kwargs={"trust_remote_code": True})
            m.max_seq_length = MAX_SEQ_LEN
            print(f"[LOAD] ok via attn={attn} in {time.time() - t0:.0f}s", flush=True)
            return m, attn
        except Exception as e:
            last_err = e
            print(f"[LOAD] attn={attn} failed: {type(e).__name__}: {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError(f"all attention backends failed; last: {last_err}")


def probe_batch_size(model, start):
    """Find the largest batch (halving from `start`) that survives a short encode.
    ST sorts by length internally, so a warmup on the longest texts is
    representative of the worst case in the full run."""
    longest = sorted(uniq_texts, key=len, reverse=True)[:WARMUP_TEXTS]
    bs = start
    while bs >= BATCH_MIN:
        try:
            model.encode(longest, batch_size=bs, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
            print(f"[PROBE] batch_size={bs} ok", flush=True)
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"[PROBE] batch_size={bs} OOM -> halving", flush=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"[PROBE] batch_size={bs} OOM -> halving", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        bs //= 2
    raise RuntimeError(f"OOM even at batch_size={BATCH_MIN}")


def encode_all(model, texts, batch_size):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    torch.cuda.synchronize()
    secs = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return np.asarray(emb, dtype=np.float32), secs, peak_gb


def run_pass(tag, quantized, save_to=None):
    model, attn = load_model(quantized)

    if "n_tokens" not in results.get("data", {}):
        lens = [len(x) for x in model.tokenizer(uniq_texts, add_special_tokens=True)["input_ids"]]
        results["data"]["avg_tokens"] = float(np.mean(lens))
        results["data"]["max_tokens"] = int(np.max(lens))
        results["data"]["n_tokens"] = int(np.sum(lens))
        print(f"[DATA] tokens/item: avg {np.mean(lens):.1f} max {np.max(lens)} "
              f"total {np.sum(lens):,}", flush=True)
        save_results()

    entry = {"attn": attn}

    if save_to is not None:
        # Save BEFORE encoding: the quality numbers are the run's main product,
        # but a save failure must not cost us the encode that follows it.
        try:
            if os.path.isdir(save_to):
                shutil.rmtree(save_to)
            t0 = time.time()
            model.save(save_to)
            size_gb = sum(os.path.getsize(os.path.join(dp, f))
                          for dp, _, fs in os.walk(save_to) for f in fs) / 1024**3
            entry["saved"] = True
            entry["on_disk_gb"] = size_gb
            entry["files"] = sorted(os.listdir(save_to))
            print(f"[{tag}] saved to {save_to} in {time.time() - t0:.0f}s | "
                  f"on-disk {size_gb:.2f}GB (5GB zip limit)", flush=True)
        except Exception as e:
            entry["saved"] = False
            entry["save_error"] = f"{type(e).__name__}: {e}"
            print(f"[{tag}] SAVE FAILED: {type(e).__name__}: {e}", flush=True)

    bs = probe_batch_size(model, BATCH_START)
    emb, secs, peak = encode_all(model, uniq_texts, bs)
    dim = emb.shape[1]
    id_to_vec = {i: emb[k] for k, i in enumerate(uniq_ids)}
    macro, overall, ncat = score_val(id_to_vec, dim)

    entry.update({
        "macro_pr_auc": macro, "overall_ap": overall, "n_cat_scored": ncat,
        "batch_size": bs, "dim": int(dim), "encode_secs": secs,
        "items_per_s": len(uniq_texts) / secs,
        "tokens_per_s": results["data"]["n_tokens"] / secs,
        "peak_vram_gb": peak,
    })
    results[tag] = entry
    save_results()

    print(f"[{tag}] macro_pr_auc={macro:.4f} overall_ap={overall:.4f} "
          f"cats={ncat} | {entry['items_per_s']:.1f} items/s "
          f"({entry['tokens_per_s']:,.0f} tok/s) | peak VRAM {peak:.1f}GB | dim {dim}",
          flush=True)

    del model, emb, id_to_vec
    gc.collect(); torch.cuda.empty_cache()
    return entry


from sentence_transformers import SentenceTransformer

# ------------------------------------------------- pass 1: full precision
print("=" * 80 + f"\n[{DTYPE_TAG.upper()}] full-precision pass", flush=True)
run_pass(DTYPE_TAG, quantized=False)

# ------------------------------------------------- pass 2: int8 + save
print("=" * 80 + "\n[INT8] 8-bit pass (load, save quantized weights, re-encode)", flush=True)
try:
    run_pass("int8", quantized=True, save_to=INT8_DIR)
except Exception as e:
    results["int8"] = {"failed": f"{type(e).__name__}: {e}"}
    save_results()
    print(f"[INT8] PASS FAILED: {type(e).__name__}: {e}", flush=True)


# ---------------------------------------------------------------- projections
peak_key = next((k for k in GPU_PEAK_TFLOPS if k in gpu_name), None)
ratio = (H100_PEAK / GPU_PEAK_TFLOPS[peak_key]) if peak_key else 1.0

for tag in (DTYPE_TAG, "int8"):
    e = results.get(tag) or {}
    if "items_per_s" not in e:
        continue
    ips = e["items_per_s"]
    h100_ips = ips * ratio
    e["h100_items_per_s_est"] = h100_ips
    for n in (PRIVATE_ITEMS_ESTIMATE, 500_000):
        print(f"[PROJ {tag}] {n:,} items: this GPU {n / ips / 60:.1f}min | "
              f"H100 est {n / h100_ips / 60:.1f}min (x{ratio:.1f}, indicative)", flush=True)
save_results()


# ---------------------------------------------------------------- summary
print("=" * 80 + "\n[SUMMARY]\n" + json.dumps(results, indent=2, default=str), flush=True)

print("\n[VERDICT]")
full = results.get(DTYPE_TAG, {})
i8 = results.get("int8", {})
if "macro_pr_auc" in full:
    print(f"  raw {DTYPE_TAG} macro_pr_auc = {full['macro_pr_auc']:.4f} "
          f"on {len(val):,} val pairs")
    print(f"    references: random floor = {RANDOM_FLOOR:.4f} | "
          f"baseline = 0.3710 | LoRA cross-encoder = 0.5282")
if "macro_pr_auc" in i8:
    print(f"  raw int8 macro_pr_auc = {i8['macro_pr_auc']:.4f} "
          f"(quantization delta = {i8['macro_pr_auc'] - full.get('macro_pr_auc', float('nan')):+.4f})")
if i8.get("saved"):
    print(f"  int8 on-disk = {i8['on_disk_gb']:.2f}GB (5GB zip limit) at {INT8_DIR}")
    print("  -> download via `kaggle kernels output maxamsecret/giga3b-quantize-benchmark`")
else:
    print(f"  int8 save did NOT succeed: {i8.get('save_error') or i8.get('failed')}")
print("[DONE]", flush=True)
