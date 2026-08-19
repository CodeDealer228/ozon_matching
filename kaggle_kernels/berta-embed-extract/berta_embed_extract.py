"""
Extracts BERTA+LoRA pooled pair embeddings (the same mean-pooled vector that
normally feeds the classifier's Linear layer, taken BEFORE that layer) for
the canonical human train split (292,523 pairs) and val split (73,131
pairs), using the current best checkpoint -- rays0n/berta-llm-continue-train
(LLM continue-train experiment, val_macro_pr_auc=0.7255, superseding
berta-lora-sft-continue-5-epochs's own 0.7246 checkpoint). Feeds the
CatBoost-on-embeddings track:
../berta-catboost-noleak trains only on val-split embeddings (both BERTA's
own train and this catboost's train stay disjoint -- no leakage); ../
berta-catboost-leak trains on train-split embeddings and validates on val
(BERTA saw these train pairs during its own training, so this variant is
an explicit, labeled leakage comparison, not the real submission candidate).

Architecture/resume mechanics (ClassifierHead dropout wrapper, MEAN pooling,
autocast-inside-forward, human item cache, category excision) are unchanged
from berta-lora-sft-continue-5-epochs -- see that script or ../berta-hardneg-
balanced's module docstring for the full rationale, not repeated here. This
kernel only ever runs forward passes (no_grad, eval mode) -- no optimizer,
no training loop, no checkpoint saving. EVAL_PROBE_CAP=4096 (not the older
512) per the same real-measured headroom finding as the other post-fix
scripts in this project (batch=512 only used 1.69/11GB budget).
"""
import os
import sys
import gc
import ctypes
import json
import shutil
import time

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer, set_seed
from sklearn.model_selection import train_test_split

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "sergeyzh/BERTA"

ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 128
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64
PAIR_MAX_LENGTH = 128

CLASSIFIER_DROPOUT = 0.10   # unused at inference (eval mode zeroes dropout),
                              # only needed to reconstruct ClassifierHead's
                              # shape so its state_dict loads cleanly.

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 3.0
EXTRACT_BATCH_SIZE = 32   # probe starting point
EVAL_PROBE_CAP = 1024   # was 4096 -- a real run hit a low-level "CUDA error:
                           # unspecified launch failure" probing candidate=2048
                           # right after a clean, successful batch=1024 pass
                           # (2.85/11GB used); capping at the last known-good
                           # value rather than chasing a flaky/async CUDA fault
                           # for marginal extra throughput on an inference-only
                           # (no backward) extraction job.
PROBE_ROUND_TO = 4

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 16.0
RAM_HARD_GB = 24.0
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_embeddings")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


# =====================================================================
# HELPERS (unchanged from the rest of this project's BERTA scripts)
# =====================================================================

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


def _log_memory(stage):
    rss_gb = _current_rss_gb()
    if rss_gb is None:
        print(f"[MEM] {stage}: unavailable")
        return
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        print(f"[MEM] {stage}: rss={rss_gb:.2f}GB available={avail:.2f}GB")
    except ImportError:
        print(f"[MEM] {stage}: rss={rss_gb:.2f}GB")


def _require_disk_space(path, required_bytes, label, margin=1.15):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    need = int(required_bytes * margin)
    if free < need:
        raise RuntimeError(f"[DISK] insufficient space for {label}: need {need / 1e9:.2f}GB, have {free / 1e9:.2f}GB")


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def build_item_texts(items_lazy):
    return (
        items_lazy.select(["id", "name", "category", "attributes"])
        .with_columns([
            pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS),
            pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_CATEGORY_CHARS),
            pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS),
        ])
        .with_columns(
            (pl.lit("Name: ") + pl.col("name") + pl.lit(" Category: ") + pl.col("category") +
             pl.lit(" Attributes: ") + pl.col("attributes")).str.slice(0, MAX_ITEM_CHARS).alias("text")
        )
        .select(["id", "text"])
        .collect()
    )


def tokenize_items_to_memmap(items_pl, tokenizer, out_dir, max_length=ITEM_MAX_LENGTH):
    n = items_pl.height
    ids_path = os.path.join(out_dir, "ids.bin")
    input_ids_path = os.path.join(out_dir, "input_ids.bin")
    attn_path = os.path.join(out_dir, "attention_mask.bin")
    bytes_per_row = 8 + max_length * 4 + max_length * 1
    _require_disk_space(out_dir, n * bytes_per_row, "item token cache")

    ids_mm = np.memmap(ids_path, dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(input_ids_path, dtype=np.int32, mode="w+", shape=(n, max_length))
    attn_mm = np.memmap(attn_path, dtype=np.int8, mode="w+", shape=(n, max_length))

    all_ids = items_pl.get_column("id").to_numpy()
    all_texts = items_pl.get_column("text").to_list()

    chunk_size = TOKENIZE_CHUNK_SIZE_START
    written = 0
    pbar = tqdm(total=n, desc="Tokenize items", dynamic_ncols=True)
    while written < n:
        cur = min(chunk_size, n - written)
        chunk_texts = [PREFIX + t for t in all_texts[written:written + cur]]
        enc = tokenizer(chunk_texts, add_special_tokens=False, padding="max_length",
                         truncation=True, max_length=max_length, return_tensors="np")
        ids_mm[written:written + cur] = all_ids[written:written + cur]
        input_ids_mm[written:written + cur] = enc["input_ids"].astype(np.int32)
        attn_mm[written:written + cur] = enc["attention_mask"].astype(np.int8)
        written += cur
        pbar.update(cur)
        gc.collect()
        _malloc_trim()
        rss_gb = _current_rss_gb()
        if rss_gb is not None:
            if rss_gb < RAM_TARGET_GB:
                chunk_size = min(TOKENIZE_CHUNK_SIZE_MAX, int(chunk_size * RAM_GROWTH_FACTOR))
            elif rss_gb > RAM_HARD_GB:
                chunk_size = max(TOKENIZE_CHUNK_SIZE_MIN, int(chunk_size * RAM_SHRINK_FACTOR))
    pbar.close()
    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()


def _find_token_pattern_starts(arr, pattern):
    pattern = np.asarray(pattern)
    plen = len(pattern)
    n_positions = arr.shape[1] - plen + 1
    match = np.ones((arr.shape[0], n_positions), dtype=bool)
    for offset, tok in enumerate(pattern):
        match &= (arr[:, offset:offset + n_positions] == tok)
    rows, cols = np.where(match)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def build_item_cache(tokenizer):
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    items = build_item_texts(pl.scan_parquet(items_path))
    n_items = items.height
    print(f"[CACHE] human scope items: {n_items:,}")

    cache_dir = os.path.join(WORKING_DIR, "berta_item_tokens")
    os.makedirs(cache_dir, exist_ok=True)
    tokenize_items_to_memmap(items, tokenizer, cache_dir)
    all_ids = items.get_column("id").to_numpy().copy()
    del items
    gc.collect()

    item_input_ids = np.asarray(np.memmap(os.path.join(cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                                           shape=(n_items, ITEM_MAX_LENGTH))).copy()
    item_attention_mask = np.asarray(np.memmap(os.path.join(cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                                                shape=(n_items, ITEM_MAX_LENGTH))).copy()

    id_to_row = {int(item_id): i for i, item_id in enumerate(all_ids)}
    item_lengths = item_attention_mask.sum(axis=1).astype(np.int32)
    category_label_start = _find_token_pattern_starts(item_input_ids, CATEGORY_LABEL_PATTERN)
    attr_label_start = _find_token_pattern_starts(item_input_ids, ATTRIBUTES_LABEL_PATTERN)

    return item_input_ids, item_lengths, category_label_start, attr_label_start, id_to_row


def allocate_pair_budget(len1, len2, budget):
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def build_compact_ids(item_input_ids, row, item_length, category_start, attr_start):
    if category_start < 0 or attr_start < 0:
        return item_input_ids[row, :item_length]
    head = item_input_ids[row, :category_start]
    tail = item_input_ids[row, attr_start:item_length]
    return np.concatenate((head, tail))


class PairIndexDataset(Dataset):
    def __init__(self, row1, row2):
        self.row1 = row1
        self.row2 = row2

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, idx):
        return self.row1[idx], self.row2[idx]


def make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start,
                          cls_id, sep_id, pad_id, pair_max_length):
    content_budget = pair_max_length - 3

    def collate_fn(batch):
        row1, row2 = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        token_type_ids = np.zeros((bsz, pair_max_length), dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            r1, r2 = row1[i], row2[i]
            compact1 = build_compact_ids(item_input_ids, r1, int(item_lengths[r1]),
                                          int(category_label_start[r1]), int(attr_label_start[r1]))
            compact2 = build_compact_ids(item_input_ids, r2, int(item_lengths[r2]),
                                          int(category_label_start[r2]), int(attr_label_start[r2]))
            cl1, cl2 = len(compact1), len(compact2)
            if cl1 + cl2 <= content_budget:
                keep1, keep2 = cl1, cl2
            else:
                keep1, keep2 = allocate_pair_budget(cl1, cl2, content_budget)
            seg1 = compact1[:keep1]
            seg2 = compact2[:keep2]
            n1 = 1 + keep1 + 1
            n2 = keep2 + 1
            n = n1 + n2
            input_ids[i, 0] = cls_id
            input_ids[i, 1:1 + keep1] = seg1
            input_ids[i, 1 + keep1] = sep_id
            input_ids[i, n1:n1 + keep2] = seg2
            input_ids[i, n1 + keep2] = sep_id
            token_type_ids[i, n1:n] = 1
            attention_mask[i, :n] = 1

        return (torch.from_numpy(input_ids), torch.from_numpy(attention_mask), torch.from_numpy(token_type_ids))

    return collate_fn


def make_pair_loader(pairs_pd, id_to_row, item_input_ids, item_lengths, category_label_start, attr_label_start,
                      cls_id, sep_id, pad_id, pair_max_length, batch_size):
    row1 = pairs_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = pairs_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    dataset = PairIndexDataset(row1, row2)
    collate_fn = make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start,
                                       cls_id, sep_id, pad_id, pair_max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                       collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, pooled):
        return self.linear(self.dropout(pooled))


class BertaEncoderPooler(nn.Module):
    """Same encoder + MEAN pooling as BertaCrossEncoderClassifier, but stops
    BEFORE the classifier -- returns the pooled vector itself."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).float()


def load_pooler_from_checkpoint(device, adapter_dir):
    print(f"[LOAD] base encoder {MODEL_NAME} + adapter from {adapter_dir} ...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model = BertaEncoderPooler(peft_model).to(device)
    model.eval()
    return model, hidden_dim


def _probe_peak_gb(batch_size, n_gpu, pair_max_length, probe_model, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros((batch_size, pair_max_length), dtype=torch.long, device=device)
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)
    with torch.no_grad():
        out = probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask,
                           token_type_ids=dummy_token_type_ids)
    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))
    del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids, out
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(model, device, n_gpu, pair_max_length, start_batch, probe_cap):
    if not torch.cuda.is_available():
        return start_batch
    print(f"[PROBE] searching for max extract batch under {TARGET_VRAM_GB}GB/GPU "
          f"(margin {SAFETY_MARGIN_GB}GB), capped at {probe_cap} ...")
    candidate = start_batch
    last_good = None
    while candidate <= probe_cap:
        try:
            peak_gb = _probe_peak_gb(candidate, n_gpu, pair_max_length, model, device)
            print(f"[PROBE] batch={candidate:5d} -> peak={peak_gb:.2f}GB/GPU")
            if peak_gb >= TARGET_VRAM_GB - SAFETY_MARGIN_GB:
                break
            last_good = candidate
            candidate *= 2
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] batch={candidate:5d} -> OOM")
            break
        gc.collect()
    if last_good is None:
        last_good = max(PROBE_ROUND_TO, start_batch // 2)
    lo, hi = last_good, min(candidate, probe_cap)
    while hi - lo > PROBE_ROUND_TO:
        mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
        if mid <= lo:
            break
        try:
            peak_gb = _probe_peak_gb(mid, n_gpu, pair_max_length, model, device)
            print(f"[PROBE] batch={mid:5d} -> peak={peak_gb:.2f}GB/GPU")
            if peak_gb >= TARGET_VRAM_GB - SAFETY_MARGIN_GB:
                hi = mid
            else:
                lo = mid
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            hi = mid
        gc.collect()
    print(f"[PROBE] Selected extract batch={lo}")
    return lo


def _find_resume_checkpoint_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "train_state.json" in files and "adapter" in _dirs:
                return root
    return None


@torch.no_grad()
def extract_split(model, loader, device):
    all_emb = []
    for batch in tqdm(loader, desc="Extract", dynamic_ncols=True):
        input_ids, attention_mask, token_type_ids = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = token_type_ids.to(device, non_blocking=True)
        pooled = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_emb.append(pooled.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("BERTA+LoRA pooled embedding extraction")
    print("=" * 90)

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    matches_pl = pl.read_parquet(matches_path)
    matches_pl = (
        matches_pl
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([
            pl.col("target").fill_null(0).cast(pl.Float32),
            (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
        ])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()
    del matches_pl
    gc.collect()

    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels_all)
    train_pd = matches_pd.iloc[train_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    print(f"[DATA] train={len(train_pd):,} val={len(val_pd):,} (expect 292,523 / 73,131)")
    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    item_input_ids, item_lengths, category_label_start, attr_label_start, id_to_row = build_item_cache(tokenizer)
    _log_memory("after item cache")

    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([train_pd["id1"], train_pd["id2"], val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_path)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    train_pd["category1"] = train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
    gc.collect()

    checkpoint_dir = _find_resume_checkpoint_dir()
    if checkpoint_dir is None:
        raise RuntimeError("Checkpoint not found under /kaggle/input -- expected "
                            "rays0n/berta-llm-continue-train mounted (kernel_sources).")
    model, hidden_dim = load_pooler_from_checkpoint(device, os.path.join(checkpoint_dir, "adapter"))
    print(f"[MODEL] hidden_dim={hidden_dim}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1:
        print(f"[GPU] Using DataParallel on {n_gpu} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))

    extract_batch_size = run_vram_probe(model, device, n_gpu, PAIR_MAX_LENGTH, EXTRACT_BATCH_SIZE, EVAL_PROBE_CAP)

    for split_name, split_pd in (("val", val_pd), ("train", train_pd)):
        t0 = time.perf_counter()
        loader = make_pair_loader(split_pd, id_to_row, item_input_ids, item_lengths,
                                   category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                                   PAIR_MAX_LENGTH, extract_batch_size)
        emb = extract_split(model, loader, device)
        print(f"[EXTRACT] {split_name}: {emb.shape} in {time.perf_counter() - t0:.1f}s")

        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_embeddings.npy"), emb.astype(np.float32))
        split_pd[["id1", "id2", "label", "category1"]].to_parquet(
            os.path.join(OUTPUT_DIR, f"{split_name}_meta.parquet"), index=False
        )
        del emb, loader
        gc.collect()
        _malloc_trim()
        _log_memory(f"after extract {split_name}")

    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME, "hidden_dim": hidden_dim, "pair_max_length": PAIR_MAX_LENGTH,
            "source_checkpoint": "rays0n/berta-llm-continue-train (val_macro_pr_auc=0.7255)",
            "pooling": "mean (mask-weighted), BEFORE the classifier head",
        }, f, indent=2)

    print(f"[DONE] embeddings extracted -> {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
