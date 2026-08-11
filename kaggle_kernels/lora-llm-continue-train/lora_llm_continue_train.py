"""
Continue LoRA SFT of the already-trained cross-encoder checkpoint
(maxamsecret/ozon-lora-minilm-finetuned, best/ = epoch 7, val macro PR-AUC
0.5282) on matches_llm.parquet -- the large-scale LLM-labeled pool
(11.19M pairs, continuous target in [0,1], never used before).

Key differences from notebooks/cross-encoder-sft-pipline (3).ipynb (which
this script extends, not replaces):
  - Trains on matches_llm.parquet instead of matches.parquet. The soft label
    is fed directly into BCEWithLogitsLoss (no >=0.5 threshold) -- that's the
    whole point of using LLM-scored data instead of human binary labels.
  - Validates on the *same* canonical val split used for the original LoRA
    run (recomputed inline from matches.parquet, byte-identical code) --
    matches_llm and matches.parquet are confirmed disjoint in both pairs and
    item-id universe, so there is no leakage risk, but a defensive anti-join
    against val is kept as cheap insurance and will warn loudly if it ever
    finds an overlap on Kaggle's hosted data.
  - Resumes the SAME LoRA adapter + classifier head (PeftModel.from_pretrained
    with is_trainable=True) instead of a fresh get_peft_model() init.
  - Validates + checkpoints every ~10% of steps within the single epoch
    (not just at epoch end), so the best-so-far adapter is never far from
    disk, and pushes it as a new dataset version at 25/50/75/100% progress
    milestones so progress survives a hard Kaggle session crash too.
  - LR=4e-4 (2x the original run's 2e-4) with a cosine schedule instead of
    linear -- see kaggle_kernels/lora-llm-continue-train/../../CLAUDE.md and
    the plan this script was built from for the log-based reasoning.

Structured as importable functions + a main() guarded by
`if __name__ == "__main__"` so the mechanics (tokenize_to_memmap, the
checkpoint/windowed-metric logic, compute_metrics, etc.) can be exercised
locally against synthetic data before spending real Kaggle GPU quota --
Kaggle itself runs this file top-to-bottom exactly the same either way,
since __name__ == "__main__" there too.
"""

import os
import sys
import gc
import ctypes
import json
import math
import shutil
import hashlib
import subprocess

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel, get_scheduler, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

try:
    import peft.import_utils
    import peft.tuners.lora.torchao

    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    print("[INFO] torchao disabled for PEFT")
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

ITEMS_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
MATCHES_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
LORA_DIR = "/kaggle/input/datasets/maxamsecret/ozon-lora-minilm-finetuned/lora_mini_self_attention_memmap/best"

MAX_LENGTH = 128
MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640

EPOCHS = 1
LR = 4e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
N_CHECKPOINTS = 10
PUSH_MILESTONES = [0.25, 0.50, 0.75, 1.00]

CHUNK_SIZE = 8_000
TOKENIZE_BATCH_START = 8_000
TOKENIZE_BATCH_MIN = 1_000

TOTAL_BATCH_SIZE = 2048
MICRO_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 1.5
PROBE_CAP = 8192
PROBE_ROUND_TO = 8

MAX_TRAIN_ROWS = 500_000         # SMOKE TEST VALUE -- set back to None before the real run
ENABLE_DATASET_PUSH = False      # flip to True only after Kaggle Secrets are set (see plan)
LORA_DATASET_ID = "maxamsecret/ozon-lora-minilm-finetuned"
LORA_DATASET_NESTED_PATH = "lora_mini_self_attention_memmap/best"

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_llm_continue_train")
CACHE_DIR = os.path.join(WORKING_DIR, "token_cache_llm_continue")
MLFLOW_TRACKING_DIR = os.path.join(WORKING_DIR, "experiments", "mlruns")


# =====================================================================
# SMALL UTILITIES
# =====================================================================

def _malloc_trim():
    # gc.collect() frees Python objects but glibc malloc often keeps the
    # underlying arena pages reserved rather than returning them to the OS,
    # so RSS can stay elevated long after the objects are gone -- observed
    # concretely during development: RSS grew ~19.5GB while tokenizing only
    # ~93K rows (val+train combined) despite del + gc.collect() every chunk,
    # which is out of proportion to the ~774 bytes/row actually written to
    # the memmaps. malloc_trim(0) forces glibc to release what it can back
    # to the OS. No-op (caught) on non-glibc platforms (e.g. local Windows
    # dev/test runs) -- only real hardening happens on the Kaggle container.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _log_cgroup_limit():
    # psutil.virtual_memory()/os.sysconf report HOST-level RAM, which on a
    # shared Docker/Kubernetes host (as Kaggle kernels run on) can overstate
    # what this specific container is actually allowed to use -- a run can
    # get Linux-OOM-killed well before the host-level "available" figure
    # looks low. Read the cgroup limit directly so this mismatch is visible
    # in the logs instead of silently confusing "available" readings.
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw.isdigit():
                print(f"[MEM] cgroup limit ({path}): {int(raw) / 1e9:.2f}GB")
            else:
                print(f"[MEM] cgroup limit ({path}): {raw}")
            return
        except Exception:
            continue
    print("[MEM] cgroup limit: unavailable (not running under a readable cgroup)")


def _log_memory(stage):
    try:
        import psutil
        p = psutil.Process(os.getpid())
        print(f"[MEM] {stage}: rss={p.memory_info().rss / 1e9:.2f}GB "
              f"available={psutil.virtual_memory().available / 1e9:.2f}GB")
        return
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    print(f"[MEM] {stage}: rss={int(line.split()[1]) / 1e6:.2f}GB (via /proc, psutil unavailable)")
                    return
    except Exception:
        pass
    print(f"[MEM] {stage}: unavailable")


def _require_disk_space(path, required_bytes, label, margin=1.35):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    need = int(required_bytes * margin)
    print(f"[DISK] {label}: free={free / 1e9:.2f}GB required(+35% margin)={need / 1e9:.2f}GB")
    if free < need:
        raise RuntimeError(f"[DISK] insufficient space for {label}: need {need / 1e9:.2f}GB, have {free / 1e9:.2f}GB")


def compute_metrics(y_true, y_prob, categories=None):
    # Competition metric: average_precision_score computed separately per
    # category, then a plain mean (macro) across categories. NOT macro over
    # the match/non-match classes -- see CLAUDE.md for why that distinction
    # matters (an earlier run reported the wrong quantity under this name).
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), 0

    pos_pr_auc = float(average_precision_score(y_true, y_prob))

    if categories is None:
        return pos_pr_auc, pos_pr_auc, 0

    categories = np.asarray(categories)
    per_category_aps = []
    for cat in np.unique(categories):
        mask = categories == cat
        y_t, y_p = y_true[mask], y_prob[mask]
        if len(np.unique(y_t)) < 2:
            continue
        per_category_aps.append(average_precision_score(y_t, y_p))

    if not per_category_aps:
        return float("nan"), pos_pr_auc, 0

    return float(np.mean(per_category_aps)), pos_pr_auc, len(per_category_aps)


# =====================================================================
# TOKENIZATION / MEMMAP CACHE
# =====================================================================

class IndexDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return idx


def _memmap_paths(cache_dir, prefix):
    return {
        "input_ids": os.path.join(cache_dir, f"{prefix}_input_ids.bin"),
        "attention_mask": os.path.join(cache_dir, f"{prefix}_attention_mask.bin"),
        "token_type_ids": os.path.join(cache_dir, f"{prefix}_token_type_ids.bin"),
        "labels": os.path.join(cache_dir, f"{prefix}_labels.bin"),
        "category": os.path.join(cache_dir, f"{prefix}_category.bin"),
    }


def _memmap_cache_valid(paths, n_max, max_length, label_dtype):
    expected = {
        "input_ids": n_max * max_length * np.dtype(np.int32).itemsize,
        "attention_mask": n_max * max_length * np.dtype(np.int8).itemsize,
        "token_type_ids": n_max * max_length * np.dtype(np.int8).itemsize,
        "labels": n_max * np.dtype(label_dtype).itemsize,
        "category": n_max * np.dtype(np.int16).itemsize,
    }
    return all(os.path.exists(p) and os.path.getsize(p) == expected[k] for k, p in paths.items())


def _tokenize_adaptive(tokenizer, t1, t2, max_length, batch_size_box, batch_size_min):
    """Yields (enc, chunk_start, batch_len). batch_size_box is a 1-element
    list mutated in place so a shrink (on MemoryError) persists as the
    starting point for later calls in this same tokenize_to_memmap run,
    instead of retrying the too-big size every chunk. A real Linux
    OOM-kill is often a SIGKILL the process can't catch at all (unlike CUDA
    OOM, which is a clean error) -- this is best-effort coverage for the
    cases Python *can* catch, not an equivalent safety net to the VRAM probe.
    """
    i, n = 0, len(t1)
    while i < n:
        b = min(batch_size_box[0], n - i)
        try:
            enc = tokenizer(
                text=t1[i:i + b], text_pair=t2[i:i + b],
                padding="max_length", truncation=True,
                max_length=max_length, return_tensors="np",
            )
        except MemoryError:
            gc.collect()
            if batch_size_box[0] <= batch_size_min:
                raise RuntimeError(f"tokenizer batch already at floor {batch_size_min}, still OOM'ing")
            batch_size_box[0] = max(batch_size_min, batch_size_box[0] // 2)
            print(f"[TOKENIZE][OOM] backing off tokenize_batch -> {batch_size_box[0]}")
            continue
        yield enc, i, b
        i += b


def tokenize_to_memmap(
    pairs_pd: pd.DataFrame,          # columns: id1, id2, label
    item_texts: pl.DataFrame,        # columns: id, category, text
    category_to_id: dict,
    tokenizer,
    prefix: str,
    max_length: int,
    label_dtype,                     # REQUIRED, no default -- np.float32 for
                                      # soft LLM labels, np.int8 for binary
                                      # val labels. Getting this wrong
                                      # silently truncates soft labels to 0/1
                                      # on memmap write (numpy does not raise).
    cache_dir: str,
    chunk_size: int = CHUNK_SIZE,
    tokenize_batch_start: int = TOKENIZE_BATCH_START,
    tokenize_batch_min: int = TOKENIZE_BATCH_MIN,
):
    n_max = len(pairs_pd)
    paths = _memmap_paths(cache_dir, prefix)

    if _memmap_cache_valid(paths, n_max, max_length, label_dtype):
        print(f"[TOKENIZE] {prefix}: valid cache found ({n_max:,} rows), skipping re-tokenization "
              f"(only survives a re-run within this same live session, not a session crash)")
        return paths, n_max

    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)

    bytes_per_row = (
        max_length * np.dtype(np.int32).itemsize          # input_ids
        + max_length * np.dtype(np.int8).itemsize          # attention_mask
        + max_length * np.dtype(np.int8).itemsize          # token_type_ids
        + np.dtype(label_dtype).itemsize                   # labels
        + np.dtype(np.int16).itemsize                       # category
    )
    _require_disk_space(cache_dir, n_max * bytes_per_row, f"{prefix} memmap")

    input_ids_mm = np.memmap(paths["input_ids"], dtype=np.int32, mode="w+", shape=(n_max, max_length))
    attention_mm = np.memmap(paths["attention_mask"], dtype=np.int8, mode="w+", shape=(n_max, max_length))
    token_type_mm = np.memmap(paths["token_type_ids"], dtype=np.int8, mode="w+", shape=(n_max, max_length))
    labels_mm = np.memmap(paths["labels"], dtype=label_dtype, mode="w+", shape=(n_max,))
    category_mm = np.memmap(paths["category"], dtype=np.int16, mode="w+", shape=(n_max,))

    item_texts_id1 = item_texts.rename({"id": "id1", "text": "text1", "category": "category1"})
    item_texts_id2 = item_texts.select(["id", "text"]).rename({"id": "id2", "text": "text2"})

    unknown_id = category_to_id.get("UNKNOWN", -1)
    tokenize_batch_box = [tokenize_batch_start]
    written = 0
    total_chunks = math.ceil(n_max / chunk_size)

    for start in tqdm(range(0, n_max, chunk_size), total=total_chunks, desc=f"Tokenize {prefix}", dynamic_ncols=True):
        chunk_pd = pairs_pd.iloc[start:start + chunk_size]
        chunk_pl = pl.from_pandas(chunk_pd).with_row_index("_row_idx")

        # maintain_order + the row-index sort are belt-and-suspenders: polars
        # currently happens to preserve left-row order on plain inner joins,
        # but that isn't a documented guarantee, and every column pulled
        # below (text1/text2/label/category) comes from this SAME `joined`
        # frame so they stay internally consistent with each other even if
        # it weren't -- this just also keeps them aligned with the original
        # pairs_pd row order, which downstream code doesn't currently rely
        # on but costs nothing to keep true.
        joined = (
            chunk_pl
            .join(item_texts_id1, on="id1", how="inner", maintain_order="left")
            .join(item_texts_id2, on="id2", how="inner", maintain_order="left")
            .sort("_row_idx")
        )

        if joined.height == 0:
            continue

        texts1 = joined.get_column("text1").to_list()
        texts2 = joined.get_column("text2").to_list()
        labels = joined.get_column("label").to_numpy()
        cat_ids = np.array(
            [category_to_id.get(c, unknown_id) for c in joined.get_column("category1").to_list()],
            dtype=np.int16,
        )

        for enc, t_start, b in _tokenize_adaptive(
            tokenizer, texts1, texts2, max_length, tokenize_batch_box, tokenize_batch_min
        ):
            if written + b > n_max:
                b = n_max - written
            if b <= 0:
                break

            input_ids_mm[written:written + b] = enc["input_ids"][:b].astype(np.int32)
            attention_mm[written:written + b] = enc["attention_mask"][:b].astype(np.int8)
            if "token_type_ids" in enc:
                token_type_mm[written:written + b] = enc["token_type_ids"][:b].astype(np.int8)
            else:
                token_type_mm[written:written + b] = 0
            labels_mm[written:written + b] = labels[t_start:t_start + b]
            category_mm[written:written + b] = cat_ids[t_start:t_start + b]

            written += b
            if written >= n_max:
                break

        del joined, texts1, texts2, labels, cat_ids, chunk_pl, chunk_pd
        gc.collect()
        _malloc_trim()

    for mm in (input_ids_mm, attention_mm, token_type_mm, labels_mm, category_mm):
        mm.flush()
    del input_ids_mm, attention_mm, token_type_mm, labels_mm, category_mm
    gc.collect()

    return paths, written


def make_memmap_loader(paths, length, max_length, batch_size, shuffle, label_dtype):
    if length <= 0:
        raise ValueError("length must be > 0")

    input_ids_mm = np.memmap(paths["input_ids"], dtype=np.int32, mode="r", shape=(length, max_length))
    attention_mm = np.memmap(paths["attention_mask"], dtype=np.int8, mode="r", shape=(length, max_length))
    token_type_mm = np.memmap(paths["token_type_ids"], dtype=np.int8, mode="r", shape=(length, max_length))
    labels_mm = np.memmap(paths["labels"], dtype=label_dtype, mode="r", shape=(length,))
    category_mm = np.memmap(paths["category"], dtype=np.int16, mode="r", shape=(length,))

    dataset = IndexDataset(length)

    def collate_fn(idx_list):
        idx = np.asarray(idx_list, dtype=np.int64)
        input_ids = torch.from_numpy(np.asarray(input_ids_mm[idx], dtype=np.int64))
        attention_mask = torch.from_numpy(np.asarray(attention_mm[idx], dtype=np.int64))
        token_type_ids = torch.from_numpy(np.asarray(token_type_mm[idx], dtype=np.int64))
        labels = torch.from_numpy(np.asarray(labels_mm[idx], dtype=np.float32))
        categories = np.asarray(category_mm[idx], dtype=np.int64)
        return input_ids, attention_mask, token_type_ids, labels, categories

    drop_last = shuffle and (length >= batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
    )
    handles = [input_ids_mm, attention_mm, token_type_mm, labels_mm, category_mm]
    return loader, handles


# =====================================================================
# MODEL
# =====================================================================

class CrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(out.last_hidden_state[:, 0, :]).view(-1)


def load_resumed_model(device):
    print("[MODEL] Loading base encoder ...")
    try:
        base_model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="sdpa")
    except TypeError:
        base_model = AutoModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size

    print(f"[MODEL] Resuming LoRA adapter from {LORA_DIR} (is_trainable=True) ...")
    peft_model = PeftModel.from_pretrained(base_model, LORA_DIR, is_trainable=True)

    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, (
        "LoRA adapter loaded with requires_grad=False -- is_trainable=True missing or ignored. "
        "Training would silently only move the classifier head."
    )
    print(f"[MODEL] trainable LoRA params: {n_trainable_lora:,}")

    classifier_head = nn.Linear(hidden_dim, 1)
    classifier_state = torch.load(os.path.join(LORA_DIR, "classifier.pt"), map_location="cpu")
    classifier_head.load_state_dict(classifier_state)
    print("[MODEL] Resumed classifier head from classifier.pt")

    model = CrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


# =====================================================================
# VRAM AUTO-PROBE (reused verbatim from the training notebook's philosophy)
# =====================================================================

def _probe_peak_gb(batch_size, n_gpu, max_length, probe_model, probe_loss_fn, trainable_params, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, max_length), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)

    for p in trainable_params:
        p.grad = None
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = probe_model(
            input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, token_type_ids=dummy_token_type_ids
        )
        loss = probe_loss_fn(logits, dummy_labels)
    loss.backward()

    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))

    for p in trainable_params:
        p.grad = None
    del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids, dummy_labels, logits, loss
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(model, trainable_params, device, n_gpu, max_length, start_batch):
    if not torch.cuda.is_available():
        print("[PROBE] No GPU available, skipping batch-size probe.")
        return start_batch

    print(f"[PROBE] Searching for max micro-batch under {TARGET_VRAM_GB:.1f}GB/GPU "
          f"(safety margin {SAFETY_MARGIN_GB:.1f}GB) ...")
    probe_loss_fn = torch.nn.BCEWithLogitsLoss()
    probe_threshold = TARGET_VRAM_GB - SAFETY_MARGIN_GB

    candidate = start_batch
    last_good, last_good_gb = None, None

    while candidate <= PROBE_CAP:
        try:
            peak_gb = _probe_peak_gb(candidate, n_gpu, max_length, model, probe_loss_fn, trainable_params, device)
            print(f"[PROBE] batch={candidate:5d} -> peak={peak_gb:.2f}GB/GPU")
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] batch={candidate:5d} -> OOM")
            break

        last_good, last_good_gb = candidate, peak_gb
        if peak_gb > probe_threshold:
            break
        candidate *= 2

    if last_good is not None and last_good_gb is not None and last_good_gb <= probe_threshold and candidate <= PROBE_CAP:
        lo, hi = last_good, candidate
        while hi - lo > PROBE_ROUND_TO:
            mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
            if mid <= lo:
                break
            try:
                peak_gb = _probe_peak_gb(mid, n_gpu, max_length, model, probe_loss_fn, trainable_params, device)
                print(f"[PROBE] batch={mid:5d} -> peak={peak_gb:.2f}GB/GPU")
                if peak_gb > probe_threshold:
                    hi = mid
                else:
                    lo, last_good, last_good_gb = mid, mid, peak_gb
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                print(f"[PROBE] batch={mid:5d} -> OOM")
                hi = mid

    if last_good is None:
        print(f"[PROBE] First candidate ({start_batch}) already OOM'd -- keeping config value "
              f"and relying on GRAD_ACCUM; consider lowering MAX_LENGTH or MICRO_BATCH_SIZE by hand.")
        return start_batch

    discovered = max(PROBE_ROUND_TO, (last_good // PROBE_ROUND_TO) * PROBE_ROUND_TO)
    print(f"[PROBE] Selected MICRO_BATCH_SIZE={discovered} (was {start_batch}, measured peak={last_good_gb:.2f}GB/GPU)")
    return discovered


# =====================================================================
# EVAL
# =====================================================================

def evaluate(eval_model, loader, loss_fn, device, use_amp):
    # categories come from the SAME memmap the loader reads (5th collate
    # element) instead of a separately pre-computed array -- this removes
    # the class of bug where an external categories array could silently
    # drift out of alignment with the loader's row order.
    eval_model.eval()

    all_logits, all_labels, all_categories = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            input_ids, attention_mask, token_type_ids, batch_labels, batch_categories = batch

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = eval_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                logits = eval_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = loss_fn(logits, batch_labels_dev.float())
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.float().cpu())
            all_labels.append(batch_labels.cpu())
            all_categories.append(batch_categories)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    categories = np.concatenate(all_categories, axis=0)

    val_loss = total_loss / max(1, n_batches)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = (labels.numpy() >= 0.5).astype(int)  # val labels are already binary; kept robust regardless

    macro_pr_auc, pos_pr_auc, n_categories = compute_metrics(y_true, probs, categories)

    del logits, labels, probs, y_true, all_logits, all_labels, all_categories
    gc.collect()

    return {
        "val_loss": float(val_loss),
        "macro_pr_auc": float(macro_pr_auc),
        "pos_pr_auc": float(pos_pr_auc),
        "n_categories_evaluated": int(n_categories),
    }


# =====================================================================
# CRASH-RESILIENT CHECKPOINT PUSH
# =====================================================================

def _bootstrap_kaggle_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = us.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = us.get_secret("KAGGLE_KEY")
        print("[SECRETS] Kaggle API credentials loaded from kernel secrets.")
        return True
    except Exception as e:
        print(f"[WARN] Kaggle secrets unavailable, dataset-version push disabled this run: {e}")
        return False


def _push_best_as_dataset_version(save_dir, message, working_dir):
    push_root = os.path.join(working_dir, "dataset_push_staging")
    nested = os.path.join(push_root, *LORA_DATASET_NESTED_PATH.split("/"))
    try:
        if os.path.exists(push_root):
            shutil.rmtree(push_root)
        os.makedirs(nested, exist_ok=True)
        for fname in os.listdir(save_dir):
            shutil.copy2(os.path.join(save_dir, fname), os.path.join(nested, fname))
        with open(os.path.join(push_root, "dataset-metadata.json"), "w") as f:
            json.dump({"id": LORA_DATASET_ID, "title": "Ozon LoRA MiniLM Fine-tuned"}, f)
        subprocess.check_call([
            "kaggle", "datasets", "version", "-p", push_root,
            "-m", message, "-r", "zip", "-t",
        ])
        print(f"[PUSH] dataset version created: {message}")
        return True
    except Exception as e:
        print(f"[WARN] dataset-version push failed (non-fatal, training continues): {e}")
        return False


def save_checkpoint(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.encoder.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(raw.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))


# =====================================================================
# MAIN
# =====================================================================

def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft", "mlflow"])
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    print("=" * 90)
    print("CONFIG")
    print("=" * 90)
    print(f"model:            {MODEL_NAME}")
    print(f"max_length:       {MAX_LENGTH}")
    print(f"epochs:           {EPOCHS}")
    print(f"lr:               {LR}")
    print(f"total_batch_size: {TOTAL_BATCH_SIZE}")
    print(f"micro_batch_size: {MICRO_BATCH_SIZE}")
    print(f"max_train_rows:   {MAX_TRAIN_ROWS}")
    print(f"enable_dataset_push: {ENABLE_DATASET_PUSH}")
    print(f"cache_dir:        {CACHE_DIR}")
    print(f"output_dir:       {OUTPUT_DIR}")
    print("=" * 90)

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment("ozon-matching")

    has_kaggle_creds = _bootstrap_kaggle_secrets() if ENABLE_DATASET_PUSH else False

    _log_cgroup_limit()
    _log_memory("start")

    # -----------------------------------------------------------------
    # Canonical val split -- byte-identical to the original LoRA training
    # code (filter nulls -> cast label -> select -> train_test_split).
    # Only val_pd is used downstream; matches.parquet's train split is not
    # used for training in this script (matches_llm is used instead).
    # -----------------------------------------------------------------
    print("[DATA] Loading matches.parquet for canonical val split ...")
    matches_pl = pl.read_parquet(MATCHES_PATH)
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
    val_pd = matches_pd.iloc[val_idx].reset_index(drop=True)
    print(f"[DATA] matches.parquet rows: {len(matches_pd):,} | recomputed val rows: {len(val_pd):,} "
          f"(expected 73,131 -- cross-check against CLAUDE.md)")

    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[DATA] val_fingerprint={val_fingerprint}")

    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # -----------------------------------------------------------------
    # matches_llm.parquet -- soft labels, defensive anti-join vs val
    # -----------------------------------------------------------------
    print("[DATA] Loading matches_llm.parquet ...")
    llm_pl = pl.read_parquet(MATCHES_LLM_PATH)
    llm_pl = (
        llm_pl
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null() & pl.col("id1").ne(pl.col("id2")))
        .with_columns(pl.col("target").fill_null(0.0).cast(pl.Float32).alias("label"))
        .select(["id1", "id2", "label"])
    )

    val_pairs_norm = pl.DataFrame({
        "pmin": np.minimum(val_pd["id1"].to_numpy(), val_pd["id2"].to_numpy()),
        "pmax": np.maximum(val_pd["id1"].to_numpy(), val_pd["id2"].to_numpy()),
    }).unique()
    llm_pl = llm_pl.with_columns([
        pl.min_horizontal("id1", "id2").alias("pmin"),
        pl.max_horizontal("id1", "id2").alias("pmax"),
    ])
    n_before = llm_pl.height
    llm_pl = llm_pl.join(val_pairs_norm, on=["pmin", "pmax"], how="anti").drop(["pmin", "pmax"])
    n_dropped = n_before - llm_pl.height
    print(f"[LEAK-GUARD] matches_llm rows before={n_before:,} after anti-join vs val={llm_pl.height:,} "
          f"(dropped {n_dropped:,})")
    if n_dropped > 0:
        print(f"[WARN] {n_dropped:,} matches_llm pairs overlapped the val split and were dropped -- "
              f"Kaggle's hosted matches_llm.parquet may differ from the local copy checked during planning.")

    if MAX_TRAIN_ROWS is not None:
        llm_pl = llm_pl.sample(n=min(MAX_TRAIN_ROWS, llm_pl.height), seed=SEED)
        print(f"[SMOKE] MAX_TRAIN_ROWS set -> sampled down to {llm_pl.height:,} train rows")

    train_pd = llm_pl.to_pandas()
    del llm_pl, val_pairs_norm
    gc.collect()
    print(f"[DATA] train (matches_llm) rows: {len(train_pd):,}")

    # -----------------------------------------------------------------
    # items.parquet -- semi-join to needed ids + streaming collect.
    #
    # Earlier design assumed skipping an id-filter was fine since matches_llm
    # needs most of the ~13.4M-row catalog anyway for the real full run --
    # true for final row count, but WRONG about peak memory: a smoke-test
    # run on Kaggle with this unfiltered chain got OOM-killed (Linux SIGKILL,
    # "Killed" in the log) at RSS=21.7GB/32GB right after this collect(),
    # before tokenization even started. Root cause: `.str.slice()` truncates
    # AFTER polars has already decoded the full-length `attributes` column
    # from parquet (per CLAUDE.md, raw attributes are large enough that even
    # json.loads-ing them at 13M-row scale OOM'd during training) -- a lazy
    # chain alone doesn't prevent that intermediate from being fully
    # materialized. Two fixes, both applied: (1) a semi-join against the ids
    # actually referenced by this run's train+val pairs, which the query
    # optimizer can push down before the expensive string work -- this
    # collapses a capped smoke run (e.g. MAX_TRAIN_ROWS=20_000) from ~13.4M
    # rows to a few hundred thousand, and even for the real full run trims
    # some (matches_llm + human ids together are a few % short of the full
    # catalog); (2) engine="streaming" so the collect pipelines in chunks
    # instead of building the full-length intermediate columns for all rows
    # at once, which is what actually matters for the real full run where
    # the semi-join alone wouldn't remove much.
    # -----------------------------------------------------------------
    needed_ids = pd.unique(
        pd.concat([train_pd["id1"], train_pd["id2"], val_pd["id1"], val_pd["id2"]])
    )
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    print(f"[DATA] unique item ids needed: {len(needed_ids):,}")

    print("[DATA] Loading items.parquet (semi-join to needed ids, streaming, column-pruned) ...")
    items = (
        pl.scan_parquet(ITEMS_PATH)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "name", "category", "attributes"])
        .with_columns([
            pl.col("name").fill_null("").cast(pl.Utf8).str.slice(0, MAX_NAME_CHARS),
            pl.col("category").fill_null("").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS),
            pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS),
        ])
        .with_columns(
            (pl.lit("Name: ") + pl.col("name") + pl.lit(" Category: ") + pl.col("category") +
             pl.lit(" Attributes: ") + pl.col("attributes")).str.slice(0, MAX_ITEM_CHARS).alias("text")
        )
        .select(["id", "category", "text"])
        .collect(engine="streaming")
    )
    print(f"[DATA] items loaded: {items.height:,}")
    del needed_ids, needed_ids_lazy
    gc.collect()
    _malloc_trim()
    _log_memory("after items load")

    category_vocab = sorted(items.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    print(f"[DATA] category vocab size: {len(category_vocab)}")

    # -----------------------------------------------------------------
    # Tokenize -> memmap cache (train: soft float32 labels; val: binary
    # int8 labels -- label_dtype is a required arg specifically so this
    # can't be silently gotten wrong at a call site).
    # -----------------------------------------------------------------
    print("[TOKENIZER] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[TOKENIZE] Building val memmap (int8 binary labels) ...")
    val_paths, val_len = tokenize_to_memmap(
        pairs_pd=val_pd, item_texts=items, category_to_id=category_to_id, tokenizer=tokenizer,
        prefix="val", max_length=MAX_LENGTH, label_dtype=np.int8, cache_dir=CACHE_DIR,
    )
    print(f"[TOKENIZE] val effective rows: {val_len:,}")

    print("[TOKENIZE] Building train memmap (float32 soft labels) ...")
    train_paths, train_len = tokenize_to_memmap(
        pairs_pd=train_pd, item_texts=items, category_to_id=category_to_id, tokenizer=tokenizer,
        prefix="train", max_length=MAX_LENGTH, label_dtype=np.float32, cache_dir=CACHE_DIR,
    )
    print(f"[TOKENIZE] train effective rows: {train_len:,}")

    if train_len == 0 or val_len == 0:
        raise RuntimeError("Train or val dataset is empty after filtering.")

    del items, val_pd, train_pd
    gc.collect()
    _malloc_trim()
    _log_memory("after tokenize (items/val_pd/train_pd freed)")

    # Local, per-run copies of the batch-size config -- the VRAM probe below
    # may discover a different micro-batch size than the static default, and
    # reassigning the module-level constants from inside main() would need a
    # `global` declared before every earlier read in this function (illegal:
    # SyntaxError "name used prior to global declaration"). Plain locals
    # sidestep that entirely.
    micro_batch_size = MICRO_BATCH_SIZE
    eval_batch_size = EVAL_BATCH_SIZE
    total_batch_size = TOTAL_BATCH_SIZE

    train_loader, train_handles = make_memmap_loader(
        train_paths, train_len, MAX_LENGTH, micro_batch_size, shuffle=True, label_dtype=np.float32
    )
    val_loader, val_handles = make_memmap_loader(
        val_paths, val_len, MAX_LENGTH, eval_batch_size, shuffle=False, label_dtype=np.int8
    )

    train_labels_mm = np.memmap(train_paths["labels"], dtype=np.float32, mode="r", shape=(train_len,))
    pos_rate = float(np.mean(train_labels_mm[:]))
    del train_labels_mm
    gc.collect()
    print(f"[DATA] train mean soft label: {pos_rate:.4f}")

    # -----------------------------------------------------------------
    # Model: resume LoRA + classifier head, DataParallel across both T4s
    # -----------------------------------------------------------------
    model = load_resumed_model(device)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1:
        print(f"[GPU] Using DataParallel on {n_gpu} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
    else:
        print("[GPU] Using single device")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"[MODEL] Total params: {total_params:,} | Trainable: {trainable_params_count:,} "
          f"({100.0 * trainable_params_count / max(1, total_params):.4f}%)")
    _log_memory("after model load")

    discovered = run_vram_probe(model, trainable_params, device, n_gpu, MAX_LENGTH, micro_batch_size)
    grad_accum = max(1, total_batch_size // micro_batch_size)

    if discovered != micro_batch_size:
        micro_batch_size = discovered
        eval_batch_size = micro_batch_size * 2
        if micro_batch_size >= total_batch_size:
            total_batch_size = micro_batch_size
            grad_accum = 1
        else:
            grad_accum = max(1, round(total_batch_size / micro_batch_size))
            total_batch_size = micro_batch_size * grad_accum

        print("[PROBE] Rebuilding data loaders with the new batch size ...")
        train_loader, train_handles = make_memmap_loader(
            train_paths, train_len, MAX_LENGTH, micro_batch_size, shuffle=True, label_dtype=np.float32
        )
        val_loader, val_handles = make_memmap_loader(
            val_paths, val_len, MAX_LENGTH, eval_batch_size, shuffle=False, label_dtype=np.int8
        )
        print(f"[PROBE] New config: MICRO_BATCH_SIZE={micro_batch_size} | EVAL_BATCH_SIZE={eval_batch_size} | "
              f"GRAD_ACCUM={grad_accum} | effective TOTAL_BATCH_SIZE={total_batch_size}")
    _log_memory("after probe")

    # -----------------------------------------------------------------
    # Optimizer / cosine schedule -- computed AFTER the probe, since it may
    # have rebuilt the loader / grad_accum around the discovered batch size.
    # -----------------------------------------------------------------
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_training_steps * WARMUP_RATIO))

    scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )
    print(f"[TRAIN] steps per epoch: {num_update_steps_per_epoch}")
    print(f"[TRAIN] total steps:     {total_training_steps}")
    print(f"[TRAIN] warmup steps:    {warmup_steps} ({WARMUP_RATIO:.0%})")

    if pos_rate <= 0.0 or pos_rate >= 1.0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(np.clip((1.0 - pos_rate) / pos_rate, 0.1, 20.0))
    print(f"[LOSS] BCE pos_weight: {pos_weight_value:.4f}")
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_interval = max(1, math.ceil(num_update_steps_per_epoch / N_CHECKPOINTS))
    print(f"[TRAIN] checkpoint every {checkpoint_interval} optimizer steps (~{N_CHECKPOINTS} per epoch)")

    # -----------------------------------------------------------------
    # Train loop with intra-epoch validation/checkpointing
    # -----------------------------------------------------------------
    mlflow.start_run(run_name="lora_llm_continue_train")
    mlflow.log_params({
        "variant": "lora_llm_continue_train",
        "encoder": MODEL_NAME,
        "resumed_from": LORA_DIR,
        "max_length": MAX_LENGTH,
        "micro_batch_size": micro_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_total_batch_size": total_batch_size,
        "lr": LR,
        "scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "split_seed": SEED,
        "val_fingerprint": val_fingerprint,
        "train_rows": train_len,
        "val_rows": val_len,
        "n_checkpoints_per_epoch": N_CHECKPOINTS,
        "enable_dataset_push": ENABLE_DATASET_PUSH and has_kaggle_creds,
    })

    log_rows = []
    best_macro = -1.0
    best_since_last_push = False
    push_milestones_remaining = list(PUSH_MILESTONES)
    global_step = 0

    window_logits, window_labels_soft, window_categories = [], [], []
    window_loss_sum, window_loss_count = 0.0, 0

    def run_checkpoint():
        nonlocal window_logits, window_labels_soft, window_categories
        nonlocal window_loss_sum, window_loss_count, best_macro, best_since_last_push
        nonlocal push_milestones_remaining

        train_logits = torch.cat(window_logits, dim=0).numpy()
        train_labels_soft = torch.cat(window_labels_soft, dim=0).numpy()
        train_cats = np.concatenate(window_categories, axis=0)
        train_probs = 1.0 / (1.0 + np.exp(-train_logits))
        train_labels_bin = (train_labels_soft >= 0.5).astype(int)
        train_macro, train_pos_ap, _ = compute_metrics(train_labels_bin, train_probs, train_cats)
        train_window_loss = window_loss_sum / max(1, window_loss_count)

        window_logits, window_labels_soft, window_categories = [], [], []
        window_loss_sum, window_loss_count = 0.0, 0

        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        val_metrics = evaluate(eval_model, val_loader, loss_fn, device, use_amp)
        model.train()  # mandatory -- evaluate() leaves the module in eval mode, and this run
                        # resumes training WITHIN the same epoch, unlike the original per-epoch-only loop.

        progress = global_step / total_training_steps
        row = {
            "global_step": global_step, "progress": round(progress, 4),
            "train_loss": train_window_loss, "train_macro_pr_auc": train_macro, "train_pos_pr_auc": train_pos_ap,
            "val_loss": val_metrics["val_loss"], "val_macro_pr_auc": val_metrics["macro_pr_auc"],
            "val_pos_pr_auc": val_metrics["pos_pr_auc"], "lr": scheduler.get_last_lr()[0],
        }
        log_rows.append(row)
        pl.DataFrame(log_rows).write_csv(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
        mlflow.log_metrics({k: v for k, v in row.items() if k != "progress"}, step=global_step)

        print("=" * 90)
        print(f"[CKPT step={global_step}/{total_training_steps} progress={progress:.0%}] "
              f"train_loss={train_window_loss:.5f} train_macro={train_macro:.5f} | "
              f"val_loss={row['val_loss']:.5f} val_macro={row['val_macro_pr_auc']:.5f}")
        print("=" * 90)
        _log_memory(f"checkpoint step={global_step}")

        if not np.isnan(row["val_macro_pr_auc"]) and row["val_macro_pr_auc"] > best_macro:
            best_macro = row["val_macro_pr_auc"]
            save_dir = os.path.join(OUTPUT_DIR, "best")
            save_checkpoint(model, tokenizer, save_dir)
            best_since_last_push = True
            print(f"[SAVE] new best macro_pr_auc={best_macro:.5f} -> {save_dir}")

        while push_milestones_remaining and progress >= push_milestones_remaining[0]:
            milestone = push_milestones_remaining.pop(0)
            if ENABLE_DATASET_PUSH and has_kaggle_creds and best_since_last_push:
                _push_best_as_dataset_version(
                    os.path.join(OUTPUT_DIR, "best"),
                    f"llm-continue-train step={global_step} milestone={milestone:.0%} "
                    f"macro_pr_auc={best_macro:.5f} val_fp={val_fingerprint}",
                    WORKING_DIR,
                )
                best_since_last_push = False
            else:
                print(f"[PUSH] milestone={milestone:.0%} skipped "
                      f"(enable={ENABLE_DATASET_PUSH}, creds={has_kaggle_creds}, "
                      f"new_best_since_last_push={best_since_last_push})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            input_ids, attention_mask, token_type_ids, batch_labels, batch_categories = batch

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    loss = loss_fn(logits, batch_labels_dev)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = loss_fn(logits, batch_labels_dev)

            loss_item = loss.item()
            scaler.scale(loss / grad_accum).backward()

            # .detach() is NOT optional: the train forward pass keeps its
            # autograd graph alive (unlike evaluate()'s no_grad accumulation),
            # so skipping this would keep CUDA activations pinned for the
            # whole checkpoint window and OOM late/confusingly.
            window_logits.append(logits.detach().float().cpu())
            window_labels_soft.append(batch_labels.detach().cpu())
            window_categories.append(batch_categories)
            window_loss_sum += loss_item
            window_loss_count += 1

            need_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))
            if need_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                pbar.set_postfix(loss=round(loss_item, 5), lr=round(scheduler.get_last_lr()[0], 8), step=global_step)

                is_epoch_end = (step + 1) == len(train_loader)
                if (global_step % checkpoint_interval == 0) or is_epoch_end:
                    run_checkpoint()

    # -----------------------------------------------------------------
    # Save "last" regardless of whether it was ever the best
    # -----------------------------------------------------------------
    last_dir = os.path.join(OUTPUT_DIR, "last")
    save_checkpoint(model, tokenizer, last_dir)

    print(f"[DONE] Best macro_pr_auc={best_macro:.5f}")
    print(f"[DONE] Step log: {os.path.join(OUTPUT_DIR, 'train_log_steps.csv')}")
    print(f"[DONE] Best model dir: {os.path.join(OUTPUT_DIR, 'best')}")
    print(f"[DONE] Last model dir: {last_dir}")

    mlflow.log_metric("best_macro_pr_auc", best_macro)
    mlflow.log_artifact(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
    mlflow.end_run()
    print(f"[DONE] mlflow run logged under {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
