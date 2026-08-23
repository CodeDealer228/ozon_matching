"""
Extracts BERTA+LoRA pooled pair embeddings AND the classifier head's own
score, for the canonical VAL split ONLY (73,131 pairs) -- using the TOP
checkpoint (maxamsecret/berta-tpu-lora-top-checkpoint: the exact adapter
shipped in submits/matching-berta_tpu_lora-submit, test score 0.47, the
best real test-stand score of any submission in this project so far).

VAL ONLY, deliberately, not train: this checkpoint was trained on the
canonical human TRAIN split + the full matches_llm.parquet -- the canonical
VAL split (73,131 pairs) is the one slice of human-labeled data it never
saw. Feeding a downstream CatBoost head only this val slice (then splitting
IT 80/20 for catboost's own train/val) keeps catboost's training data
disjoint from anything BERTA itself trained on -- the same no-leak
methodology already established in kaggle_kernels/berta-catboost-noleak,
just applied to a different (better) base checkpoint. See
kaggle_kernels/berta-catboost-leak's own module docstring for why the
leaking variant exists at all (to quantify inflation, not to submit).

Reuses the human_scope item cache from maxamsecret/berta-tokenize-llm-256
(mounted via kernel_sources) instead of re-tokenizing from scratch (unlike
the older kaggle_kernels/berta-embed-extract, which predates that cache and
tokenized items_human.parquet itself at ITEM_MAX_LENGTH=128) -- this
checkpoint trains/infers at PAIR_MAX_LENGTH=256, so extraction must use the
SAME length to produce embeddings that actually match how the model will be
used, not an arbitrary shorter one.

Feeds kaggle_kernels/berta-catboost-top-final: features are
[pooled_embedding (768) + classifier_score (1) + tfidf_score (from
kaggle_kernels/tfidf-category-index-fit, joined by id1/id2 downstream) +
category (native CatBoost categorical, not this kernel's concern)].
"""
import os
import sys
import gc
import json
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
ITEM_MAX_LENGTH = 256
MAX_CATEGORY_CHARS = 64
PAIR_MAX_LENGTH = 256   # matches the top checkpoint's own training/inference contract

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 3.0
EXTRACT_BATCH_SIZE = 32
EVAL_PROBE_CAP = 1024
PROBE_ROUND_TO = 4

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_top_embeddings")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _find_dir_with_file(filename, near=None):
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files and (near is None or near in root):
                return root
    return None


def _find_resume_adapter_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    return None


# =====================================================================
# ITEM CACHE (human_scope, reused as-is)
# =====================================================================

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


def load_human_scope_cache():
    scope_dir = _find_dir_with_file("cache_meta.json", near="human_scope")
    if scope_dir is None:
        raise RuntimeError("human_scope item cache not found under /kaggle/input -- expected "
                            "maxamsecret/berta-tokenize-llm-256 mounted (kernel_sources).")
    with open(os.path.join(scope_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    assert max_length == ITEM_MAX_LENGTH, f"{scope_dir}: max_length={max_length} != {ITEM_MAX_LENGTH}"
    assert meta["prefix"] == PREFIX, f"{scope_dir}: prefix={meta['prefix']!r} != {PREFIX!r}"

    ids = np.memmap(os.path.join(scope_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    input_ids = np.memmap(os.path.join(scope_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                           shape=(n_items, max_length))
    attn = np.memmap(os.path.join(scope_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                      shape=(n_items, max_length))
    item_lengths = np.asarray(attn).sum(axis=1).astype(np.int32)
    category_label_start = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_PATTERN)
    attr_label_start = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_PATTERN)
    id_to_row = {int(i): row for row, i in enumerate(ids.tolist())}
    print(f"[{_elapsed():7.1f}s][CACHE] {scope_dir}: {n_items:,} items loaded", flush=True)
    return input_ids, item_lengths, category_label_start, attr_label_start, id_to_row


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


# =====================================================================
# MODEL -- encoder + mean-pool + classifier, returns BOTH the pooled
# vector and the classifier's own scalar score (unlike the older
# berta-embed-extract, which stopped before the classifier).
# =====================================================================

class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.10):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, pooled):
        return self.linear(self.dropout(pooled))


class BertaEncoderAndHead(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).float()
            score = self.classifier(pooled).view(-1).float()
            return pooled, score


def load_model_from_checkpoint(device, merged_dir, resume_dir):
    print(f"[LOAD] base encoder {MODEL_NAME} + adapter from {resume_dir} ...")
    encoder_dir = os.path.join(merged_dir, "merged_encoder")
    base_model = AutoModel.from_pretrained(encoder_dir, dtype=torch.bfloat16)
    hidden_dim = base_model.config.hidden_size
    peft_model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=False)

    classifier_head = ClassifierHead(hidden_dim)
    raw_state = torch.load(os.path.join(resume_dir, "classifier.pt"), map_location="cpu")
    if "linear.weight" in raw_state:
        classifier_head.load_state_dict(raw_state)
    else:
        classifier_head.linear.load_state_dict(raw_state)

    model = BertaEncoderAndHead(peft_model, classifier_head).to(device)
    model.eval()
    return model, hidden_dim


def _probe_peak_gb(batch_size, n_gpu, pair_max_length, probe_model, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros((batch_size, pair_max_length), dtype=torch.long, device=device)
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)
    with torch.no_grad():
        _pooled, _score = probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask,
                                       token_type_ids=dummy_token_type_ids)
    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))
    del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids, _pooled, _score
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


@torch.no_grad()
def extract_val(model, loader, device):
    all_pooled, all_scores = [], []
    for batch in tqdm(loader, desc="Extract", dynamic_ncols=True):
        input_ids, attention_mask, token_type_ids = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = token_type_ids.to(device, non_blocking=True)
        pooled, score = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_pooled.append(pooled.cpu().numpy())
        all_scores.append(score.cpu().numpy())
    return np.concatenate(all_pooled, axis=0), np.concatenate(all_scores, axis=0)


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("BERTA TOP checkpoint: pooled embedding + classifier score extraction (VAL split only)")
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
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    print(f"[DATA] val={len(val_pd):,} (expect 73,131) -- train split NOT extracted, see module docstring")
    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    item_input_ids, item_lengths, category_label_start, attr_label_start, id_to_row = load_human_scope_cache()

    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_path)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
    gc.collect()

    merged_dir = _find_dir_with_file("merge_meta.json")
    if merged_dir is None:
        raise RuntimeError("Merged base not found under /kaggle/input -- expected "
                            "maxamsecret/berta-merge-lora mounted (kernel_sources).")
    resume_dir = _find_resume_adapter_dir()
    if resume_dir is None:
        raise RuntimeError("Top checkpoint adapter not found under /kaggle/input -- expected "
                            "maxamsecret/berta-tpu-lora-top-checkpoint mounted (dataset_sources).")
    model, hidden_dim = load_model_from_checkpoint(device, merged_dir, resume_dir)
    print(f"[MODEL] hidden_dim={hidden_dim}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1:
        print(f"[GPU] Using DataParallel on {n_gpu} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))

    extract_batch_size = run_vram_probe(model, device, n_gpu, PAIR_MAX_LENGTH, EXTRACT_BATCH_SIZE, EVAL_PROBE_CAP)

    t0 = time.perf_counter()
    loader = make_pair_loader(val_pd, id_to_row, item_input_ids, item_lengths,
                               category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                               PAIR_MAX_LENGTH, extract_batch_size)
    pooled, classifier_score = extract_val(model, loader, device)
    print(f"[EXTRACT] val: pooled={pooled.shape} score={classifier_score.shape} "
          f"in {time.perf_counter() - t0:.1f}s")

    np.save(os.path.join(OUTPUT_DIR, "val_embeddings.npy"), pooled.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "val_classifier_score.npy"), classifier_score.astype(np.float32))
    val_pd[["id1", "id2", "label", "category1"]].to_parquet(
        os.path.join(OUTPUT_DIR, "val_meta.parquet"), index=False
    )

    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME, "hidden_dim": hidden_dim, "pair_max_length": PAIR_MAX_LENGTH,
            "source_checkpoint": "maxamsecret/berta-tpu-lora-top-checkpoint "
                                  "(shipped in submits/matching-berta_tpu_lora-submit, test=0.47)",
            "split": "val only (73131 pairs, no-leak -- BERTA never trained on this slice)",
            "pooling": "mean (mask-weighted)",
            "classifier_score_is": "raw logit (sigmoid NOT applied) -- apply torch.sigmoid downstream if needed",
        }, f, indent=2)

    print(f"[DONE] embeddings+scores extracted -> {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
