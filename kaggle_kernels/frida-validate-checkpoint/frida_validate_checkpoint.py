"""
Standalone FRIDA LoRA checkpoint validator -- computes real macro PR-AUC on
the canonical val split (73,131 pairs) for whatever adapter+classifier is
mounted via dataset_sources/kernel_sources, without needing to re-run or
wait on the training kernel itself.

Why this exists: kaggle_kernels/frida-lora-v2-epoch (and any future FRIDA
run) has a wall-clock "quiet checkpoint" fallback that skips the expensive
full-val eval once close to Kaggle's session cap, to guarantee SOME
resumable progress survives a hard kill -- exactly the scenario point 8 of
this project's own task list anticipated ("если валидация не успеет пройти
за каггл-сессию, то отдельным кернелом провалидировать"). This kernel is
that fallback: point it at any checkpoint's dataset_sources, get a real,
independently-computed macro PR-AUC back, no GPU training involved.

GENERIC BY DESIGN -- swap the checkpoint by changing dataset_sources/
kernel_sources, nothing in this file's own logic needs to change per
checkpoint (searches /kaggle/input by content, same convention as every
other resume-capable script in this project). PAIR_MAX_LENGTH is the one
constant that DOES need to match whatever the checkpoint was actually
trained at -- set explicitly below per checkpoint, not auto-detected (no
reliable way to infer it from the adapter files themselves).

First run (as pushed): validates maximmarkovbb/frida-lora-checkpoint-v1-best
(the FRIDA continue-2ep run's own best, val_macro_pr_auc=0.69377 as reported
by that run itself, at PAIR_MAX_LENGTH=128) -- both to produce a real,
independently-verified number for a checkpoint that already has one (a
sanity check on this validator itself) and because it's the one checkpoint
currently guaranteed to be available. Re-point dataset_sources at
frida-lora-v2-epoch's own checkpoint once that run actually finishes/gets
cancelled with recoverable output.
"""
import os
import sys
import gc
import json
import hashlib
import subprocess
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "peft"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoTokenizer, T5EncoderModel, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel


# =====================================================================
# CONFIG -- PAIR_MAX_LENGTH must match the checkpoint being validated.
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 128   # v1-best's own training length -- change this when
                          # re-pointing at a different checkpoint.
MAX_CATEGORY_CHARS = 64

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

EVAL_BATCH_SIZE = 256   # forward-only, conservative fixed constant (no
                          # auto-probe -- keeps this a lightweight, quick-
                          # turnaround kernel; raise if a run shows real
                          # headroom).

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_validation")

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


def _find_item_cache_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "input_ids.bin" in files:
                return root
    return None


def _find_checkpoint_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    return None


# =====================================================================
# ITEM CACHE + PAIR ASSEMBLY (unchanged from frida-lora-v2-epoch)
# =====================================================================

def _find_token_pattern_starts(arr, tok_a, tok_b):
    mask = (arr[:, :-1] == tok_a) & (arr[:, 1:] == tok_b)
    rows, cols = np.where(mask)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def load_item_cache(cache_dir):
    with open(os.path.join(cache_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    assert max_length == ITEM_MAX_LENGTH, f"{cache_dir}: max_length={max_length} != {ITEM_MAX_LENGTH}"

    ids = np.memmap(os.path.join(cache_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    input_ids = np.memmap(os.path.join(cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                           shape=(n_items, max_length))
    attn = np.memmap(os.path.join(cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                      shape=(n_items, max_length))
    lengths = np.asarray(attn).sum(axis=1).astype(np.int32)

    arr = np.asarray(input_ids)
    category_start = _find_token_pattern_starts(arr, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _find_token_pattern_starts(arr, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    both_found = (category_start >= 0) & (attr_start >= 0)
    category_span = np.where(both_found, attr_start - category_start, 0).astype(np.int32)
    compact_lengths = lengths - category_span

    id_to_row = {int(item_id): i for i, item_id in enumerate(np.asarray(ids))}
    print(f"[CACHE] {cache_dir}: {n_items:,} items loaded", flush=True)
    return input_ids, lengths, category_start, attr_start, compact_lengths, id_to_row


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
    def __init__(self, row1, row2, labels, categories):
        self.row1 = row1
        self.row2 = row2
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, idx):
        return self.row1[idx], self.row2[idx], self.labels[idx], self.categories[idx]


def make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start, pad_id, pair_max_length):
    def collate_fn(batch):
        row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            compact1 = build_compact_ids(item_input_ids, row1[i], int(item_lengths[row1[i]]),
                                          int(category_label_start[row1[i]]), int(attr_label_start[row1[i]]))
            compact2 = build_compact_ids(item_input_ids, row2[i], int(item_lengths[row2[i]]),
                                          int(category_label_start[row2[i]]), int(attr_label_start[row2[i]]))
            cl1, cl2 = len(compact1), len(compact2)
            if cl1 + cl2 <= pair_max_length:
                keep1, keep2 = cl1, cl2
            else:
                keep1, keep2 = allocate_pair_budget(cl1, cl2, pair_max_length)
            combined = np.concatenate((compact1[:keep1], compact2[:keep2]))
            n = len(combined)
            input_ids[i, :n] = combined
            attention_mask[i, :n] = 1

        return (
            torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
            torch.tensor(labels, dtype=torch.float32), np.asarray(categories, dtype=np.int64),
        )
    return collate_fn


class FridaCrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls).view(-1).float()


def compute_metrics(y_true, y_prob, categories):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    categories = np.asarray(categories)
    per_category_aps = []
    for cat in np.unique(categories):
        mask = categories == cat
        y_t, y_p = y_true[mask], y_prob[mask]
        if len(np.unique(y_t)) < 2:
            continue
        per_category_aps.append(average_precision_score(y_t, y_p))
    macro = float(np.mean(per_category_aps)) if per_category_aps else float("nan")
    return macro, len(per_category_aps)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print(f"FRIDA checkpoint validator -- PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}")
    print("=" * 90)
    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_id = tokenizer.pad_token_id

    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    matches_pl = pl.read_parquet(matches_path)
    matches_pl = (
        matches_pl.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([pl.col("target").fill_null(0).cast(pl.Float32),
                       (pl.col("target") >= 0.5).cast(pl.Int8).alias("label")])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()
    del matches_pl
    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)
    _train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels_all)
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[DATA] val={len(val_pd):,} val_fingerprint={val_fingerprint} (expect 73,131 / efddb3def997f87b)")
    del matches_pd, labels_all, indices, _train_idx, val_idx
    gc.collect()

    cache_dir = _find_item_cache_dir()
    if cache_dir is None:
        raise RuntimeError("No item-token cache found under /kaggle/input -- expected a FRIDA item "
                            "cache (frida-tokenized-items-human or frida-tokenize-items-v2-cache) "
                            "mounted via dataset_sources/kernel_sources.")
    item_input_ids, item_lengths, category_label_start, attr_label_start, compact_lengths, id_to_row = \
        load_item_cache(cache_dir)

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_human_path).join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
    gc.collect()

    checkpoint_dir = _find_checkpoint_dir()
    if checkpoint_dir is None:
        raise RuntimeError("No adapter checkpoint found under /kaggle/input -- expected "
                            "adapter_config.json mounted via dataset_sources.")
    print(f"[MODEL] loading base {MODEL_NAME} + adapter from {checkpoint_dir} ...")
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.d_model
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)

    classifier_head = nn.Linear(hidden_dim, 1)
    classifier_head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "classifier.pt"), map_location="cpu"))

    model = FridaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    model.eval()

    row1 = val_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = val_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    labels = val_pd["label"].to_numpy().astype(np.float32)
    categories = val_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)
    dataset = PairIndexDataset(row1, row2, labels, categories)
    collate_fn = make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start,
                                       pad_id, PAIR_MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2,
                         collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    all_logits, all_labels, all_categories = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", dynamic_ncols=True):
            input_ids, attention_mask, batch_labels, batch_categories = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.float().cpu())
            all_labels.append(batch_labels)
            all_categories.append(batch_categories)
    print(f"[EVAL] forward pass done in {time.perf_counter() - t0:.1f}s")

    logits = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    categories_arr = np.concatenate(all_categories, axis=0)
    probs = torch.sigmoid(logits).numpy()
    y_true = (labels_t.numpy() >= 0.5).astype(int)
    macro_pr_auc, n_categories = compute_metrics(y_true, probs, categories_arr)

    result = {
        "checkpoint_dir": checkpoint_dir, "pair_max_length": PAIR_MAX_LENGTH,
        "val_fingerprint": val_fingerprint, "n_val": len(val_pd),
        "n_categories_evaluated": n_categories, "val_macro_pr_auc": macro_pr_auc,
    }
    print(f"[RESULT] {json.dumps(result, indent=2)}")
    with open(os.path.join(OUTPUT_DIR, "validation_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
