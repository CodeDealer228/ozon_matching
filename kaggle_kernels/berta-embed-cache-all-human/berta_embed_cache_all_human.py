"""
PREPARED, NOT YET LAUNCHED (2026-08-20) -- see module docstring end for why.

Caches BERTA+LoRA pooled pair embeddings + classifier score for ALL of
`matches.parquet` (365,654 pairs -- both the canonical train AND val split,
not just val), using the `kaggle_kernels/berta-tpu-human-continue-384`
"best" checkpoint (epoch 1 of a planned 10, val_macro_pr_auc=0.7551 --
published as maxamsecret/berta-human-continue-384-best-checkpoint
specifically for this kernel).

===========================================================================
WHY ALL 365,654 PAIRS, NOT JUST THE 73,131-PAIR VAL SLICE
===========================================================================
kaggle_kernels/berta-catboost-top-final (CatBoost on the ORIGINAL top
checkpoint's embeddings) turned out to be data-starved in a way that only
became clear after the fact: that checkpoint trained on the human TRAIN
split (292,523 rows) + matches_llm.parquet, so only the 73,131-row VAL
split was actually leak-free for a downstream head -- and 58,504 rows (80%
of that, catboost's own train) for 771 features (768 raw embedding dims +
3) turned out to be a real too-many-features/too-few-rows overfit trap:
get_feature_importance() showed 96.11% of weight on the embedding dims
(classifier_score itself, the one feature independently known to be worth
0.47 on the real test, got only 1.9%) -- and the resulting submission
scored 0.457 on test, WORSE than the classifier alone, despite scoring
BETTER on catboost's own held-out val (0.7467 vs. 0.7087). PCA(32) +
stronger regularization (kaggle_kernels' README / submits/berta_catboost_pca32)
recovered most but not all of that gap (test 0.469, still marginally below
0.47) -- concentrating classifier_score's importance to 19.3% but not
eliminating the underlying data-scarcity problem.

The `berta-tpu-human-continue-384` checkpoint offers a way out: it is a
DIFFERENT set of weights from the one used for berta-catboost-top-final's
embeddings, resumed-and-further-trained specifically on the human-only
split -- but critically, for the purpose of CACHING embeddings to train a
FUTURE downstream head, what matters is not "did any BERTA-family model
ever see this pair" but "did THIS SPECIFIC set of weights, extracting
embeddings for a NEW downstream model, use these particular pairs as ITS
OWN training signal for the new head". A multi-layer head trained on top
of frozen embeddings from this checkpoint, using ALL 365,654 pairs, is ~5x
more data than the 73,131-pair no-leak slice available before -- enough to
plausibly support real capacity (a small MLP, not a 4000-tree GBDT on 768
raw dims) without repeating the same overfit failure. Head architecture
is still undecided (a later step); this kernel only produces the cached
features that any such head would consume, plus an intermediate-layer
readout that could itself feed CatBoost later (per the same reasoning
kaggle_kernels/berta-catboost-top-final already used, but on richer,
task-specific-but-not-yet-committed features instead of frozen 768-dim
pooled output alone).

===========================================================================
LARGE, SAFE BATCH SIZE -- "эффективный но безопасный" (efficient but safe)
===========================================================================
This is INFERENCE ONLY (no backward pass, no optimizer state) -- VRAM
headroom is much larger than any training run in this project, and this
job (365,654 pairs @ PAIR_MAX_LENGTH=384) is big enough that batch size
directly drives wall-clock cost. Reuses kaggle_kernels/berta-top-embed-
extract's proven VRAM auto-probe (doubling + binary search under a real
measured budget, not a guessed constant) rather than a fixed batch --
that kernel's own probe found real, substantial headroom at this model
size (batch=512 only used a fraction of an 11GB budget in an earlier,
similar extraction job) -- EVAL_PROBE_CAP raised to 4096 here (from that
kernel's 1024) specifically to let the probe search that documented
headroom instead of capping the search before finding it. Still bounded
by SAFETY_MARGIN_GB and the same binary-search-to-a-known-good-value
pattern used everywhere else in this project's GPU probes, so "large" does
not mean "unbounded/guessed".

===========================================================================
CRASH SAFETY -- incremental memmap writes, not one big save-at-the-end
===========================================================================
365,654 pairs is roughly 5x the previous 73,131-pair extraction job -- long
enough a run that losing ALL progress to a late crash would be a real,
avoidable waste. Output arrays (pooled embeddings, classifier scores) are
np.memmap files opened once and written incrementally, batch by batch, with
periodic .flush() calls -- not accumulated in a Python list and saved only
at the very end. A crash partway through leaves a real, readable partial
result on disk (the unwritten tail stays whatever memmap's initial fill
value is -- callers should check progress.json's own "batches_done" count,
not assume the whole file is valid) rather than nothing at all. This is a
deliberately SIMPLE safety measure (no resume-from-checkpoint logic, no
retry orchestration) -- matches the actual risk here (a single few-hour
forward-pass job, not a multi-session training run) without over-building
for a scenario this project's own conventions would call unlikely.

===========================================================================
NOT YET LAUNCHED
===========================================================================
Per direct instruction (2026-08-20): prepare this kernel, do not push it
yet. The user wants to see whether berta-tpu-human-continue-384 (the
checkpoint this kernel would cache embeddings from) actually earns its
keep on the real test stand before spending a GPU session caching 365,654
pairs' worth of embeddings from it -- if a later checkpoint or a different
architecture turns out better, this kernel's own CHECKPOINT_DATASET
constant is the one thing that would need to change before pushing.
"""
import os
import sys
import gc
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

from transformers import AutoTokenizer, AutoModel, set_seed

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
ITEM_MAX_LENGTH = 256   # human_scope cache's own length -- see PAIR_MAX_LENGTH note below
MAX_CATEGORY_CHARS = 64
PAIR_MAX_LENGTH = 384   # matches berta-tpu-human-continue-384's own training contract

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 3.0
EXTRACT_BATCH_SIZE = 64          # probe starting point (well above the 32
                                    # training used -- inference has no
                                    # gradient/optimizer memory to share with)
EVAL_PROBE_CAP = 4096            # raised from berta-top-embed-extract's 1024 --
                                    # see module docstring's "LARGE, SAFE
                                    # BATCH SIZE" section for why.
PROBE_ROUND_TO = 8

FLUSH_EVERY_BATCHES = 50   # periodic memmap .flush() -- see module
                             # docstring's crash-safety section.

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_all_human_embeddings")

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
# ITEM CACHE (human_scope, reused as-is -- same as berta-top-embed-extract
# and berta-tpu-human-continue-384)
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                       collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=False)


# =====================================================================
# MODEL -- encoder + mean-pool + classifier, returns both.
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
def extract_all_incremental(model, loader, device, n_total, hidden_dim, out_dir):
    """Writes pooled embeddings + classifier scores to memmap files
    INCREMENTALLY, batch by batch, flushing periodically -- see module
    docstring's crash-safety section. A `progress.json` sidecar records how
    many rows are actually valid (written) at any point, since a crash mid-
    run leaves the memmap's unwritten tail as whatever np.memmap's default
    fill is, not a clean "file ends here" marker."""
    pooled_path = os.path.join(out_dir, "all_human_embeddings.bin")
    score_path = os.path.join(out_dir, "all_human_classifier_score.bin")
    pooled_mm = np.memmap(pooled_path, dtype=np.float32, mode="w+", shape=(n_total, hidden_dim))
    score_mm = np.memmap(score_path, dtype=np.float32, mode="w+", shape=(n_total,))

    written = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Extract (all human)", dynamic_ncols=True)):
        input_ids, attention_mask, token_type_ids = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = token_type_ids.to(device, non_blocking=True)
        pooled, score = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        n = pooled.shape[0]
        pooled_mm[written:written + n] = pooled.float().cpu().numpy()
        score_mm[written:written + n] = score.float().cpu().numpy()
        written += n

        if (batch_idx + 1) % FLUSH_EVERY_BATCHES == 0:
            pooled_mm.flush()
            score_mm.flush()
            with open(os.path.join(out_dir, "progress.json"), "w") as f:
                json.dump({"rows_written": written, "n_total": n_total,
                           "elapsed_s": _elapsed()}, f, indent=2)

    pooled_mm.flush()
    score_mm.flush()
    with open(os.path.join(out_dir, "progress.json"), "w") as f:
        json.dump({"rows_written": written, "n_total": n_total, "elapsed_s": _elapsed(),
                   "complete": written == n_total}, f, indent=2)
    del pooled_mm, score_mm
    return written


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("BERTA human-continue-384 checkpoint: pooled embedding + classifier score, ALL human pairs")
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
    all_pd = matches_pl.to_pandas()
    del matches_pl
    gc.collect()
    print(f"[DATA] ALL human pairs: {len(all_pd):,} (expect 365,654 -- train+val combined, no split here)")

    item_input_ids, item_lengths, category_label_start, attr_label_start, id_to_row = load_human_scope_cache()

    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([all_pd["id1"], all_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_path)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    all_pd["category1"] = all_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
    gc.collect()

    merged_dir = _find_dir_with_file("merge_meta.json")
    if merged_dir is None:
        raise RuntimeError("Merged base not found under /kaggle/input -- expected "
                            "maxamsecret/berta-merge-lora mounted (kernel_sources).")
    resume_dir = _find_resume_adapter_dir()
    if resume_dir is None:
        raise RuntimeError("Checkpoint not found under /kaggle/input -- expected "
                            "maxamsecret/berta-human-continue-384-best-checkpoint mounted "
                            "(dataset_sources).")
    model, hidden_dim = load_model_from_checkpoint(device, merged_dir, resume_dir)
    print(f"[MODEL] hidden_dim={hidden_dim}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1:
        print(f"[GPU] Using DataParallel on {n_gpu} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))

    extract_batch_size = run_vram_probe(model, device, n_gpu, PAIR_MAX_LENGTH, EXTRACT_BATCH_SIZE, EVAL_PROBE_CAP)

    t0 = time.perf_counter()
    loader = make_pair_loader(all_pd, id_to_row, item_input_ids, item_lengths,
                               category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                               PAIR_MAX_LENGTH, extract_batch_size)
    n_written = extract_all_incremental(model, loader, device, len(all_pd), hidden_dim, OUTPUT_DIR)
    print(f"[EXTRACT] wrote {n_written:,}/{len(all_pd):,} rows in {time.perf_counter() - t0:.1f}s")

    all_pd[["id1", "id2", "label", "category1"]].to_parquet(
        os.path.join(OUTPUT_DIR, "all_human_meta.parquet"), index=False
    )
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME, "hidden_dim": hidden_dim, "pair_max_length": PAIR_MAX_LENGTH,
            "source_checkpoint": "maxamsecret/berta-human-continue-384-best-checkpoint "
                                  "(epoch 1 of kaggle_kernels/berta-tpu-human-continue-384, "
                                  "val_macro_pr_auc=0.7551)",
            "scope": "ALL matches.parquet pairs (365,654 -- train+val combined)",
            "pooling": "mean (mask-weighted)",
            "classifier_score_is": "raw logit (sigmoid NOT applied)",
            "n_written": n_written, "n_total": len(all_pd),
        }, f, indent=2)

    print(f"[DONE] embeddings+scores for ALL human pairs -> {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
