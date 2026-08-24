# %% [code]
"""
CPU-only kernel: assembles ONE submission zip from the exact `last`
checkpoint of kirpich682/frida-full-sft-continue-e-llmfull-w2
(`experiment=continue_e_llmfull_w2_2ep`).

Same proven pipeline as kaggle_kernels/frida-full-sft-submit-build (which
produced the current project-best submission, test=0.481) -- full fine-tune,
no LoRA/PEFT, bf16 save_pretrained -- with one clean kernel_source and one
exact-label target instead of the older builder's accumulated inputs.

WHY CONTENT-BASED SOURCE SELECTION, NOT A HARDCODED PATH PER EXPERIMENT
===========================================================================
The source kernel is attached via kernel_sources (not dataset_sources) --
its mount point under /kaggle/input may drift, so
dataset_sources in this project (see every other kernel's own
_find_dataset_file/_find_resume_source docstrings), and kernel_sources has
no reason to be more stable. So _find_all_checkpoints walks /kaggle/input
once, collects every directory holding {encoder.pt, head.pt,
train_state.json}, and groups them by train_state.json's own "experiment"
field -- the SAME field frida_full_sft_continue.py's own _find_resume_source
uses to tell "my own prior session" from "a different experiment". This is
robust to whatever exact directory layout each kernel's own output happens
to have.

WHICH CHECKPOINT: exact directory label `last`, selected by both the
train_state.json experiment field and basename. It must not silently drift
to `best` or checkpoint_epochNN. The source state is epoch 1, global step
28,277, epoch_complete=false, and has no human/LLM validation metric.

Text/pair contract, PAIR_MAX_LENGTH=188/BATCH_SIZE_FIXED=5248/
ITEM_MAX_LENGTH=256, and the whole run.py/utils.py inference pipeline are
byte-identical to frida-full-sft-submit-build (all 5 continue experiments
share frida_full_sft_continue.py's own base, which is itself a fork of
frida_full_sft.py with the SAME HEAD_HIDDEN/text contract -- verified
against the base file directly, not assumed from family resemblance).
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "safetensors"])

import torch
from transformers import AutoTokenizer, T5EncoderModel

MODEL_NAME = "ai-forever/FRIDA"

# experiment name (train_state.json's own "experiment" field) -> output zip
# stem -> pair_max_length. best-checkpoint picks (_pick_checkpoint), served
# at 188 -- the new standard (see PAIR_MAX_LENGTH's own comment in UTILS_PY).
TARGET_EXPERIMENTS = []

# experiment name -> exact checkpoint label (_pick_checkpoint_by_label, NOT
# best/last) -> output zip stem -> pair_max_length. Added 2026-08-23 per
# direct instruction: a-baseline's own checkpoint_epoch03 (="best") already
# scored 0.478 on the real test bench at PAIR_MAX_LENGTH=180 -- these two
# build the SAME experiment's epoch01/epoch02 checkpoints at the SAME 180
# length, so epoch is the only thing that varies against that already-real
# data point (not conflated with the 188 length change TARGET_EXPERIMENTS
# above now uses by default).
EXTRA_TARGETS = [
    ("continue_e_llmfull_w2_2ep", "last",
     "matching-frida_continue_e_last_188-submit", 188),
]

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def _log(msg):
    print(f"[BUILD] {msg}", flush=True)


def _find_all_checkpoints():
    """One walk of /kaggle/input, collecting every {encoder.pt, head.pt,
    train_state.json} directory with its parsed train_state.json --
    content-based, not path-based (see module docstring)."""
    found = []
    if not os.path.isdir("/kaggle/input"):
        return found
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"encoder.pt", "head.pt", "train_state.json"} <= set(files):
            with open(os.path.join(root, "train_state.json"), encoding="utf-8") as f:
                state = json.load(f)
            found.append((root, state))
    return found


def _pick_checkpoint(all_checkpoints, experiment_name):
    """best (highest real val_macro_pr_auc) over last (highest global_step,
    unvalidated) -- see module docstring."""
    candidates = [(root, s) for root, s in all_checkpoints if s.get("experiment") == experiment_name]
    if not candidates:
        raise RuntimeError(f"[BUILD] no checkpoint anywhere under /kaggle/input has "
                           f"experiment={experiment_name!r} -- is its kernel_source attached "
                           f"and has it saved at least one checkpoint (quiet or validated)?")
    validated = [(root, s) for root, s in candidates if s.get("val_macro_pr_auc") is not None]
    if validated:
        root, state = max(validated, key=lambda rs: rs[1]["val_macro_pr_auc"])
        kind = "VALIDATED (best)"
    else:
        root, state = max(candidates, key=lambda rs: rs[1].get("global_step", 0))
        kind = "UNVALIDATED (quiet safety-net only -- no real eval has run yet for this experiment)"
    _log(f"  picked [{kind}] {root}")
    _log(f"  train_state: epoch={state.get('epoch')} epoch_complete={state.get('epoch_complete')} "
        f"global_step={state.get('global_step')} val_macro_pr_auc={state.get('val_macro_pr_auc')} "
        f"llm_val_macro_pr_auc={state.get('llm_val_macro_pr_auc')}")
    return root, state


def _pick_checkpoint_by_label(all_checkpoints, experiment_name, label):
    """Exact-label pick (e.g. 'checkpoint_epoch01'), for EXTRA_TARGETS --
    unlike _pick_checkpoint, this does NOT choose best-vs-last; it demands
    the one specific checkpoint directory whose own basename matches
    `label`, so building "epoch 1's own submission" can't silently drift
    to some other checkpoint of the same experiment."""
    candidates = [(root, s) for root, s in all_checkpoints
                 if s.get("experiment") == experiment_name and os.path.basename(root) == label]
    if not candidates:
        raise RuntimeError(f"[BUILD] no checkpoint under /kaggle/input has experiment="
                           f"{experiment_name!r} AND label={label!r} -- is its kernel_source "
                           f"attached and did that exact checkpoint actually get saved?")
    if len(candidates) > 1:
        raise RuntimeError(f"[BUILD] {len(candidates)} checkpoints match experiment="
                           f"{experiment_name!r} AND label={label!r} -- ambiguous, refusing to "
                           f"guess which one: {[r for r, _s in candidates]}")
    root, state = candidates[0]
    _log(f"  picked [EXACT LABEL {label!r}] {root}")
    _log(f"  train_state: epoch={state.get('epoch')} epoch_complete={state.get('epoch_complete')} "
        f"global_step={state.get('global_step')} val_macro_pr_auc={state.get('val_macro_pr_auc')} "
        f"llm_val_macro_pr_auc={state.get('llm_val_macro_pr_auc')}")
    return root, state


# =====================================================================
# run.py / src/utils.py -- byte-identical to frida-full-sft-submit-build's
# own (see that kernel for the full history/reasoning behind every choice
# here -- not re-derived, this is the same proven pipeline that produced
# the current project-best submission, test=0.481).
# =====================================================================

RUN_PY = '''import argparse
import os

from src.utils import predict_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "frida-full-sft")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        base_model_path=BASE_MODEL_PATH,
        classifier_path=CLASSIFIER_PATH,
        output_csv_path=args.output_path,
        # batch_size intentionally omitted -- uses predict_pipeline's own
        # BATCH_SIZE_FIXED default.
    )


if __name__ == "__main__":
    main()
'''

UTILS_PY = '''import os

# Hardware-aware env tuning, byte-identical to every other FRIDA submission
# in this project (submits/frida/*/src/utils.py) -- 20 CPU cores / 200GB
# RAM / 1x H100 80GB at inference time (CLAUDE.md).
os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")
# Added after real evidence in a prior FRIDA submission (see
# submits/frida/tpu_human_llm_best_180_224_5120/src/utils.py's own
# comment): a real OOM at a large fixed batch reported "reserved by
# PyTorch but unallocated" in the multi-GiB range -- this changes only how
# the CUDA caching allocator manages memory internally, never what gets
# computed, so it is zero quality risk to keep here unconditionally.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, logging as hf_logging

# ---------------------------------------------------------------------
# Text/pair contract -- byte-matches kaggle_kernels/frida-tokenize-items-cpu
# (built frida-tokenize-items-v3, which this checkpoint's own item cache --
# and therefore its own training -- used). NOT the older 140/64/640 @
# ITEM_MAX_LENGTH=224 formula some earlier FRIDA LoRA submissions in this
# repo use -- a different (v2) cache generation this checkpoint never saw.
# ---------------------------------------------------------------------
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 529
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 30  # 731
PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256
# These checkpoints trained at PAIR_MAX_LENGTH=192 (frida_full_sft_continue.py,
# same as frida_full_sft.py). Serve-shorter-than-trained, per direct
# instruction. STANDARDIZED TO 188 (2026-08-23): frida_continue_c_llm3m
# served at 188/this same BATCH_SIZE_FIXED scored **0.487 on the real test
# bench -- new project best** (beat the 180-length a-baseline build's own
# 0.478), AND fits the time budget (~16-19 min for the frida+head forward
# pass at this length/batch, confirmed on the real stand) -- so 188 is now
# the default for every build this kernel produces, not just an experiment.
# Substituted per-build via the literal token below (plain str.replace, not
# .format -- this whole string is full of other unrelated {} that .format
# would choke on), so a single kernel run can still serve a DIFFERENT
# length for a specific target (e.g. the a-baseline epoch01/02 zips below,
# built at 180 on purpose to stay comparable to the already-tested 0.478
# data point -- see EXTRA_TARGETS).
PAIR_MAX_LENGTH = __PAIR_MAX_LENGTH__
BATCH_SIZE_FIXED = 5248

CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes" (+ colon=30 immediately after)
LABEL_COLON_TOKEN = 30

# v2 attribute-cleanup policy, byte-identical to the cache-build kernel:
# demote (to the end of the serialized JSON, before the MAX_ATTR_CHARS
# slice) any value that is long, contains a link, or mentions "oem".
LONG_VALUE_CHARS = 200
LINK_MARKER = "https"
OEM_MARKER = "oem"


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


def reorder_attributes(attr_str):
    if not attr_str:
        return attr_str
    try:
        d = json.loads(attr_str)
    except Exception:
        return attr_str
    if not isinstance(d, dict) or not d:
        return attr_str
    ranked = [(_demotion_rank(k, v), k, v) for k, v in d.items()]
    ranked.sort(key=lambda t: t[0])
    return json.dumps({k: v for _r, k, v in ranked}, ensure_ascii=False)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_bf16_capability(device):
    """Diagnostic only -- does not change dtype. See
    submits/frida/tpu_human_llm_best_180_224_5120/src/utils.py's own
    version of this function for the full CLAUDE.md/Giga3B-probe reasoning
    (T4-class GPUs report is_bf16_supported()=True but have no native bf16
    tensor cores; the real gate is compute capability >= 8)."""
    if device.type != "cuda":
        return
    major, _minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if major >= 8:
        print(f"[PRECISION] {name}: compute capability {major}.x -- native bf16 tensor cores, "
              f"bf16 is the fast path here", flush=True)
    else:
        print(f"[PRECISION] WARNING: {name}: compute capability {major}.x has NO native bf16 "
              f"tensor cores -- bf16 runs an EMULATED, slow path here. Expected only if this is "
              f"a non-H100 dry run, not the real judging stand.", flush=True)


def _reorder_and_slice(attr):
    return reorder_attributes(attr)[:MAX_ATTR_CHARS]


def _load_needed_item_texts(items_path, needed_ids):
    """Byte-matches kaggle_kernels/frida-tokenize-items-cpu's own text
    assembly. Single-threaded reorder, deliberately -- see
    submits/frida/tpu_human_llm_best_180_224_5120/src/utils.py's own
    comment for the real multiprocessing.Pool(os.cpu_count()) crash
    (os.cpu_count() reported 240 on the real judging container; forking
    240 processes hit BlockingIOError from os.fork() itself) this project
    already found and removed -- not reintroduced here."""
    df = pd.read_parquet(
        items_path, engine="pyarrow", columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", list(set(needed_ids)))],
    )
    names = df["name"].fillna("").astype(str).tolist()
    categories = df["category"].fillna("").astype(str).tolist()
    attributes = df["attributes"].fillna("").astype(str).tolist()
    attrs_ordered = [_reorder_and_slice(a) for a in attributes]

    texts = []
    for name, category, attr_ordered in zip(names, categories, attrs_ordered):
        name = name[:MAX_NAME_CHARS]
        category = category[:MAX_CATEGORY_CHARS]
        texts.append(f"Name: {name} Category: {category} Attributes: {attr_ordered}"[:MAX_ITEM_CHARS])
    return dict(zip(df["id"].tolist(), texts))


def _find_token_pattern_starts(arr, tok_a, tok_b):
    mask = (arr[:, :-1] == tok_a) & (arr[:, 1:] == tok_b)
    rows, cols = np.where(mask)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


TOKENIZE_CHUNK_SIZE = 14_000


def _tokenize_items(tokenizer, id_to_text):
    """Tokenizes every unique item ONCE (paraphrase: prefix, WITH special
    tokens, matching the training cache), then locates the
    Category:/Attributes: label span per item for excision."""
    ids = list(id_to_text.keys())
    texts = [PREFIX + id_to_text[i] for i in ids]

    input_ids_chunks = []
    lengths_chunks = []
    for start in range(0, len(texts), TOKENIZE_CHUNK_SIZE):
        chunk = texts[start:start + TOKENIZE_CHUNK_SIZE]
        enc = tokenizer(chunk, add_special_tokens=True, padding="max_length", truncation=True,
                         max_length=ITEM_MAX_LENGTH, return_tensors="np")
        input_ids_chunks.append(enc["input_ids"].astype(np.int32))
        lengths_chunks.append(enc["attention_mask"].sum(axis=1).astype(np.int32))
    input_ids = np.concatenate(input_ids_chunks, axis=0)
    lengths = np.concatenate(lengths_chunks, axis=0)
    del input_ids_chunks, lengths_chunks
    gc.collect()

    category_start = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)

    id_to_item = {}
    for row, item_id in enumerate(ids):
        id_to_item[item_id] = (input_ids[row], int(lengths[row]), int(category_start[row]), int(attr_start[row]))
    return id_to_item


def allocate_pair_budget(len1, len2, budget):
    """Fair-share truncation over COMPACT (category-excised) lengths: a
    side under half-budget is never cut below its real length; slack from
    a short side goes to the other; if both exceed half, split evenly."""
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def _build_compact_ids(input_ids, item_length, category_start, attr_start):
    if category_start < 0 or attr_start < 0:
        return input_ids[:item_length]
    head = input_ids[:category_start]
    tail = input_ids[attr_start:item_length]
    return np.concatenate((head, tail))


def _build_pairs(match_df, id_to_item, pad_id):
    """Excise Category from each side, fair-share truncate to
    PAIR_MAX_LENGTH, concatenate -- no CLS/SEP insertion (T5 has no
    segment concept; each side's own bos/eos from tokenization already
    marks its boundary)."""
    n = len(match_df)
    input_ids = np.full((n, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    attention_mask = np.zeros((n, PAIR_MAX_LENGTH), dtype=np.int64)

    id1_list = match_df["id1"].tolist()
    id2_list = match_df["id2"].tolist()
    n_missing = 0
    for i in range(n):
        item1 = id_to_item.get(id1_list[i])
        item2 = id_to_item.get(id2_list[i])
        if item1 is None or item2 is None:
            n_missing += 1
            continue
        compact1 = _build_compact_ids(*item1)
        compact2 = _build_compact_ids(*item2)
        cl1, cl2 = len(compact1), len(compact2)
        if cl1 + cl2 <= PAIR_MAX_LENGTH:
            keep1, keep2 = cl1, cl2
        else:
            keep1, keep2 = allocate_pair_budget(cl1, cl2, PAIR_MAX_LENGTH)
        combined = np.concatenate((compact1[:keep1], compact2[:keep2]))
        m = len(combined)
        input_ids[i, :m] = combined
        attention_mask[i, :m] = 1

    if n_missing:
        print(f"[WARN] {n_missing} pairs referenced an item id missing from items.parquet "
              f"(left as all-pad -- attention_mask all-zero for those rows)", flush=True)
    return input_ids, attention_mask


class FridaFullSftClassifier(nn.Module):
    """CLS (position-0) pooling -- matches frida_full_sft_continue.py's own
    class exactly (FRIDA's model card recommendation)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).view(-1)


class FullSftHead(nn.Module):
    """[1024, 512, 128, 1] -- verbatim copy of frida_full_sft_continue.py's
    own FullSftHead class. State_dict key shape (blocks.N.0/1/2.*, out.*)
    must match exactly for classifier.pt to load."""

    def __init__(self, hidden_dim, layer_sizes=(1024, 512, 128), dropout=0.2):
        super().__init__()
        blocks = []
        d = hidden_dim
        for h in layer_sizes:
            blocks.append(nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


def _load_tokenizer(base_model_path):
    return AutoTokenizer.from_pretrained(base_model_path)


def _load_model_weights(base_model_path, classifier_path, device):
    """No PEFT/LoRA here -- this checkpoint is a genuine full fine-tune.
    models/frida-full-sft/ IS the fine-tuned weights directly (built via
    model.save_pretrained(..., safe_serialization=True) on the actual
    checkpoint, not ai-forever/FRIDA's own pretrained weights) -- a plain
    from_pretrained() call on it loads OUR fine-tuned weights, exactly the
    same well-tested loading path every prior FRIDA submission in this
    project already uses for its own base model, just with no adapter to
    layer on top."""
    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model = T5EncoderModel.from_pretrained(base_model_path, attn_implementation="eager",
                                                 torch_dtype=torch.bfloat16)
    except (TypeError, ValueError):
        model = T5EncoderModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    hf_logging.set_verbosity(default_verbosity)
    hidden_dim = model.config.d_model

    classifier_head = FullSftHead(hidden_dim, layer_sizes=(1024, 512, 128), dropout=0.2)
    classifier_head.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    # Match the encoder's dtype -- same reasoning every prior FRIDA
    # submission's own head-dtype cast uses: CUDA inference runs under
    # torch.autocast anyway, this mainly keeps a CPU dry-run consistent.
    classifier_head = classifier_head.to(dtype=torch.bfloat16)

    full_model = FridaFullSftClassifier(model, classifier_head)
    full_model.to(device)
    full_model.eval()
    return full_model


# Progress logging every ~20% of rows, not per-batch tqdm -- same
# reasoning as every prior FRIDA submission: a run that dies on time (not
# an exception) leaves no other record of how far it got.
PROGRESS_LOG_FRACTION = 0.20


def _run_inference(model, input_ids_all, attention_mask_all, device, batch_size):
    """FIXED-SHAPE batching -- every GPU-facing forward call gets EXACTLY
    batch_size rows, including a repeated-last-row-padded final batch, per
    this project's own real OOM history with ragged final batches (see
    submits/frida/tpu_human_llm_best_180_224_5120/src/utils.py's own
    comment for the seq=192/batch=6912-vs-8192 anomaly this fixes)."""
    use_amp = device.type == "cuda"
    n = input_ids_all.shape[0]
    probs = np.empty(n, dtype=np.float32)
    t0 = time.time()
    next_log_at = max(1, int(n * PROGRESS_LOG_FRACTION))

    def run_one_batch(ids_np, mask_np, n_valid):
        input_ids = torch.from_numpy(ids_np).to(device, non_blocking=True)
        attention_mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(logits.float()).cpu().numpy()[:n_valid]

    def maybe_log_progress(done):
        nonlocal next_log_at
        if done < next_log_at:
            return
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        print(f"[7/7]   progress: {done:,}/{n:,} ({100 * done / n:.0f}%) "
              f"in {elapsed:.1f}s ({rate:.1f} pairs/s, ETA {eta:.0f}s)", flush=True)
        while next_log_at <= done:
            next_log_at += max(1, int(n * PROGRESS_LOG_FRACTION))

    with torch.inference_mode():
        n_full = (n // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            probs[i:i + batch_size] = run_one_batch(
                input_ids_all[i:i + batch_size], attention_mask_all[i:i + batch_size], batch_size)
            maybe_log_progress(i + batch_size)

        if n_full < n:
            tail = n - n_full
            pad_ids = np.tile(input_ids_all[n - 1:n], (batch_size, 1))
            pad_mask = np.tile(attention_mask_all[n - 1:n], (batch_size, 1))
            pad_ids[:tail] = input_ids_all[n_full:n]
            pad_mask[:tail] = attention_mask_all[n_full:n]
            probs[n_full:n] = run_one_batch(pad_ids, pad_mask, tail)
            maybe_log_progress(n)

    print(f"[7/7]   inference done: {n:,}/{n:,} (100%) in {time.time() - t0:.1f}s", flush=True)
    return probs


def predict_pipeline(items_path, matches_path, base_model_path, classifier_path,
                      output_csv_path, batch_size=BATCH_SIZE_FIXED):
    """Model weights load in a BACKGROUND THREAD, overlapping with item
    -text loading + tokenization + pair-building on the main thread -- same
    scheduling optimization every prior FRIDA submission in this project
    uses (nothing before inference actually needs the model weights, only
    the tokenizer, which loads separately and fast)."""
    t0 = time.time()
    print("=== Starting FRIDA full-SFT (no LoRA) prediction pipeline ===", flush=True)

    print("[1/7] Loading tokenizer (fast) + starting model-weight load in the background ...",
          flush=True)
    device = _get_device()
    _log_bf16_capability(device)
    tokenizer = _load_tokenizer(base_model_path)
    model_holder = {}
    model_error = {}

    def _load_model_thread():
        try:
            model_holder["model"] = _load_model_weights(base_model_path, classifier_path, device)
        except Exception as e:  # noqa: BLE001 -- re-raised on the main thread below
            model_error["exc"] = e

    model_thread = threading.Thread(target=_load_model_thread, name="model-weights-loader")
    model_thread.start()
    print(f"[1/7] tokenizer ready, model loading in background ({time.time() - t0:.1f}s)", flush=True)

    print("[2/7] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[2/7] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} "
          f"({time.time() - t0:.1f}s)", flush=True)

    print("[3/7] Loading needed item texts ...", flush=True)
    id_to_text = _load_needed_item_texts(items_path, needed_ids)
    print(f"[3/7] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print(f"[4/7] Tokenizing items (chunks of {TOKENIZE_CHUNK_SIZE:,}) "
          "+ locating Category:/Attributes: labels ...", flush=True)
    id_to_item = _tokenize_items(tokenizer, id_to_text)
    print(f"[4/7] done ({time.time() - t0:.1f}s)", flush=True)

    print(f"[5/7] Building pairs (PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}) ...", flush=True)
    input_ids_all, attention_mask_all = _build_pairs(match_df, id_to_item, tokenizer.pad_token_id)
    print(f"[5/7] done ({time.time() - t0:.1f}s)", flush=True)

    print("[6/7] Waiting for background model-weight load (if not already done) ...", flush=True)
    model_thread.join()
    if "exc" in model_error:
        raise model_error["exc"]
    model = model_holder["model"]
    print(f"[6/7] model ready ({time.time() - t0:.1f}s)", flush=True)

    print(f"[7/7] Running inference (batch_size={batch_size}) ...", flush=True)
    probs = _run_inference(model, input_ids_all, attention_mask_all, device, batch_size=batch_size)
    print(f"[7/7] done ({time.time() - t0:.1f}s)", flush=True)

    results_df = pd.DataFrame({
        "id1": match_df["id1"].to_numpy(),
        "id2": match_df["id2"].to_numpy(),
        "predict": probs,
    })

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - t0
    print(f"=== Done in {elapsed:.1f}s ({len(match_df) / max(elapsed, 1e-9):.1f} pairs/sec) ===", flush=True)
    return results_df
'''

METADATA_JSON = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}


def _build_one_submission(ckpt_dir, state, zip_stem, pair_max_length):
    build_dir = os.path.join(WORKING_DIR, f"_submission_build_{zip_stem}")
    zip_path = os.path.join(WORKING_DIR, f"{zip_stem}.zip")
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)
    models_dir = os.path.join(build_dir, "models", "frida-full-sft")
    os.makedirs(models_dir, exist_ok=True)

    # ---- 1. base architecture + tokenizer, fresh from the Hub each time
    # (cheap, ~1.65GB bf16 -- simpler and safer than trying to cache/reuse
    # a T5EncoderModel instance across 3 builds with different weights) ----
    _log(f"  downloading {MODEL_NAME} tokenizer + config ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    hidden_dim = model.config.d_model

    # ---- 2. overwrite with the ACTUAL fine-tuned weights ----
    encoder_sd_path = os.path.join(ckpt_dir, "encoder.pt")
    _log(f"  loading fine-tuned encoder weights from {encoder_sd_path} ...")
    encoder_sd = torch.load(encoder_sd_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"[BUILD] encoder state_dict mismatch for {zip_stem} -- "
                           f"missing={missing} unexpected={unexpected}. Refusing to ship a "
                           f"submission built on a silently-wrong weight load.")
    del encoder_sd
    _log("  encoder weights loaded and verified (strict key match)")

    # ---- 3. bf16 cast + save via the model's OWN save_pretrained -- NOT a
    # raw safetensors.save_file on the state_dict, which would hit T5's
    # real shared.weight/encoder.embed_tokens.weight tied-parameter
    # aliasing. ----
    model = model.to(dtype=torch.bfloat16)
    _log(f"  saving bf16 model + config to {models_dir} ...")
    model.save_pretrained(models_dir, safe_serialization=True)
    tokenizer.save_pretrained(models_dir)
    del model
    _log("  model.safetensors + config.json + tokenizer files written")

    # ---- 4. classifier head, same bf16 cast, plain torch.save (matches
    # every prior FRIDA submission's own classifier.pt convention) ----
    head_sd_path = os.path.join(ckpt_dir, "head.pt")
    _log(f"  loading classifier head weights from {head_sd_path} ...")
    head_sd = torch.load(head_sd_path, map_location="cpu")
    head_sd_bf16 = {k: v.to(dtype=torch.bfloat16) for k, v in head_sd.items()}
    classifier_path = os.path.join(build_dir, "classifier.pt")
    torch.save(head_sd_bf16, classifier_path)
    _log(f"  classifier.pt written ({os.path.getsize(classifier_path):,} bytes)")

    # ---- 5. run.py / src/utils.py / metadata.json ----
    with open(os.path.join(build_dir, "run.py"), "w", encoding="utf-8") as f:
        f.write(RUN_PY)
    src_dir = os.path.join(build_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "utils.py"), "w", encoding="utf-8") as f:
        f.write(UTILS_PY.replace("__PAIR_MAX_LENGTH__", str(pair_max_length)))
    with open(os.path.join(build_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(METADATA_JSON, f, indent=2)
    _log("  run.py / src/utils.py / metadata.json written")

    # ---- 6. zip -- flat root, same layout scripts/build_submission_zip.py's
    # own validator expects for every other submission in this project ----
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    total_size = 0
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _dirs, files in os.walk(build_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, build_dir)
                zf.write(full, arcname)
                total_size += os.path.getsize(full)
                n_files += 1
    zip_size = os.path.getsize(zip_path)
    _log(f"  zipped {n_files} files, {total_size / 1e6:.1f}MB uncompressed -> "
        f"{zip_size / 1e6:.1f}MB ({zip_size / (5 * 1024**3) * 100:.1f}% of the 5GB cap)")

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        assert bad is None, f"[BUILD] corrupt zip entry in {zip_stem}: {bad}"
    _log(f"  zip integrity OK -> {zip_path}")

    shutil.rmtree(build_dir)
    return zip_path


def main():
    all_checkpoints = _find_all_checkpoints()
    _log(f"found {len(all_checkpoints)} checkpoint dir(s) under /kaggle/input across all "
        f"attached kernel_sources; experiments present: "
        f"{sorted(set(s.get('experiment') for _r, s in all_checkpoints))}")

    # Each experiment's build is independent -- one missing/unattached
    # kernel_source (e.g. a private source kernel this builder doesn't yet
    # have read access to) must NOT block the other two, which may already
    # be fully ready. No silent partial success either: every experiment
    # not built is named loudly in the final summary, and the kernel exits
    # non-zero if anything is missing, so a real re-run is still visibly
    # needed once the source is attached -- this just stops one temporarily
    # -unavailable source from wasting the two that already succeeded.
    n_total = len(TARGET_EXPERIMENTS) + len(EXTRA_TARGETS)
    built, failed = [], []
    for experiment_name, zip_stem, pair_max_length in TARGET_EXPERIMENTS:
        _log(f"=== {experiment_name} (best) @ PAIR_MAX_LENGTH={pair_max_length} -> {zip_stem}.zip ===")
        try:
            ckpt_dir, state = _pick_checkpoint(all_checkpoints, experiment_name)
            zip_path = _build_one_submission(ckpt_dir, state, zip_stem, pair_max_length)
            built.append(zip_path)
        except Exception as e:
            _log(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(f"{experiment_name} (best)")

    for experiment_name, label, zip_stem, pair_max_length in EXTRA_TARGETS:
        _log(f"=== {experiment_name} ({label}) @ PAIR_MAX_LENGTH={pair_max_length} -> "
            f"{zip_stem}.zip ===")
        try:
            ckpt_dir, state = _pick_checkpoint_by_label(all_checkpoints, experiment_name, label)
            zip_path = _build_one_submission(ckpt_dir, state, zip_stem, pair_max_length)
            built.append(zip_path)
        except Exception as e:
            _log(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(f"{experiment_name} ({label})")

    # per the same convention as frida-full-sft-submit-build: the kernel's
    # ONLY output is the zip files -- nothing else should remain in
    # /kaggle/working (each _build_one_submission already cleaned its own
    # staging dir).
    _log(f"done -- {len(built)}/{n_total} zip(s) built: {built}")
    if failed:
        _log(f"NOT built (see FAILED lines above for why): {failed}")
        raise RuntimeError(f"[BUILD] {len(failed)}/{n_total} target(s) could not be built: "
                           f"{failed} -- re-run once their kernel_source is actually attached/has "
                           f"that checkpoint. The zip(s) that DID build are still in "
                           f"/kaggle/working, this failure is so the run's overall status honestly "
                           f"reflects partial completion.")


if __name__ == "__main__":
    main()
