# %% [code]
"""
CPU-only kernel: assembles the H100-judging-stand submission zip for
frida-full-sft's "best" checkpoint (val_macro_pr_auc=0.7582, epoch 3/3,
maxamsecret/frida-full-sft-w) -- a genuine full fine-tune, NOT a LoRA
adapter (kaggle_kernels/frida-full-sft/frida_full_sft.py). Does no
inference itself; its only job is to produce ONE file in /kaggle/working:
the submission zip. No TPU/GPU needed for this.

WHY THIS SUBMISSION'S PIPELINE IS SIMPLER THAN EVERY PRIOR FRIDA SUBMISSION
IN THIS PROJECT: every earlier one (submits/frida/*) ships a LoRA adapter
+ a separately-bundled frida-base, because the adapter is a small delta on
top of pretrained weights that must stay available at inference. This
checkpoint has no adapter at all -- every one of T5EncoderModel's own
parameters IS the fine-tuned result. So the "base model" bundled here is
not ai-forever/FRIDA's own pretrained weights at all; it's THIS
checkpoint's weights, saved via the model's own .save_pretrained(), fully
replacing whatever came from the Hub. Consequence: no peft dependency, no
adapter directory, no offline wheels bundling at all -- a real
simplification, not a corner cut (see run.py/utils.py's own docstrings for
what's genuinely inherited from the LoRA submissions' hard-won lessons
vs. what's structurally unnecessary here).

encoder.pt (3.87GB, fp32, a plain torch.save of model.encoder.state_dict()
-- NOT a HuggingFace save_pretrained() directory, no config.json alongside
it) is loaded into a freshly constructed T5EncoderModel(config), cast to
bf16, then re-saved via .save_pretrained(..., safe_serialization=True) --
NOT a raw safetensors.torch.save_file(state_dict, ...) on the loaded
state_dict directly, which would hit T5's own real tied-weight aliasing
(shared.weight / encoder.embed_tokens.weight are the SAME Parameter
object) and either crash or silently duplicate that tensor on disk;
save_pretrained's own serialization path already handles this correctly.
bf16 conversion roughly halves the ~3.87GB fp32 weight to ~1.94GB,
leaving real margin under the 5GB zip cap and matching how every FRIDA
LoRA submission in this project already ships its own base weights.

Text/pair contract byte-matches kaggle_kernels/frida-tokenize-items-cpu
(the kernel that built frida-tokenize-items-v3, which this checkpoint's
own item cache -- and therefore this checkpoint's own training -- used):
MAX_NAME_CHARS/MAX_CATEGORY_CHARS/MAX_ATTR_CHARS = 140/32/529,
ITEM_MAX_LENGTH=256, PREFIX="paraphrase: ", the v2 attribute-cleanup
policy (demote long/https/oem values), Category:-span excision +
fair-share pair truncation. This is NOT the older 140/64/640 @
ITEM_MAX_LENGTH=224 formula some earlier FRIDA LoRA submissions in this
repo used (submits/frida/tpu_human_llm_best_180_224_5120/src/utils.py) --
that was a different (v2) cache generation this checkpoint never trained
against; copying that formula here would be a real train/serve mismatch
despite looking almost identical.

PAIR_MAX_LENGTH=180 / BATCH_SIZE_FIXED=5248 / ITEM_MAX_LENGTH=256: per
direct instruction, chosen close to (not copied from) the LoRA
submissions' own proven values (176-192 pair length, ~5120 batch) as a
starting point for this checkpoint's own timing on the same H100 judging
hardware class -- this checkpoint trained at PAIR_MAX_LENGTH=192, so 180
is a real, accepted train/serve mismatch already precedented by every
prior FRIDA submission's own choice to serve shorter than trained (see
run.py/utils.py's own comments for the CLAUDE.md-documented timing reason).
Not yet measured on the real stand at these exact numbers -- this build
produces a candidate to test, not a validated-safe configuration.
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
CHECKPOINT_HINT = "/kaggle/input/datasets/maxamsecret/frida-full-sft-w/frida_full_sft/best"

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
BUILD_DIR = os.path.join(WORKING_DIR, "_submission_build")
ZIP_PATH = os.path.join(WORKING_DIR, "matching-frida_full_sft-submit.zip")


def _log(msg):
    print(f"[BUILD] {msg}", flush=True)


def _find_checkpoint_dir():
    """Content-based fallback, same convention every kernel in this project
    uses for /kaggle/input paths (the exact mount point under
    /kaggle/input/datasets/... has drifted before -- see this project's own
    CLAUDE.md/kaggle-experiment-safety notes on dataset path resolution)."""
    if os.path.isfile(os.path.join(CHECKPOINT_HINT, "encoder.pt")):
        return CHECKPOINT_HINT
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "encoder.pt" in files and "head.pt" in files and os.path.basename(root) == "best":
                return root
    raise RuntimeError("[BUILD] no best/ checkpoint (encoder.pt + head.pt) found under "
                       "/kaggle/input -- is maxamsecret/frida-full-sft-w mounted?")


# =====================================================================
# run.py / src/utils.py -- the ACTUAL submission code, embedded as string
# literals (this is a single-file Kaggle kernel with no access to this
# project's own repo files at judging time, so the submission's source
# has to be self-contained here, not imported from elsewhere).
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
# This checkpoint trained at PAIR_MAX_LENGTH=192 (frida_full_sft.py). 180
# is a deliberate serve-shorter-than-trained choice, per direct
# instruction, close to (not copied from) prior FRIDA submissions' own
# proven pair lengths on this same H100 judging hardware class (CLAUDE.md:
# 192 measured ~1.65x over Public's time budget there) -- an accepted
# train/serve mismatch, not assumed lossless, and not yet measured on the
# real stand at this exact value.
PAIR_MAX_LENGTH = 180
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
    """CLS (position-0) pooling -- matches frida_full_sft.py's own class
    exactly (FRIDA's model card recommendation)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).view(-1)


class FullSftHead(nn.Module):
    """[1024, 512, 128, 1] -- verbatim copy of frida_full_sft.py's own
    FullSftHead class. State_dict key shape (blocks.N.0/1/2.*, out.*) must
    match exactly for classifier.pt to load."""

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


def main():
    if os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)
    models_dir = os.path.join(BUILD_DIR, "models", "frida-full-sft")
    os.makedirs(models_dir, exist_ok=True)

    ckpt_dir = _find_checkpoint_dir()
    _log(f"checkpoint: {ckpt_dir}")
    with open(os.path.join(ckpt_dir, "train_state.json")) as f:
        train_state = json.load(f)
    _log(f"train_state.json: {train_state}")

    # ---- 1. base architecture + tokenizer, fresh from the Hub (this build
    # kernel has internet; the SUBMISSION it produces does not) ----
    _log(f"downloading {MODEL_NAME} tokenizer + config ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    hidden_dim = model.config.d_model
    _log(f"hidden_dim={hidden_dim}")

    # ---- 2. overwrite with the ACTUAL fine-tuned weights ----
    encoder_sd_path = os.path.join(ckpt_dir, "encoder.pt")
    _log(f"loading fine-tuned encoder weights from {encoder_sd_path} ...")
    encoder_sd = torch.load(encoder_sd_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"[BUILD] encoder state_dict mismatch -- missing={missing} "
                           f"unexpected={unexpected}. Refusing to ship a submission built "
                           f"on a silently-wrong weight load.")
    del encoder_sd
    _log("encoder weights loaded and verified (strict key match)")

    # ---- 3. bf16 cast (halves the ~3.87GB fp32 weight, matches every prior
    # FRIDA submission's own base-model dtype choice) + save via the
    # model's OWN save_pretrained -- NOT a raw safetensors.save_file on the
    # state_dict, which would hit T5's real shared.weight/
    # encoder.embed_tokens.weight tied-parameter aliasing. ----
    model = model.to(dtype=torch.bfloat16)
    _log(f"saving bf16 model + config to {models_dir} ...")
    model.save_pretrained(models_dir, safe_serialization=True)
    tokenizer.save_pretrained(models_dir)
    del model
    _log("model.safetensors + config.json + tokenizer files written")

    # ---- 4. classifier head, same bf16 cast, plain torch.save (matches
    # every prior FRIDA submission's own classifier.pt convention) ----
    head_sd_path = os.path.join(ckpt_dir, "head.pt")
    _log(f"loading classifier head weights from {head_sd_path} ...")
    head_sd = torch.load(head_sd_path, map_location="cpu")
    head_sd_bf16 = {k: v.to(dtype=torch.bfloat16) for k, v in head_sd.items()}
    classifier_path = os.path.join(BUILD_DIR, "classifier.pt")
    torch.save(head_sd_bf16, classifier_path)
    _log(f"classifier.pt written ({os.path.getsize(classifier_path):,} bytes)")

    # ---- 5. run.py / src/utils.py / metadata.json ----
    with open(os.path.join(BUILD_DIR, "run.py"), "w", encoding="utf-8") as f:
        f.write(RUN_PY)
    src_dir = os.path.join(BUILD_DIR, "src")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "utils.py"), "w", encoding="utf-8") as f:
        f.write(UTILS_PY)
    with open(os.path.join(BUILD_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(METADATA_JSON, f, indent=2)
    _log("run.py / src/utils.py / metadata.json written")

    # ---- 6. zip -- flat root (metadata.json, run.py, classifier.pt, src/,
    # models/), same layout scripts/build_submission_zip.py's own validator
    # expects for every other submission in this project ----
    if os.path.isfile(ZIP_PATH):
        os.remove(ZIP_PATH)
    total_size = 0
    n_files = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _dirs, files in os.walk(BUILD_DIR):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, BUILD_DIR)
                zf.write(full, arcname)
                total_size += os.path.getsize(full)
                n_files += 1
    zip_size = os.path.getsize(ZIP_PATH)
    _log(f"zipped {n_files} files, {total_size / 1e6:.1f}MB uncompressed -> "
        f"{zip_size / 1e6:.1f}MB ({zip_size / (5 * 1024**3) * 100:.1f}% of the 5GB cap)")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        assert bad is None, f"[BUILD] corrupt zip entry: {bad}"
    _log("zip integrity OK")

    # ---- 7. per direct instruction, the kernel's OUTPUT is ONLY the zip --
    # remove the staging directory so nothing else lands in /kaggle/working.
    shutil.rmtree(BUILD_DIR)
    _log(f"done -- {ZIP_PATH} is the only output")


if __name__ == "__main__":
    main()
