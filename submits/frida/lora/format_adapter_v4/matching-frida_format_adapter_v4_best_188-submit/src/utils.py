import os

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

# New item-text contract (no prefix and no category in model input):
#   product name
#   attribute name = attribute value
#   attribute name = attribute value
MAX_NAME_CHARS, MAX_ATTR_CHARS = 160, 640
ITEM_MAX_LENGTH = 188
# These checkpoints trained at PAIR_MAX_LENGTH=192 (frida_full_sft_continue.py,
# same as frida_full_sft.py). 180 is a deliberate serve-shorter-than-trained
# choice, per direct instruction, matching frida-full-sft-submit-build's own
# already-proven value (that build's submission scored 0.481 on the real
# test bench at this exact PAIR_MAX_LENGTH/BATCH_SIZE_FIXED).
PAIR_MAX_LENGTH = 188
BATCH_SIZE_FIXED = 5248

# v2 attribute-cleanup policy, byte-identical to the cache-build kernel:
# demote any value that is long, contains a link, or mentions "oem".
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
        return []
    try:
        d = json.loads(attr_str)
    except Exception:
        return []
    if not isinstance(d, dict) or not d:
        return []
    ranked = [(_demotion_rank(k, v), k, v) for k, v in d.items()]
    ranked.sort(key=lambda t: t[0])
    return [(k, v) for _r, k, v in ranked]


def _clean_field(value):
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text.replace(" = ", "=")


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


def _format_item_text(name, attributes):
    name = _clean_field(name)[:MAX_NAME_CHARS]
    lines = []
    for key, value in reorder_attributes(attributes):
        key = _clean_field(key)
        value = _clean_field(value)
        if key or value:
            lines.append(f"{key} = {value}")
    attrs = "\n".join(lines)[:MAX_ATTR_CHARS]
    return f"{name}\n{attrs}" if attrs else name


def _load_needed_item_texts(items_path, needed_ids):
    """Builds the approved no-prefix/no-category item text. Category is
    still loaded from parquet but deliberately kept out of model input.
    Single-threaded formatting, deliberately -- see
    submits/frida/tpu_human_llm_best_180_224_5120/src/utils.py's own
    comment for the real multiprocessing.Pool(os.cpu_count()) crash
    (os.cpu_count() reported 240 on the real judging container; forking
    240 processes hit BlockingIOError from os.fork() itself) this project
    already found and removed -- not reintroduced here."""
    df = pd.read_parquet(
        items_path, engine="pyarrow", columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", list(set(needed_ids)))],
    )
    texts = [
        _format_item_text(name, attributes)
        for name, attributes in zip(df["name"], df["attributes"])
    ]
    return dict(zip(df["id"].tolist(), texts))


TOKENIZE_CHUNK_SIZE = 14_000


def _tokenize_items(tokenizer, id_to_text):
    """Tokenizes every unique item once, without prefix or category."""
    ids = list(id_to_text.keys())
    texts = [id_to_text[i] for i in ids]

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

    id_to_item = {}
    for row, item_id in enumerate(ids):
        id_to_item[item_id] = (input_ids[row], int(lengths[row]))
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


def _build_pairs(match_df, id_to_item, pad_id):
    """Fair-share truncate and concatenate two already-formatted items."""
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
        compact1 = item1[0][:item1[1]]
        compact2 = item2[0][:item2[1]]
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

    print(f"[4/7] Tokenizing items (chunks of {TOKENIZE_CHUNK_SIZE:,}) ...", flush=True)
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
