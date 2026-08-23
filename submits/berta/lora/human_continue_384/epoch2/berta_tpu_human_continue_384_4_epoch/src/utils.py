import os

# Hardware-aware env tuning, mirrored from submits/lora/src/utils.py -- 20 CPU
# cores / 200GB RAM / 1x H100 80GB at inference time.
os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("RAYON_NUM_THREADS", "32")

import sys
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging


def maybe_offline_install(wheels_dir):
    """peft (+ its dependency closure) is absent from
    odsai/ecup26-matching-baseline:1.0 -- this submission loads a LoRA
    adapter WITHOUT merging it (the adapter here was trained on TPU against
    an already-merged base, so a second merge step would just be extra
    build machinery for no inference-time gain).

    This checkpoint is kaggle_kernels/berta-tpu-human-continue-384's own
    "best" (epoch 1 of a planned 10, val_macro_pr_auc=0.7551 -- val declined
    monotonically every epoch after that, 0.7501/0.7462/.../0.7217 by epoch
    10, real overfitting on the much smaller human-only train split, 292,523
    rows vs. the 11.48M-row LLM-mixed set the ORIGINAL top checkpoint saw).

    --target + sys.path append (not a normal site-packages install), exact
    package list with --no-deps, --no-index --find-links wheels_dir: same
    reasoning as submits/frida_lora_adapter/src/utils.py's own
    maybe_offline_install -- see that docstring for the full history
    (read-only /root/.local on the real judging container, PEP 668
    externally-managed-environment, --target's isolated dependency
    resolution). Package names are canonical PyPI names verified from each
    wheel's own METADATA, not assumed from filenames."""
    if not os.path.isdir(wheels_dir):
        print(f"[SETUP] no wheels dir at {wheels_dir}, skipping offline install", flush=True)
        return
    pkgs = ["peft", "accelerate", "huggingface_hub", "click", "hf-xet", "httpx", "httpcore",
            "h11", "anyio", "idna", "certifi", "colorama", "typer", "annotated-doc", "rich",
            "pygments", "markdown-it-py", "mdurl", "shellingham", "packaging", "tqdm", "filelock",
            "fsspec", "typing_extensions", "psutil", "pyyaml", "safetensors"]
    target_dir = os.path.join(tempfile.gettempdir(), "berta_tpu_human_continue_384_offline_pkgs")
    print(f"[SETUP] offline-installing {pkgs} (--no-deps, exact list, from {wheels_dir}) "
          f"into {target_dir} ...", flush=True)
    try:
        os.makedirs(target_dir, exist_ok=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-cache-dir", "--no-deps",
             "--find-links", wheels_dir, "--target", target_dir, *pkgs]
        )
        sys.path.append(target_dir)
        print(f"[SETUP] offline install ok, {target_dir} appended to sys.path", flush=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[WARN] offline install failed ({e}) -- continuing, import will fail loudly "
              f"if peft is truly absent", flush=True)


# Must byte-match kaggle_kernels/berta-tpu-human-continue-384/
# berta_tpu_human_continue_384.py (the run that produced THIS checkpoint) --
# PAIR_MAX_LENGTH=384, raised again from the 256 the checkpoint this one
# resumes from used. ITEM_MAX_LENGTH stays 256 -- there is no item-token
# cache built at a longer length, so no single item ever contributes more
# than 256 tokens to a pair regardless of the 384 pair budget (a known,
# documented limitation of that training run, not an oversight here).
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640
PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 384

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_needed_item_texts(items_path, needed_ids):
    df = pd.read_parquet(
        items_path,
        engine="pyarrow",
        columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", list(set(needed_ids)))],
    )
    name = df["name"].fillna("").astype(str).str.slice(0, MAX_NAME_CHARS)
    category = df["category"].fillna("").astype(str).str.slice(0, MAX_CATEGORY_CHARS)
    attributes = df["attributes"].fillna("").astype(str).str.slice(0, MAX_ATTR_CHARS)
    text = (
        "Name: " + name + " Category: " + category + " Attributes: " + attributes
    ).str.slice(0, MAX_ITEM_CHARS)
    return dict(zip(df["id"].tolist(), text.tolist()))


def _find_token_pattern_starts(arr, pattern):
    """First index i per row where arr[row, i:i+len(pattern)] == pattern, or
    -1 if not found. Vectorized, arbitrary-length pattern (BERTA's
    "Attributes:" is a 4-token WordPiece split, not 2)."""
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


def _tokenize_items(tokenizer, id_to_text):
    """Tokenizes every unique item ONCE (paraphrase: prefix, WITHOUT special
    tokens -- CLS/SEP/token_type_ids assembled at pair-build time), then
    locates the Category:/Attributes: label span per item for excision."""
    ids = list(id_to_text.keys())
    texts = [PREFIX + id_to_text[i] for i in ids]

    enc = tokenizer(texts, add_special_tokens=False, padding="max_length", truncation=True,
                     max_length=ITEM_MAX_LENGTH, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int32)
    lengths = enc["attention_mask"].sum(axis=1).astype(np.int32)

    category_start = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_PATTERN)
    attr_start = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_PATTERN)

    id_to_item = {}
    for row, item_id in enumerate(ids):
        id_to_item[item_id] = (input_ids[row], int(lengths[row]), int(category_start[row]), int(attr_start[row]))
    return id_to_item


def allocate_pair_budget(len1, len2, budget):
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


def _build_pairs(match_df, id_to_item, cls_id, sep_id, pad_id):
    """[CLS] item1[:keep1] [SEP] item2[:keep2] [SEP] + token_type_ids (0 for
    CLS+item1+first SEP, 1 for item2+final SEP) -- real BERT pair format."""
    n = len(match_df)
    content_budget = PAIR_MAX_LENGTH - 3   # reserve CLS + SEP + SEP
    input_ids = np.full((n, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    token_type_ids = np.zeros((n, PAIR_MAX_LENGTH), dtype=np.int64)
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

        if cl1 + cl2 <= content_budget:
            keep1, keep2 = cl1, cl2
        else:
            keep1, keep2 = allocate_pair_budget(cl1, cl2, content_budget)

        seg1 = compact1[:keep1]
        seg2 = compact2[:keep2]
        n1 = 1 + keep1 + 1
        n2 = keep2 + 1
        m = n1 + n2

        input_ids[i, 0] = cls_id
        input_ids[i, 1:1 + keep1] = seg1
        input_ids[i, 1 + keep1] = sep_id
        input_ids[i, n1:n1 + keep2] = seg2
        input_ids[i, n1 + keep2] = sep_id
        token_type_ids[i, n1:m] = 1
        attention_mask[i, :m] = 1

    if n_missing:
        print(f"[WARN] {n_missing} pairs referenced an item id missing from items.parquet "
              f"(left as all-pad -- attention_mask all-zero for those rows)", flush=True)

    return input_ids, attention_mask, token_type_ids


class ClassifierHead(nn.Module):
    """Dropout -> Linear. The dropout is inert at inference (model.eval()),
    but the wrapper must exist so the checkpoint's own state_dict keys
    ("linear.weight"/"linear.bias", NOT bare "weight"/"bias") load cleanly
    -- this project already hit a real crash loading such a checkpoint into
    a bare nn.Linear."""

    def __init__(self, hidden_dim, dropout_p=0.10):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, pooled):
        return self.linear(self.dropout(pooled))


class BertaCrossEncoderClassifier(nn.Module):
    """MEAN pooling (mask-weighted average over last_hidden_state) -- BERTA's
    own model card explicitly replaces FRIDA's CLS pooling with mean
    pooling, and every BERTA training kernel in this project follows that."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).float()
        return self.classifier(pooled).view(-1).float()


def _load_model(base_model_path, adapter_path, classifier_path, device):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    # torch_dtype=, not dtype= -- this repo hit that exact kwarg-rename
    # incompatibility once already (giga3b probe phase); the judging image's
    # transformers version is unverified, so use the longer-supported name.
    try:
        base_model = AutoModel.from_pretrained(base_model_path, attn_implementation="sdpa",
                                                  torch_dtype=torch.bfloat16)
    except (TypeError, ValueError):
        base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    hf_logging.set_verbosity(default_verbosity)

    # is_trainable defaults to False -- correct at inference.
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    hidden_dim = base_model.config.hidden_size

    classifier_head = ClassifierHead(hidden_dim)
    raw_state = torch.load(classifier_path, map_location="cpu")
    if "linear.weight" in raw_state:
        classifier_head.load_state_dict(raw_state)
    else:
        classifier_head.linear.load_state_dict(raw_state)
    # Classifier stays FLOAT32 -- deliberately NOT cast to bf16 alongside the
    # encoder. This mirrors the training script exactly (kaggle_kernels/
    # berta-llm-tpu-sft: frozen base bf16, LoRA + classifier fp32, and
    # forward() upcasts the pooled vector with .float() before the head).
    # Casting it to bf16 here -- as submits/frida_lora_adapter does for ITS
    # head, where the training-side pairing was different -- produced a real
    # local crash: "mat1 and mat2 must have the same dtype, but got Float
    # and BFloat16", because forward() hands the head an fp32 pooled vector.
    classifier_head = classifier_head.to(dtype=torch.float32)

    model = BertaCrossEncoderClassifier(peft_model, classifier_head)
    model.to(device)
    model.eval()
    return model, tokenizer


def _run_inference(model, input_ids_all, attention_mask_all, token_type_ids_all, device, batch_size):
    use_amp = device.type == "cuda"
    n = input_ids_all.shape[0]
    probs = np.empty(n, dtype=np.float32)

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            input_ids = torch.from_numpy(input_ids_all[i:i + batch_size]).to(device, non_blocking=True)
            attention_mask = torch.from_numpy(attention_mask_all[i:i + batch_size]).to(device, non_blocking=True)
            token_type_ids = torch.from_numpy(token_type_ids_all[i:i + batch_size]).to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            probs[i:i + input_ids.shape[0]] = torch.sigmoid(logits.float()).cpu().numpy()

    return probs


def predict_pipeline(items_path, matches_path, base_model_path, adapter_path, classifier_path,
                      output_csv_path, wheels_dir, batch_size=1024):
    t0 = time.time()
    print("=== Starting BERTA+LoRA (human-continue-384 checkpoint, seq=384) prediction pipeline ===", flush=True)

    maybe_offline_install(wheels_dir)

    print("[1/6] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/6] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/6] Loading needed item texts ...", flush=True)
    id_to_text = _load_needed_item_texts(items_path, needed_ids)
    print(f"[2/6] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/6] Loading BERTA+LoRA model + tokenizer ...", flush=True)
    device = _get_device()
    model, tokenizer = _load_model(base_model_path, adapter_path, classifier_path, device)
    print(f"[3/6] device={device} ({time.time() - t0:.1f}s)", flush=True)

    print("[4/6] Tokenizing items (once each) + locating Category:/Attributes: labels ...", flush=True)
    id_to_item = _tokenize_items(tokenizer, id_to_text)
    print(f"[4/6] done ({time.time() - t0:.1f}s)", flush=True)

    print(f"[5/6] Building pairs ([CLS] A [SEP] B [SEP], excise Category, fair-share truncate "
          f"to PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}) ...", flush=True)
    input_ids_all, attention_mask_all, token_type_ids_all = _build_pairs(
        match_df, id_to_item, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    )
    print(f"[5/6] done ({time.time() - t0:.1f}s)", flush=True)

    print("[6/6] Running inference ...", flush=True)
    probs = _run_inference(model, input_ids_all, attention_mask_all, token_type_ids_all,
                            device, batch_size=batch_size)
    print(f"[6/6] done ({time.time() - t0:.1f}s)", flush=True)

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
