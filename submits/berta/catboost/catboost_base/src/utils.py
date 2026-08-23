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
    """peft and catboost (+ their own dependency closures) are absent from
    odsai/ecup26-matching-baseline:1.0 -- same situation this project's
    submits/frida_lora_adapter already solved for peft; catboost is a new
    addition here since this submission replaces the trained classifier
    head with a CatBoost model on top of BERTA's pooled embeddings.

    --target + sys.path append (not a normal site-packages install), exact
    package list with --no-deps, --no-index --find-links wheels_dir: same
    reasoning as submits/frida_lora_adapter/src/utils.py's own
    maybe_offline_install -- see that function's docstring for the full
    history (read-only /root/.local on the real judging container, PEP 668
    externally-managed-environment, --target's isolated dependency
    resolution). Package names are canonical PyPI names verified from each
    wheel's own METADATA, not assumed from filenames."""
    if not os.path.isdir(wheels_dir):
        print(f"[SETUP] no wheels dir at {wheels_dir}, skipping offline install", flush=True)
        return
    # graphviz/plotly/matplotlib/six/narwhals deliberately NOT bundled despite
    # being in catboost's own declared Requires-Dist -- confirmed directly
    # (local import with those three blocked via a meta_path hook) that
    # CatBoostClassifier/.load_model()/.predict_proba() never actually import
    # them; catboost only reaches for them lazily, inside optional plotting
    # helpers this submission never calls.
    pkgs = ["peft", "accelerate", "huggingface_hub", "click", "hf-xet", "httpx", "httpcore",
            "h11", "anyio", "idna", "certifi", "colorama", "typer", "annotated-doc", "rich",
            "pygments", "markdown-it-py", "mdurl", "shellingham", "packaging", "tqdm", "filelock",
            "fsspec", "typing_extensions", "psutil", "pyyaml", "safetensors", "catboost"]
    target_dir = os.path.join(tempfile.gettempdir(), "berta_catboost_offline_pkgs")
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
              f"if peft/catboost are truly absent", flush=True)


# Must byte-match kaggle_kernels/berta-*/*.py's actual training/extraction
# pipeline (BertModel real sentence-pair format, NOT FRIDA's T5 "two
# self-contained sequences concatenated"): [CLS] item1 [SEP] item2 [SEP] +
# token_type_ids, item-level tokenization WITHOUT special tokens
# (add_special_tokens=False -- CLS/SEP/token_type_ids assembled at pair-
# build time once the actual pair is known), Category:/Attributes: label
# excision via a 4-token WordPiece pattern for "Attributes:" (unlike
# FRIDA's byte-BPE, where it's 2 tokens -- "Attributes" isn't a single
# BERTA vocab entry), MEAN pooling (mask-weighted average over
# last_hidden_state, BERTA's own explicit deviation from FRIDA's CLS
# pooling).
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640
PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 128
PAIR_MAX_LENGTH = 128

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
    -1 if not found. Vectorized, arbitrary-length pattern -- see kernel
    scripts' own version for why (BERTA's "Attributes:" splits into 4
    WordPiece tokens, not the 2 a fixed-length version would assume)."""
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
    CLS+item1+first SEP, 1 for item2+final SEP) -- real BERT pair format,
    NOT FRIDA's plain concatenation (T5 has no segment concept at all)."""
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


class BertaEncoderPooler(nn.Module):
    """Same encoder + MEAN pooling as the training/extraction pipeline's
    BertaCrossEncoderClassifier, but stops BEFORE the classifier -- returns
    the pooled vector itself, which CatBoost consumes downstream instead of
    a Linear classifier head."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).float()


def _load_model(base_model_path, adapter_path, device):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    # torch_dtype=, not dtype= -- this repo hit this exact incompatibility
    # once already (giga3b probe phase); the judging image's transformers
    # version is unverified, so use the longer-supported kwarg name.
    try:
        base_model = AutoModel.from_pretrained(base_model_path, attn_implementation="sdpa",
                                                  torch_dtype=torch.bfloat16)
    except (TypeError, ValueError):
        base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    hf_logging.set_verbosity(default_verbosity)

    # is_trainable defaults to False here -- correct at inference (no
    # gradients needed), unlike a resumed TRAINING run (this repo's own
    # documented PEFT is_trainable trap).
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    model = BertaEncoderPooler(peft_model)
    model.to(device)
    model.eval()
    return model, tokenizer


def _extract_embeddings(model, input_ids_all, attention_mask_all, token_type_ids_all, device, batch_size):
    use_amp = device.type == "cuda"
    n = input_ids_all.shape[0]
    hidden_dim = None
    emb_chunks = []

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            input_ids = torch.from_numpy(input_ids_all[i:i + batch_size]).to(device, non_blocking=True)
            attention_mask = torch.from_numpy(attention_mask_all[i:i + batch_size]).to(device, non_blocking=True)
            token_type_ids = torch.from_numpy(token_type_ids_all[i:i + batch_size]).to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pooled = model(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
            else:
                pooled = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            emb_chunks.append(pooled.float().cpu().numpy())

    return np.concatenate(emb_chunks, axis=0)


def predict_pipeline(items_path, matches_path, base_model_path, adapter_path, catboost_model_path,
                      output_csv_path, wheels_dir, batch_size=1024):
    t0 = time.time()
    print("=== Starting BERTA+LoRA embeddings -> CatBoost prediction pipeline ===", flush=True)

    maybe_offline_install(wheels_dir)
    from catboost import CatBoostClassifier

    print("[1/7] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/7] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/7] Loading needed item texts ...", flush=True)
    id_to_text = _load_needed_item_texts(items_path, needed_ids)
    print(f"[2/7] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/7] Loading BERTA+LoRA model + tokenizer ...", flush=True)
    device = _get_device()
    model, tokenizer = _load_model(base_model_path, adapter_path, device)
    print(f"[3/7] device={device} ({time.time() - t0:.1f}s)", flush=True)

    print("[4/7] Tokenizing items (once each) + locating Category:/Attributes: labels ...", flush=True)
    id_to_item = _tokenize_items(tokenizer, id_to_text)
    print(f"[4/7] done ({time.time() - t0:.1f}s)", flush=True)

    print("[5/7] Building pairs ([CLS] A [SEP] B [SEP], excise Category, fair-share truncate "
          f"to PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}) ...", flush=True)
    input_ids_all, attention_mask_all, token_type_ids_all = _build_pairs(
        match_df, id_to_item, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    )
    print(f"[5/7] done ({time.time() - t0:.1f}s)", flush=True)

    print("[6/7] Extracting pooled embeddings ...", flush=True)
    embeddings = _extract_embeddings(model, input_ids_all, attention_mask_all, token_type_ids_all,
                                       device, batch_size=batch_size)
    print(f"[6/7] done, embeddings shape={embeddings.shape} ({time.time() - t0:.1f}s)", flush=True)

    print("[7/7] Running CatBoost inference ...", flush=True)
    cb_model = CatBoostClassifier()
    cb_model.load_model(catboost_model_path)
    probs = cb_model.predict_proba(embeddings)[:, 1]
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
