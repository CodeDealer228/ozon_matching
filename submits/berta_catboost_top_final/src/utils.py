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
import json
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging


def maybe_offline_install(wheels_dir):
    """peft + catboost (+ their dependency closures) are absent from
    odsai/ecup26-matching-baseline:1.0 -- exact same package list as
    submits/berta_catboost (this submission's wheels/ is a direct copy of
    that one's). scikit-learn/joblib/scipy are NOT bundled here and don't
    need to be: the organizer's own baseline submission (submits/baseline)
    uses StandardScaler+LogisticRegression on the SAME judging image, so
    scikit-learn (and its scipy dependency) is already present -- confirmed
    by every *_probe submission in this project already calling
    `joblib.load()`/importing sklearn with no offline-install step of its
    own and no reported failure on that front."""
    if not os.path.isdir(wheels_dir):
        print(f"[SETUP] no wheels dir at {wheels_dir}, skipping offline install", flush=True)
        return
    pkgs = ["peft", "accelerate", "huggingface_hub", "click", "hf-xet", "httpx", "httpcore",
            "h11", "anyio", "idna", "certifi", "colorama", "typer", "annotated-doc", "rich",
            "pygments", "markdown-it-py", "mdurl", "shellingham", "packaging", "tqdm", "filelock",
            "fsspec", "typing_extensions", "psutil", "pyyaml", "safetensors", "catboost"]
    target_dir = os.path.join(tempfile.gettempdir(), "berta_catboost_top_final_offline_pkgs")
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


# BERTA pair-text contract -- must byte-match kaggle_kernels/berta-llm-tpu-sft
# (the run that produced maxamsecret/berta-tpu-lora-top-checkpoint, the
# adapter/classifier shipped here) and submits/berta_tpu_lora (the same
# checkpoint's own earlier submission) -- PAIR_MAX_LENGTH=256, not the
# PAIR_MAX_LENGTH=128 every earlier BERTA submission in this project used.
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640
PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 256

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

# TF-IDF raw-text contract -- must byte-match kaggle_kernels/
# tfidf-category-index-fit's own build_text_expr() (the kernel that fit the
# vectorizers shipped here): "name[:128] attributes[:384]", NOT the BERTA
# "Name: ... Category: ... Attributes: ..." formatted text above -- a
# different, independent text representation for a different signal.
TFIDF_MAX_NAME_CHARS, TFIDF_MAX_ATTR_CHARS = 128, 384


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_needed_items(items_path, needed_ids):
    """Single read of items.parquet for every id referenced by the input
    pairs -- returns per-id: the BERTA-formatted text, the TF-IDF raw text,
    and the (truncated) category string, so every downstream feature reads
    from the same one dataframe pull instead of re-reading the parquet."""
    df = pd.read_parquet(
        items_path,
        engine="pyarrow",
        columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", list(set(needed_ids)))],
    )
    name = df["name"].fillna("").astype(str)
    category_full = df["category"].fillna("UNKNOWN").astype(str)
    attributes = df["attributes"].fillna("").astype(str)

    berta_text = (
        "Name: " + name.str.slice(0, MAX_NAME_CHARS) +
        " Category: " + category_full.str.slice(0, MAX_CATEGORY_CHARS) +
        " Attributes: " + attributes.str.slice(0, MAX_ATTR_CHARS)
    ).str.slice(0, MAX_ITEM_CHARS)

    tfidf_text = (
        name.str.slice(0, TFIDF_MAX_NAME_CHARS) + " " + attributes.str.slice(0, TFIDF_MAX_ATTR_CHARS)
    ).str.replace("\n", " ", regex=False).str.strip()

    category = category_full.str.slice(0, MAX_CATEGORY_CHARS)

    ids = df["id"].tolist()
    return (
        dict(zip(ids, berta_text.tolist())),
        dict(zip(ids, tfidf_text.tolist())),
        dict(zip(ids, category.tolist())),
    )


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


def _tokenize_items(tokenizer, id_to_text):
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
    n = len(match_df)
    content_budget = PAIR_MAX_LENGTH - 3
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
    def __init__(self, hidden_dim, dropout_p=0.10):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, pooled):
        return self.linear(self.dropout(pooled))


class BertaEncoderAndHead(nn.Module):
    """MEAN pooling, THEN the classifier -- returns BOTH the pooled
    embedding (768) and the raw classifier logit, since this submission's
    CatBoost head consumes both (unlike submits/berta_tpu_lora, which only
    needs the classifier score, or submits/berta_catboost, which only needs
    the pooled embedding -- this one needs both at once, matching exactly
    what kaggle_kernels/berta-top-embed-extract extracted for training)."""

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
        logit = self.classifier(pooled).view(-1).float()
        return pooled, logit


def _load_model(base_model_path, adapter_path, classifier_path, device):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        base_model = AutoModel.from_pretrained(base_model_path, attn_implementation="sdpa",
                                                  torch_dtype=torch.bfloat16)
    except (TypeError, ValueError):
        base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    hf_logging.set_verbosity(default_verbosity)

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    hidden_dim = base_model.config.hidden_size

    classifier_head = ClassifierHead(hidden_dim)
    raw_state = torch.load(classifier_path, map_location="cpu")
    if "linear.weight" in raw_state:
        classifier_head.load_state_dict(raw_state)
    else:
        classifier_head.linear.load_state_dict(raw_state)
    # Classifier stays FLOAT32 -- see submits/berta_tpu_lora/src/utils.py's
    # own comment for the real local crash this avoids (bf16/fp32 mismatch).
    classifier_head = classifier_head.to(dtype=torch.float32)

    model = BertaEncoderAndHead(peft_model, classifier_head)
    model.to(device)
    model.eval()
    return model, tokenizer


def _run_berta(model, input_ids_all, attention_mask_all, token_type_ids_all, device, batch_size):
    use_amp = device.type == "cuda"
    n = input_ids_all.shape[0]
    pooled_chunks, logit_chunks = [], []

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            input_ids = torch.from_numpy(input_ids_all[i:i + batch_size]).to(device, non_blocking=True)
            attention_mask = torch.from_numpy(attention_mask_all[i:i + batch_size]).to(device, non_blocking=True)
            token_type_ids = torch.from_numpy(token_type_ids_all[i:i + batch_size]).to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pooled, logit = model(input_ids=input_ids, attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)
            else:
                pooled, logit = model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

            pooled_chunks.append(pooled.float().cpu().numpy())
            logit_chunks.append(logit.float().cpu().numpy())

    return np.concatenate(pooled_chunks, axis=0), np.concatenate(logit_chunks, axis=0)


# =====================================================================
# TF-IDF -- loads the ~20 small fitted per-category vectorizers (NOT a
# full-catalog matrix, see kaggle_kernels/tfidf-category-index-fit's own
# module docstring for the memory reasoning this mirrors exactly) and
# transforms on the fly, only for the items actually referenced by this
# run's input pairs -- O(pairs), never O(catalog size).
# =====================================================================

def _load_tfidf_vectorizers(tfidf_dir):
    with open(os.path.join(tfidf_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vectorizers = {}
    for category, fname in manifest["categories"].items():
        vectorizers[category] = joblib.load(os.path.join(tfidf_dir, fname))
    print(f"[TFIDF] {len(vectorizers)} category vectorizers loaded from {tfidf_dir}", flush=True)
    return vectorizers


def _score_tfidf(vectorizers, match_df, id_to_tfidf_text, id_to_category):
    n = len(match_df)
    scores = np.zeros(n, dtype=np.float32)
    tmp = pd.DataFrame({
        "idx": np.arange(n),
        "id1": match_df["id1"].to_numpy(),
        "id2": match_df["id2"].to_numpy(),
    })
    tmp["category1"] = tmp["id1"].map(id_to_category).fillna("UNKNOWN")

    for cat, group in tmp.groupby("category1"):
        vec = vectorizers.get(cat)
        if vec is None:
            continue
        texts1 = [id_to_tfidf_text.get(i, "") for i in group["id1"]]
        texts2 = [id_to_tfidf_text.get(i, "") for i in group["id2"]]
        m1 = vec.transform(texts1)
        m2 = vec.transform(texts2)
        cos = np.asarray(m1.multiply(m2).sum(axis=1)).flatten()
        scores[group["idx"].to_numpy()] = cos.astype(np.float32)

    return scores, tmp["category1"].to_numpy()


def predict_pipeline(items_path, matches_path, base_model_path, adapter_path, classifier_path,
                      tfidf_dir, catboost_model_path, output_csv_path, wheels_dir, batch_size=1024):
    t0 = time.time()
    print("=== Starting BERTA+LoRA embeddings+score + TF-IDF -> CatBoost prediction pipeline ===", flush=True)

    maybe_offline_install(wheels_dir)
    from catboost import CatBoostClassifier

    print("[1/8] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/8] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/8] Loading needed item texts (BERTA + TF-IDF variants + category) ...", flush=True)
    id_to_berta_text, id_to_tfidf_text, id_to_category = _load_needed_items(items_path, needed_ids)
    print(f"[2/8] items loaded={len(id_to_berta_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/8] Loading BERTA+LoRA model + tokenizer ...", flush=True)
    device = _get_device()
    model, tokenizer = _load_model(base_model_path, adapter_path, classifier_path, device)
    print(f"[3/8] device={device} ({time.time() - t0:.1f}s)", flush=True)

    print("[4/8] Tokenizing items (once each) + locating Category:/Attributes: labels ...", flush=True)
    id_to_item = _tokenize_items(tokenizer, id_to_berta_text)
    print(f"[4/8] done ({time.time() - t0:.1f}s)", flush=True)

    print(f"[5/8] Building pairs ([CLS] A [SEP] B [SEP], excise Category, fair-share truncate "
          f"to PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}) ...", flush=True)
    input_ids_all, attention_mask_all, token_type_ids_all = _build_pairs(
        match_df, id_to_item, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    )
    print(f"[5/8] done ({time.time() - t0:.1f}s)", flush=True)

    print("[6/8] Running BERTA (pooled embedding + classifier score) ...", flush=True)
    pooled, logit = _run_berta(model, input_ids_all, attention_mask_all, token_type_ids_all,
                                device, batch_size=batch_size)
    classifier_score = 1.0 / (1.0 + np.exp(-logit))   # sigmoid, matches training-side feature build
    print(f"[6/8] done, pooled={pooled.shape} ({time.time() - t0:.1f}s)", flush=True)

    print("[7/8] Scoring TF-IDF per-category cosine similarity ...", flush=True)
    vectorizers = _load_tfidf_vectorizers(tfidf_dir)
    tfidf_score, category_arr = _score_tfidf(vectorizers, match_df, id_to_tfidf_text, id_to_category)
    print(f"[7/8] done ({time.time() - t0:.1f}s)", flush=True)

    print("[8/8] Running CatBoost inference ...", flush=True)
    emb_cols = {f"emb_{i}": pooled[:, i] for i in range(pooled.shape[1])}
    feature_df = pd.DataFrame(emb_cols)
    feature_df["classifier_score"] = classifier_score.astype(np.float32)
    feature_df["tfidf_score"] = tfidf_score
    feature_df["category"] = category_arr.astype(str)

    cb_model = CatBoostClassifier()
    cb_model.load_model(catboost_model_path)
    probs = cb_model.predict_proba(feature_df)[:, 1]
    print(f"[8/8] done ({time.time() - t0:.1f}s)", flush=True)

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
