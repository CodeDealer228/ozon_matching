"""Build the public BERTA v4 cache and one immutable FRIDA teacher-score cache.

Scope exactly matches continue_c_llm3m: item-disjoint 70/30 human split,
3M deterministic category/target-stratified LLM train rows, and the fixed
LLM validation split.  Text is byte-compatible with utils2.py: no prefix,
no category, cleaned name plus reordered ``key = value`` attribute lines.
"""
import os
import sys
import gc
import json
import math
import shutil
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "transformers==5.15.0", "tokenizers==0.22.2"])

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr

SEED = 42
BERTA_MODEL = "sergeyzh/BERTA"
ITEM_MAX_LENGTH = 384
TEACHER_ITEM_MAX_LENGTH = 188
TEACHER_PAIR_MAX_LENGTH = 188
TEACHER_BATCH = 32
LLM_TRAIN_ROWS = 3_000_000
MAX_NAME_CHARS, MAX_ATTR_CHARS = 160, 640
LONG_VALUE_CHARS, LINK_MARKER, OEM_MARKER = 200, "https", "oem"
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "berta_format_v4_384_cache")

MATCHING_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching"
HUMAN_SPLIT_NAME = "matches_item_disjoint_70_30_seed42.parquet"
LLM_SPLIT_NAME = "llm_train_val_split.parquet"
TEACHER_MODEL_DIRNAME = "frida-format-adapter-v4"

_START = time.perf_counter()
_SHARED = {}


def log(stage, msg):
    print(f"[{time.perf_counter() - _START:8.1f}s] {stage}: {msg}", flush=True)


def find_file(name):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def rank_value(key, value):
    text = value if isinstance(value, str) else str(value)
    low = text.lower()
    if "oem" in low or "oem" in str(key).lower():
        return 3
    if LINK_MARKER in low:
        return 2
    return int(len(text) > LONG_VALUE_CHARS)


def clean(value):
    return " ".join(("" if value is None else str(value)).split()).replace(" = ", "=")


def format_item(name, attributes):
    name = clean(name)[:MAX_NAME_CHARS]
    try:
        attrs = json.loads(attributes) if attributes else {}
    except Exception:
        attrs = {}
    lines = []
    if isinstance(attrs, dict):
        for key, value in sorted(attrs.items(), key=lambda kv: rank_value(*kv)):
            key, value = clean(key), clean(value)
            if key or value:
                lines.append(f"{key} = {value}")
    attrs = "\n".join(lines)[:MAX_ATTR_CHARS]
    return f"{name}\n{attrs}" if attrs else name


def stratified_llm_sample(target, cat_id):
    rng = np.random.RandomState(SEED)
    keep = []
    categories = np.unique(cat_id)
    per_cat = LLM_TRAIN_ROWS // len(categories)
    for cat in categories:
        rows = np.flatnonzero(cat_id == cat)
        values = target[rows]
        levels, counts = np.unique(values, return_counts=True)
        quota = np.floor(counts / counts.sum() * min(per_cat, len(rows))).astype(np.int64)
        deficit = min(per_cat, len(rows)) - int(quota.sum())
        quota[np.argsort(-counts)[:deficit]] += 1
        picked = [rng.choice(rows[values == level], size=min(int(q), int((values == level).sum())), replace=False)
                  for level, q in zip(levels, quota) if q]
        keep.append(np.concatenate(picked))
    return np.concatenate(keep)


def prepare_frames():
    human = pl.read_parquet(find_file(HUMAN_SPLIT_NAME)).to_pandas()
    cats = sorted(human["category"].unique().tolist())
    cat_to_id = {c: i for i, c in enumerate(cats)}
    human["category_id"] = human["category"].map(cat_to_id).astype(np.int16)
    human["true_target"] = human["label"].astype(np.float32)
    human["scope"] = np.int8(0)
    human_train = human[human["split"] == "train"].copy()
    human_val = human[human["split"] == "val"].copy()
    llm = pl.read_parquet(find_file(LLM_SPLIT_NAME)).to_pandas()
    llm["category_id"] = llm["category"].map(cat_to_id).astype(np.int16)
    llm_train = llm[llm["split"] == "train"].reset_index(drop=True)
    llm_val = llm[llm["split"] == "val"].copy()
    selected = stratified_llm_sample(llm_train["target"].to_numpy(), llm_train["category_id"].to_numpy())
    llm_train = llm_train.iloc[selected].copy()
    llm_train["true_target"] = llm_train["target"].astype(np.float32)
    llm_train["scope"] = np.int8(1)
    llm_val["true_target"] = llm_val["label"].astype(np.float32)
    llm_val["scope"] = np.int8(1)
    cols = ["id1", "id2", "true_target", "category_id", "scope"]
    train = pd.concat([human_train[cols], llm_train[cols]], ignore_index=True)
    log("DATA", f"train={len(train):,} (human={len(human_train):,}, llm={len(llm_train):,}), human_val={len(human_val):,}, llm_val={len(llm_val):,}")
    return train, human_val[cols], llm_val[cols], cats


def collect_texts(train, human_val, llm_val):
    ids = np.unique(np.concatenate([x[["id1", "id2"]].to_numpy().ravel() for x in (train, human_val, llm_val)]))
    needed = pl.DataFrame({"id": ids}).lazy()
    human_path, all_items_path = find_file("items_human.parquet"), find_file("items.parquet")
    human = pl.scan_parquet(human_path).join(needed, on="id", how="semi").select(["id", "name", "attributes"]).collect(engine="streaming")
    all_items = pl.scan_parquet(all_items_path).join(needed, on="id", how="semi").select(["id", "name", "attributes"]).collect(engine="streaming")
    merged = pl.concat([human, all_items]).unique(subset=["id"], keep="first")
    texts = {int(i): format_item(n, a) for i, n, a in zip(merged["id"].to_list(), merged["name"].to_list(), merged["attributes"].to_list())}
    if len(texts) != len(ids):
        raise RuntimeError(f"item text coverage {len(texts):,}/{len(ids):,}")
    log("DATA", f"resolved {len(texts):,} unique v4 texts")
    return ids.astype(np.int64), texts


def tokenize_berta(ids, texts):
    tokenizer = AutoTokenizer.from_pretrained(BERTA_MODEL)
    n = len(ids)
    bytes_needed = n * (8 + ITEM_MAX_LENGTH * 5)
    if shutil.disk_usage(WORK).free < int(bytes_needed * 1.15):
        raise RuntimeError("insufficient disk for BERTA cache")
    os.makedirs(OUT, exist_ok=True)
    ids_mm = np.memmap(os.path.join(OUT, "ids.bin"), dtype=np.int64, mode="w+", shape=(n,))
    tok_mm = np.memmap(os.path.join(OUT, "input_ids.bin"), dtype=np.int32, mode="w+", shape=(n, ITEM_MAX_LENGTH))
    mask_mm = np.memmap(os.path.join(OUT, "attention_mask.bin"), dtype=np.int8, mode="w+", shape=(n, ITEM_MAX_LENGTH))
    ids_mm[:] = ids
    batch = 20_000
    for start in range(0, n, batch):
        end = min(n, start + batch)
        encoded = tokenizer([texts[int(i)] for i in ids[start:end]], add_special_tokens=False, padding="max_length", truncation=True, max_length=ITEM_MAX_LENGTH, return_tensors="np")
        tok_mm[start:end] = encoded["input_ids"].astype(np.int32)
        mask_mm[start:end] = encoded["attention_mask"].astype(np.int8)
        if start == 0 or end == n or end % 500_000 < batch:
            log("TOKEN", f"{end:,}/{n:,}")
    for x in (ids_mm, tok_mm, mask_mm):
        x.flush()
    del ids_mm, tok_mm, mask_mm
    with open(os.path.join(OUT, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n_items": n, "max_length": ITEM_MAX_LENGTH, "model": BERTA_MODEL,
                   "add_special_tokens": False, "text_contract": "utils2.py v4: no prefix/category; name + reordered key = value attributes"}, f, ensure_ascii=False, indent=2)
    log("TOKEN", "BERTA v4 cache complete")


class FullSftHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2)), nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2)), nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))])
        self.out = nn.Linear(128, 1)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


class Teacher(nn.Module):
    def __init__(self, encoder, head):
        super().__init__(); self.encoder, self.head = encoder, head
    def forward(self, ids, mask):
        return self.head(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])


def fair_pair(a, b, pad):
    out_ids = np.full((len(a["input_ids"]), TEACHER_PAIR_MAX_LENGTH), pad, dtype=np.int64)
    out_mask = np.zeros_like(out_ids)
    for row in range(len(out_ids)):
        x, y = a["input_ids"][row, :a["attention_mask"][row].sum()], b["input_ids"][row, :b["attention_mask"][row].sum()]
        half = TEACHER_PAIR_MAX_LENGTH // 2
        if len(x) + len(y) > TEACHER_PAIR_MAX_LENGTH:
            if len(x) <= half: kx, ky = len(x), TEACHER_PAIR_MAX_LENGTH - len(x)
            elif len(y) <= half: kx, ky = TEACHER_PAIR_MAX_LENGTH - len(y), len(y)
            else: kx, ky = half, TEACHER_PAIR_MAX_LENGTH - half
            x, y = x[:kx], y[:ky]
        merged = np.concatenate((x, y)); out_ids[row, :len(merged)] = merged; out_mask[row, :len(merged)] = 1
    return out_ids, out_mask


def find_teacher_assets():
    for root, _dirs, files in os.walk("/kaggle/input"):
        if (os.path.basename(root) == TEACHER_MODEL_DIRNAME and
                {"config.json", "model.safetensors", "tokenizer.json"}.issubset(files)):
            head = os.path.join(os.path.dirname(os.path.dirname(root)), "classifier.pt")
            if os.path.isfile(head):
                return root, head
    raise FileNotFoundError("FRIDA model directory + sibling classifier.pt")


def teacher_worker(_index):
    rank, world = xr.global_ordinal(), xr.world_size()
    device = torch_xla.device()
    model_dir, head_path = _SHARED["teacher_model"], _SHARED["teacher_head"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = T5EncoderModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    head = FullSftHead(encoder.config.d_model); head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model = Teacher(encoder, head.to(dtype=torch.bfloat16)).to(device).eval()
    train = _SHARED["train"]
    id1, id2 = _SHARED["train_id1"], _SHARED["train_id2"]
    texts = _SHARED["texts"]
    score = np.memmap(os.path.join(OUT, "teacher_scores.bin"), dtype=np.float32, mode="r+", shape=(len(train),))
    rows = np.arange(rank, len(train), world, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(rows), TEACHER_BATCH):
            selected = rows[start:start + TEACHER_BATCH]
            valid = len(selected)
            if valid < TEACHER_BATCH:
                selected = np.pad(selected, (0, TEACHER_BATCH - valid), mode="edge")
            a = tokenizer([texts[int(i)] for i in id1[selected]], add_special_tokens=True, padding="max_length", truncation=True, max_length=TEACHER_ITEM_MAX_LENGTH, return_tensors="np")
            b = tokenizer([texts[int(i)] for i in id2[selected]], add_special_tokens=True, padding="max_length", truncation=True, max_length=TEACHER_ITEM_MAX_LENGTH, return_tensors="np")
            ids, mask = fair_pair(a, b, tokenizer.pad_token_id)
            logits = model(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device))
            xm.mark_step()
            score[selected[:valid]] = torch.sigmoid(logits.float()).cpu().numpy()[:valid]
            if start == 0 or start + TEACHER_BATCH >= len(rows) or (start // TEACHER_BATCH) % 1000 == 0:
                print(f"[TEACHER rank={rank}] {min(start + TEACHER_BATCH, len(rows)):,}/{len(rows):,}", flush=True)
    score.flush()


def main():
    os.makedirs(OUT, exist_ok=True)
    # Fail before the multi-hour tokenization if Kaggle did not mount teacher data.
    teacher_model, teacher_head = find_teacher_assets()
    log("PREFLIGHT", f"teacher model: {teacher_model}")
    train, human_val, llm_val, categories = prepare_frames()
    ids, texts = collect_texts(train, human_val, llm_val)
    tokenize_berta(ids, texts)
    train.to_parquet(os.path.join(OUT, "train_pairs.parquet"), index=False)
    human_val.to_parquet(os.path.join(OUT, "human_val_pairs.parquet"), index=False)
    llm_val.to_parquet(os.path.join(OUT, "llm_val_pairs.parquet"), index=False)
    with open(os.path.join(OUT, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False)
    np.memmap(os.path.join(OUT, "teacher_scores.bin"), dtype=np.float32, mode="w+", shape=(len(train),)).flush()
    _SHARED.update(train=train, train_id1=train["id1"].to_numpy(), train_id2=train["id2"].to_numpy(),
                   texts=texts, teacher_model=teacher_model, teacher_head=teacher_head)
    os.environ.pop("TPU_PROCESS_ADDRESSES", None); os.environ.pop("CLOUD_TPU_TASK_ID", None)
    torch_xla.launch(teacher_worker, args=(), start_method="fork")
    scores = np.memmap(os.path.join(OUT, "teacher_scores.bin"), dtype=np.float32, mode="r", shape=(len(train),))
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError("teacher score cache is not finite probabilities")
    with open(os.path.join(OUT, "run_contract.json"), "w", encoding="utf-8") as f:
        json.dump({"human_split": HUMAN_SPLIT_NAME, "llm_train_rows": LLM_TRAIN_ROWS, "seed": SEED,
                   "teacher_model_dir": TEACHER_MODEL_DIRNAME, "teacher_pair_max_length": TEACHER_PAIR_MAX_LENGTH,
                   "n_train": len(train), "n_items": len(ids), "teacher_score_mean": float(scores.mean())}, f, indent=2)
    del scores
    log("DONE", f"public cache ready: {OUT}")


if __name__ == "__main__":
    main()
