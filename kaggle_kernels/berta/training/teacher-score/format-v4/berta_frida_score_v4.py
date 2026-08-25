"""Score the existing BERTA-v4 training pairs once with the FRIDA teacher."""
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
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

ITEM_MAX_LENGTH = PAIR_MAX_LENGTH = 188
PER_CORE_BATCH = 32  # fixed conservative batch; no TPU probing
MAX_WALL_SECONDS = 7 * 3600 + 30 * 60
MAX_NAME_CHARS, MAX_ATTR_CHARS = 160, 640
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "frida_teacher_scores_v4")
TEMP = os.path.join(WORK, "frida_teacher_items_v4")
_START = time.perf_counter()
_SHARED = {}


def log(stage, message):
    print(f"[{time.perf_counter() - _START:8.1f}s] {stage}: {message}", flush=True)


def find_file(name):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def find_teacher_assets():
    for root, _dirs, files in os.walk("/kaggle/input"):
        if (os.path.basename(root) == "frida-format-adapter-v4" and
                {"config.json", "model.safetensors", "tokenizer.json"}.issubset(files)):
            head = os.path.join(os.path.dirname(os.path.dirname(root)), "classifier.pt")
            if os.path.isfile(head):
                return root, head
    raise FileNotFoundError("FRIDA model directory + sibling classifier.pt")


def clean(value):
    return " ".join(("" if value is None else str(value)).split()).replace(" = ", "=")


def rank_value(key, value):
    value = value if isinstance(value, str) else str(value)
    low = value.lower()
    if "oem" in low or "oem" in str(key).lower():
        return 3
    if "https" in low:
        return 2
    return int(len(value) > 200)


def format_item(name, attributes):
    try:
        attrs = json.loads(attributes) if attributes else {}
    except Exception:
        attrs = {}
    lines = []
    if isinstance(attrs, dict):
        for key, value in sorted(attrs.items(), key=lambda item: rank_value(*item)):
            key, value = clean(key), clean(value)
            if key or value:
                lines.append(f"{key} = {value}")
    attrs = "\n".join(lines)[:MAX_ATTR_CHARS]
    name = clean(name)[:MAX_NAME_CHARS]
    return f"{name}\n{attrs}" if attrs else name


def load_texts(train):
    ids = np.unique(train[["id1", "id2"]].to_numpy(dtype=np.int64).ravel())
    needed = pl.DataFrame({"id": ids}).lazy()
    frames = []
    for filename in ("items_human.parquet", "items.parquet"):
        frames.append(pl.scan_parquet(find_file(filename)).join(needed, on="id", how="semi")
                      .select(["id", "name", "attributes"]).collect(engine="streaming"))
    items = pl.concat(frames).unique(subset=["id"], keep="first")
    texts = {int(item_id): format_item(name, attrs)
             for item_id, name, attrs in zip(items["id"].to_list(), items["name"].to_list(), items["attributes"].to_list())}
    if len(texts) != len(ids):
        raise RuntimeError(f"item coverage {len(texts):,}/{len(ids):,}")
    log("DATA", f"resolved {len(ids):,} v4 item texts")
    return ids, texts


def tokenize_teacher_items(ids, texts, model_dir):
    bytes_needed = len(ids) * (8 + ITEM_MAX_LENGTH * 4 + 2)
    if shutil.disk_usage(WORK).free < int(bytes_needed * 1.25):
        raise RuntimeError("insufficient working disk for temporary teacher tokens")
    os.makedirs(TEMP, exist_ok=True)
    id_mm = np.memmap(os.path.join(TEMP, "ids.bin"), dtype=np.int64, mode="w+", shape=(len(ids),))
    tok_mm = np.memmap(os.path.join(TEMP, "input_ids.bin"), dtype=np.int32, mode="w+", shape=(len(ids), ITEM_MAX_LENGTH))
    len_mm = np.memmap(os.path.join(TEMP, "lengths.bin"), dtype=np.int16, mode="w+", shape=(len(ids),))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id_mm[:] = ids
    for start in range(0, len(ids), 20_000):
        end = min(start + 20_000, len(ids))
        encoded = tokenizer([texts[int(item_id)] for item_id in ids[start:end]], add_special_tokens=True,
                            padding="max_length", truncation=True, max_length=ITEM_MAX_LENGTH, return_tensors="np")
        tok_mm[start:end] = encoded["input_ids"].astype(np.int32)
        len_mm[start:end] = encoded["attention_mask"].sum(axis=1).astype(np.int16)
        if start == 0 or end == len(ids) or end % 500_000 < 20_000:
            log("TOKEN", f"teacher items {end:,}/{len(ids):,}")
    for mm in (id_mm, tok_mm, len_mm):
        mm.flush()
    return tokenizer.pad_token_id


class FullSftHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2)),
        ])
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


def build_pairs(tokens, lengths, idx1, idx2, pad):
    out = np.full((len(idx1), PAIR_MAX_LENGTH), pad, dtype=np.int64)
    mask = np.zeros_like(out)
    first, second = tokens[idx1], tokens[idx2]
    for row, (left, right) in enumerate(zip(lengths[idx1], lengths[idx2])):
        left, right = int(left), int(right)
        if left + right <= PAIR_MAX_LENGTH:
            keep1, keep2 = left, right
        elif left <= PAIR_MAX_LENGTH // 2:
            keep1, keep2 = left, PAIR_MAX_LENGTH - left
        elif right <= PAIR_MAX_LENGTH // 2:
            keep1, keep2 = PAIR_MAX_LENGTH - right, right
        else:
            keep1 = PAIR_MAX_LENGTH // 2
            keep2 = PAIR_MAX_LENGTH - keep1
        merged = np.concatenate((first[row, :keep1], second[row, :keep2]))
        out[row, :len(merged)] = merged
        mask[row, :len(merged)] = 1
    return out, mask


def score_worker(_):
    rank, world, device = xr.global_ordinal(), xr.world_size(), torch_xla.device()
    model_dir, head_path = _SHARED["model_dir"], _SHARED["head_path"]
    encoder = T5EncoderModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    head = FullSftHead(encoder.config.d_model)
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model = nn.Sequential()  # keep the two modules registered on the TPU device
    model.encoder, model.head = encoder, head.to(dtype=torch.bfloat16)
    model.to(device).eval()
    tokens = np.memmap(os.path.join(TEMP, "input_ids.bin"), dtype=np.int32, mode="r",
                       shape=(_SHARED["n_items"], ITEM_MAX_LENGTH))
    lengths = np.memmap(os.path.join(TEMP, "lengths.bin"), dtype=np.int16, mode="r", shape=(_SHARED["n_items"],))
    score = np.memmap(os.path.join(OUT, "teacher_scores.bin"), dtype=np.float32, mode="r+", shape=(_SHARED["n_train"],))
    rows = np.arange(rank, _SHARED["n_train"], world, dtype=np.int64)
    started = time.monotonic()
    # XLA mutates tensor version counters during lazy graph execution;
    # inference_mode forbids that and crashes on the first T5 Linear.
    with torch.no_grad():
        for start in range(0, len(rows), PER_CORE_BATCH):
            selected = rows[start:start + PER_CORE_BATCH]
            valid = len(selected)
            if valid < PER_CORE_BATCH:
                selected = np.pad(selected, (0, PER_CORE_BATCH - valid), mode="edge")
            pair_ids, pair_mask = build_pairs(tokens, lengths, _SHARED["idx1"][selected], _SHARED["idx2"][selected], _SHARED["pad"])
            hidden = model.encoder(input_ids=torch.from_numpy(pair_ids).to(device), attention_mask=torch.from_numpy(pair_mask).to(device)).last_hidden_state[:, 0, :]
            probs = torch.sigmoid(model.head(hidden).float()).cpu().numpy()[:valid]
            xm.mark_step()
            score[selected[:valid]] = probs
            if start == 0 or start + PER_CORE_BATCH >= len(rows) or start % (5000 * PER_CORE_BATCH) == 0:
                score.flush()
                with open(os.path.join(OUT, f"progress_rank_{rank}.json"), "w") as f:
                    json.dump({"done": min(start + PER_CORE_BATCH, len(rows)), "total": len(rows)}, f)
                print(f"[SCORE rank={rank}] {min(start + PER_CORE_BATCH, len(rows)):,}/{len(rows):,}", flush=True)
            if time.monotonic() - _SHARED["wall_start"] >= MAX_WALL_SECONDS:
                raise TimeoutError("controlled wall-clock stop; partial scores remain NaN")
    score.flush()


def main():
    os.makedirs(OUT, exist_ok=True)
    model_dir, head_path = find_teacher_assets()
    train_path = find_file("train_pairs.parquet")
    train = pd.read_parquet(train_path)
    required = {"id1", "id2", "true_target", "category_id", "scope"}
    if len(train) != 3_256_006 or set(train.columns) != required:
        raise RuntimeError("unexpected BERTA v4 train-pairs contract")
    log("PREFLIGHT", f"teacher={model_dir}; train={len(train):,}")
    ids, texts = load_texts(train)
    pad = tokenize_teacher_items(ids, texts, model_dir)
    del texts
    gc.collect()
    pair_ids = train[["id1", "id2"]].to_numpy(dtype=np.int64)
    idx1, idx2 = np.searchsorted(ids, pair_ids[:, 0]).astype(np.int32), np.searchsorted(ids, pair_ids[:, 1]).astype(np.int32)
    if not (np.array_equal(ids[idx1], pair_ids[:, 0]) and np.array_equal(ids[idx2], pair_ids[:, 1])):
        raise RuntimeError("pair id lookup mismatch")
    score = np.memmap(os.path.join(OUT, "teacher_scores.bin"), dtype=np.float32, mode="w+", shape=(len(train),))
    score[:] = np.nan
    score.flush()
    pair_hash = hashlib.sha256(np.ascontiguousarray(pair_ids).tobytes()).hexdigest()
    _SHARED.update(model_dir=model_dir, head_path=head_path, n_items=len(ids), n_train=len(train), idx1=idx1,
                   idx2=idx2, pad=pad, wall_start=time.monotonic())
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)
    torch_xla.launch(score_worker, args=(), start_method="fork")
    score = np.memmap(os.path.join(OUT, "teacher_scores.bin"), dtype=np.float32, mode="r", shape=(len(train),))
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise RuntimeError("incomplete or invalid teacher score cache")
    contract = {"source_train_pairs": "maxamsecret/berta-tokenized-chache-v4-format", "n_train": len(train),
                "n_items": len(ids), "pair_sha256": pair_hash, "teacher_pair_max_length": PAIR_MAX_LENGTH,
                "teacher_score_mean": float(score.mean()), "teacher_score_std": float(score.std())}
    with open(os.path.join(OUT, "run_contract.json"), "w") as f:
        json.dump(contract, f, indent=2)
    shutil.rmtree(TEMP)
    log("DONE", json.dumps(contract))


if __name__ == "__main__":
    main()
