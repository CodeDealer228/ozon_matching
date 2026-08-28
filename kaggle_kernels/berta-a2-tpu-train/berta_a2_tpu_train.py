"""Matched TPU training arm for the BERTA A2 causal experiment.

Version 1 runs ARM=C1 (original-source replay).  After its output is safely
downloaded, the same file is changed only to ARM=T1 and pushed as version 2.
Both arms use the immutable paired manifest, identical row order, weights,
steps, seeds and evaluation schedule.
"""
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Kaggle's TPU image does not include PEFT and its bundled transformers is too
# old for this BERTA lineage.  Install the exact, repo-proven compatible set
# before module-level imports; --no-deps preserves the TPU torch/XLA stack.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
    "huggingface-hub==0.36.0", "transformers==4.57.6",
    "tokenizers==0.22.2", "peft==0.17.1",
])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file as safetensors_save_file
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_scheduler, set_seed

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl_xla
import torch_xla.runtime as xr

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as exc:
    print(f"[WARN] optional torchao dispatcher patch failed: {exc}", flush=True)


ARM = "C1"  # The only intended C1 -> T1 code change between kernel versions.
START = time.perf_counter()
WORKING = Path("/kaggle/working")
OUT = WORKING / f"berta_grounded_spec_tpu_{ARM.lower()}"
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_PAIRED_SHA = "b8a4b01d158eb3a3d3aafdc3391a231a876b8cd2a50abbd0d79fc8fc81a7d28a"
EXPECTED_VAL_SHA = "98e783670d63e7e43f493c2801dd82c069adf547fe1057be9cd5c7397c7da47f"
EXPECTED_SYNTHETIC_ITEMS_SHA = "2af60109cf71e045dd36b4b96c108c74eb0a267b18b602c6b14d875bf690aabd"
EXPECTED_ADAPTER_SHA = "26b294123b2445fa7bb46e0c36c2ee5ad259c218ea80e357661f49729800cd34"
EXPECTED_CLASSIFIER_SHA = "70ac231c64ad2f70ac73592afbce76c3ee9f0cb804ecb16e8b0f4258325d43fa"
EXPECTED_ROWS = 304_128
EXPECTED_VAL_ROWS = 109_648
EXPECTED_SYNTHETIC_ITEMS = 12_000
EXPECTED_STEPS = 594
GLOBAL_BATCH = 512
PER_CORE_TRAIN_BATCH = 64
PER_CORE_EVAL_BATCH = 128
PAIR_MAX_LENGTH = 256
ITEM_MAX_LENGTH = 256
PREFIX = "paraphrase: "
CATEGORY_PATTERN = np.asarray([14948, 30], dtype=np.int32)
ATTR_PATTERN = np.asarray([4816, 37519, 521, 30], dtype=np.int32)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.15
LORA_TARGETS = ["query", "key", "value", "dense"]
LORA_LR, CLASSIFIER_LR = 1e-4, 5e-5
WEIGHT_DECAY, WARMUP_RATIO = 0.04, 0.10
POS_WEIGHT = 2.894490893598893
MODEL_INIT_SEED = 1203
DATA_SEED = 420031
EVAL_STEPS = [0, 149, 297, 446, 594]
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 442
MAX_TRAIN_SECONDS = 28_800
SAFETY_SAVE_SECONDS = 3_600
_SHARED = {}


def log(rank, stage, message, force=False):
    if rank == 0 or force:
        print(f"[{time.perf_counter()-START:8.1f}s][r{rank}][{stage}] {message}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def find_file(name, expected_sha=None):
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            p = os.path.join(root, name)
            if expected_sha is None or sha256_file(p) == expected_sha:
                found.append(p)
    if len(found) != 1:
        raise RuntimeError(f"Expected one {name} sha={expected_sha}, found {found}")
    return found[0]


def find_merge_root():
    found = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "equivalence_report.json" in files and "merged_encoder" in dirs:
            found.append(root)
    if len(found) != 1:
        raise RuntimeError(f"Expected one verified second-stage merge root, found {found}")
    report = json.loads(Path(found[0], "equivalence_report.json").read_text(encoding="utf-8"))
    if not report.get("gate_passed"):
        raise RuntimeError(f"Merge gate is not passed: {report}")
    return found[0]


def find_human_scope():
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "cache_meta.json" in files and "human_scope" in root.lower():
            found.append(root)
    if len(found) != 1:
        raise RuntimeError(f"Expected one human_scope, found {found}")
    return found[0]


def pattern_starts(arr, pattern):
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    plen = len(pattern)
    for offset in range(arr.shape[1] - plen + 1):
        pending = starts < 0
        if not pending.any():
            break
        rows = np.flatnonzero(pending)
        hit = np.all(arr[rows, offset:offset+plen] == pattern[None, :], axis=1)
        starts[rows[hit]] = offset
    return starts


def load_human_cache():
    root = Path(find_human_scope())
    meta = json.loads((root / "cache_meta.json").read_text(encoding="utf-8"))
    if meta["max_length"] != ITEM_MAX_LENGTH or meta["prefix"] != PREFIX:
        raise RuntimeError(f"Human cache identity mismatch: {meta}")
    n = int(meta["n_items"])
    ids = np.memmap(root / "ids.bin", dtype=np.int64, mode="r", shape=(n,))
    tokens = np.memmap(root / "input_ids.bin", dtype=np.int32, mode="r", shape=(n, ITEM_MAX_LENGTH))
    attn = np.memmap(root / "attention_mask.bin", dtype=np.int8, mode="r", shape=(n, ITEM_MAX_LENGTH))
    lengths = np.asarray(attn).sum(1).astype(np.int16)
    return {
        "tokens": tokens, "lengths": lengths,
        "cat_start": pattern_starts(tokens, CATEGORY_PATTERN),
        "attr_start": pattern_starts(tokens, ATTR_PATTERN),
        "ids": ids, "id_to_row": {int(x): i for i, x in enumerate(ids)},
        "categories": np.load(root / "categories.npy"),
        "category_vocab": json.loads((root / "category_vocab.json").read_text(encoding="utf-8")),
        "meta": meta,
    }


def load_synthetic_cache(tokenizer):
    path = find_file("synthetic_items.parquet", EXPECTED_SYNTHETIC_ITEMS_SHA)
    frame = pd.read_parquet(path)
    if len(frame) != EXPECTED_SYNTHETIC_ITEMS or frame["id"].nunique() != len(frame):
        raise RuntimeError("Synthetic item identity mismatch")
    texts = (
        "Name: " + frame["name"].fillna("").astype(str).str.slice(0, MAX_NAME_CHARS)
        + " Category: " + frame["category"].fillna("").astype(str).str.slice(0, MAX_CATEGORY_CHARS)
        + " Attributes: " + frame["attributes"].fillna("").astype(str).str.slice(0, MAX_ATTR_CHARS)
    )
    encoded = tokenizer(
        [PREFIX + value for value in texts.tolist()], add_special_tokens=False,
        padding="max_length", truncation=True, max_length=ITEM_MAX_LENGTH,
        return_tensors="np",
    )
    tokens = encoded["input_ids"].astype(np.int32)
    lengths = encoded["attention_mask"].sum(1).astype(np.int16)
    return {
        "tokens": tokens, "lengths": lengths,
        "cat_start": pattern_starts(tokens, CATEGORY_PATTERN),
        "attr_start": pattern_starts(tokens, ATTR_PATTERN),
        "id_to_row": {int(value): index for index, value in enumerate(frame["id"])},
    }


def compact(cache, row):
    length = int(cache["lengths"][row])
    cs, ats = int(cache["cat_start"][row]), int(cache["attr_start"][row])
    if cs < 0 or ats < 0:
        return cache["tokens"][row, :length]
    return np.concatenate((cache["tokens"][row, :cs], cache["tokens"][row, ats:length]))


def allocate_pair_budget(a, b, budget):
    if a + b <= budget: return a, b
    half = budget // 2
    if a <= half and b <= half: return a, b
    if a <= half: return a, budget - a
    if b <= half: return budget - b, b
    return half, budget - half


class PairDataset(Dataset):
    def __init__(self, row1, row2, synthetic1, synthetic2, labels, weights, categories, valid):
        self.row1, self.row2 = row1, row2
        self.synthetic1, self.synthetic2 = synthetic1, synthetic2
        self.labels, self.weights, self.categories, self.valid = labels, weights, categories, valid

    def __len__(self): return len(self.row1)

    def __getitem__(self, i):
        return (self.row1[i], self.row2[i], self.synthetic1[i], self.synthetic2[i], self.labels[i],
                self.weights[i], self.categories[i], self.valid[i])


def collator(human, synthetic, cls_id, sep_id, pad_id):
    def collate(batch):
        bs = len(batch)
        ids = np.full((bs, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
        mask = np.zeros((bs, PAIR_MAX_LENGTH), dtype=np.int64)
        types = np.zeros((bs, PAIR_MAX_LENGTH), dtype=np.int64)
        labels, weights, categories, valid = [], [], [], []
        for i, (r1, r2, syn1, syn2, label, weight, cat, is_valid) in enumerate(batch):
            a = compact(synthetic if syn1 else human, int(r1))
            b = compact(synthetic if syn2 else human, int(r2))
            k1, k2 = allocate_pair_budget(len(a), len(b), PAIR_MAX_LENGTH - 3)
            n1, total = 1 + k1 + 1, 1 + k1 + 1 + k2 + 1
            ids[i, 0] = cls_id; ids[i, 1:1+k1] = a[:k1]; ids[i, 1+k1] = sep_id
            ids[i, n1:n1+k2] = b[:k2]; ids[i, n1+k2] = sep_id
            mask[i, :total] = 1; types[i, n1:total] = 1
            labels.append(label); weights.append(weight); categories.append(cat); valid.append(is_valid)
        return (torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(types),
                torch.tensor(labels, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32),
                np.asarray(categories, dtype=np.int64), np.asarray(valid, dtype=bool))
    return collate


class Head(nn.Module):
    def __init__(self, hidden):
        super().__init__(); self.dropout = nn.Dropout(0.10); self.linear = nn.Linear(hidden, 1)
    def forward(self, x): return self.linear(self.dropout(x))


class Model(nn.Module):
    def __init__(self, encoder, head): super().__init__(); self.encoder, self.head = encoder, head
    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = ((h*m).sum(1) / m.sum(1).clamp(min=1e-9)).float()
        return self.head(pooled).view(-1).float()


def load_head(path, hidden):
    head = Head(hidden); state = torch.load(path, map_location="cpu")
    if "linear.weight" in state: head.load_state_dict(state)
    else: head.linear.load_state_dict(state)
    return head


def build_model(device, merge_root, rank):
    set_seed(MODEL_INIT_SEED)
    encoder = AutoModel.from_pretrained(Path(merge_root, "merged_encoder"), torch_dtype=torch.bfloat16)
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                        target_modules=LORA_TARGETS, bias="none")
    encoder = get_peft_model(encoder, config)
    for name, param in encoder.named_parameters():
        if "lora_" in name: param.data = param.data.float()
    head = load_head(Path(merge_root, "classifier.pt"), encoder.config.hidden_size)
    model = Model(encoder, head).to(device)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or not any("lora_" in n for n, _ in trainable):
        raise RuntimeError("Fresh LoRA is not trainable")

    flat_sum = torch.stack([
        torch.stack([p.float().sum() for _n, p in trainable]).sum(),
        torch.stack([(p.float()*p.float()).sum() for _n, p in trainable]).sum(),
        torch.tensor(float(sum(p.numel() for _n, p in trainable)), device=device),
    ])
    gathered = xm.all_gather(flat_sum)
    xm.mark_step()
    matrix = gathered.detach().cpu().numpy().reshape(xr.world_size(), 3)
    if not np.allclose(matrix, matrix[0], rtol=0, atol=1e-5):
        raise RuntimeError(f"Cross-rank model initialization mismatch: {matrix.tolist()}")
    set_seed(DATA_SEED + rank)
    return model, {"trainable_count": int(sum(p.numel() for _n, p in trainable)),
                   "init_checksums": matrix.tolist()}


def make_loaders(rank, world):
    paired, val, human, synthetic = (_SHARED[k] for k in ("paired", "val", "human", "synthetic"))
    shard = paired.iloc[rank::world].reset_index(drop=True)
    if len(shard) != EXPECTED_STEPS * PER_CORE_TRAIN_BATCH:
        raise RuntimeError(f"Rank {rank} shard length {len(shard)}")
    if ARM == "T1":
        ids1, ids2 = shard["t1_id1"], shard["t1_id2"]
        syn1 = shard["t1_syn1"].to_numpy(bool)
        syn2 = shard["t1_syn2"].to_numpy(bool)
    else:
        ids1, ids2 = shard["c1_id1"], shard["c1_id2"]
        syn1 = np.zeros(len(shard), dtype=bool)
        syn2 = np.zeros(len(shard), dtype=bool)

    def resolve(values, flags):
        result = np.empty(len(values), dtype=np.int64)
        raw = values.to_numpy(np.int64)
        for is_synthetic, mapping in ((False, human["id_to_row"]), (True, synthetic["id_to_row"])):
            positions = np.flatnonzero(flags == is_synthetic)
            if len(positions):
                result[positions] = [mapping[int(raw[index])] for index in positions]
        return result

    r1, r2 = resolve(ids1, syn1), resolve(ids2, syn2)
    train_ds = PairDataset(r1, r2, syn1, syn2, shard["label"].to_numpy(np.float32),
                           shard["sample_weight"].to_numpy(np.float32),
                           np.full(len(shard), -1, dtype=np.int64), np.ones(len(shard), dtype=bool))
    cfn = collator(human, synthetic, _SHARED["cls_id"], _SHARED["sep_id"], _SHARED["pad_id"])
    train_loader = DataLoader(train_ds, batch_size=PER_CORE_TRAIN_BATCH, shuffle=False,
                              drop_last=True, num_workers=0, collate_fn=cfn)

    vr1 = val["id1"].map(human["id_to_row"]).to_numpy(np.int64)
    vr2 = val["id2"].map(human["id_to_row"]).to_numpy(np.int64)
    cats = human["categories"][vr1].astype(np.int64)
    pad = (-len(val)) % PER_CORE_EVAL_BATCH
    if pad:
        vr1 = np.concatenate([vr1, vr1[:pad]]); vr2 = np.concatenate([vr2, vr2[:pad]])
        labels = np.concatenate([val["label"].to_numpy(np.float32), val["label"].to_numpy(np.float32)[:pad]])
        cats = np.concatenate([cats, cats[:pad]]); valid = np.concatenate([np.ones(len(val), bool), np.zeros(pad, bool)])
    else:
        labels = val["label"].to_numpy(np.float32); valid = np.ones(len(val), bool)
    eval_ds = PairDataset(vr1, vr2, np.zeros(len(vr1), bool), np.zeros(len(vr1), bool),
                          labels, np.ones(len(vr1), np.float32), cats, valid)
    eval_loader = DataLoader(eval_ds, batch_size=PER_CORE_EVAL_BATCH, shuffle=False,
                             drop_last=True, num_workers=0, collate_fn=cfn)
    return train_loader, eval_loader


def compute_metrics(logits, labels, cats):
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    per_cat = {}
    for cid, name in enumerate(_SHARED["human"]["category_vocab"]):
        m = cats == cid
        if m.sum() and np.unique(labels[m]).size == 2:
            per_cat[name] = float(average_precision_score(labels[m], probs[m]))
    return {"macro_ap": float(np.mean(list(per_cat.values()))),
            "micro_ap": float(average_precision_score(labels, probs)),
            "per_category_ap": per_cat}


@torch.no_grad()
def evaluate(model, loader, device, rank, step):
    model.eval(); logits_all=[]; labels_all=[]; cats_all=[]
    mp = pl_xla.MpDeviceLoader(loader, device)
    for ids, mask, types, labels, _weights, cats, valid in mp:
        logits = model(ids, mask, types).float().cpu().numpy()
        logits_all.append(logits[valid]); labels_all.append(labels.cpu().numpy()[valid]); cats_all.append(cats[valid])
    xm.mark_step()
    logits = np.concatenate(logits_all); labels = np.concatenate(labels_all); cats = np.concatenate(cats_all)
    if len(logits) != EXPECTED_VAL_ROWS: raise RuntimeError(f"Eval rows {len(logits)}")
    result = compute_metrics(logits, labels, cats)
    if rank == 0:
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        np.save(OUT / f"val_logits_step_{step:04d}.npy", logits)
        frame = _SHARED["val"][["id1", "id2", "label"]].copy()
        frame["category_id"] = cats; frame["logit"] = logits; frame["probability"] = probs
        frame.to_parquet(OUT / f"val_predictions_step_{step:04d}.parquet", index=False)
        atomic_json(OUT / f"metrics_step_{step:04d}.json", {"arm": ARM, "step": step, **result})
    model.train(); return result


def cpu_checkpoint_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_checkpoint_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_checkpoint_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_checkpoint_tree(v) for v in value)
    return value


def save_checkpoint(model, optimizer, scheduler, step, metrics, rank, kind="eval"):
    if rank != 0: return
    xm.mark_step(); dest = OUT / f"checkpoint_step_{step:04d}_{kind}"; dest.mkdir(parents=True, exist_ok=True)
    adapter = dest / "adapter"; adapter.mkdir(exist_ok=True)
    state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model.encoder).items()}
    safetensors_save_file(state, adapter / "adapter_model.safetensors")
    model.encoder.peft_config["default"].save_pretrained(adapter)
    torch.save({k:v.detach().cpu() for k,v in model.head.state_dict().items()}, dest / "classifier.pt")
    torch.save(cpu_checkpoint_tree(optimizer.state_dict()), dest / "optimizer.pt")
    torch.save(cpu_checkpoint_tree(scheduler.state_dict()), dest / "scheduler.pt")
    if not (dest / "optimizer.pt").is_file() or not (dest / "scheduler.pt").is_file():
        raise RuntimeError(f"Checkpoint state save failed at step={step}")
    atomic_json(dest / "train_state.json", {"arm": ARM, "global_step": step,
                                             "metrics": metrics, "save_kind": kind})


def load_shared():
    merge_root = find_merge_root()
    input_identity = json.loads(Path(merge_root, "input_identity.json").read_text(encoding="utf-8"))
    if input_identity["adapter_sha256"] != EXPECTED_ADAPTER_SHA or input_identity["classifier_sha256"] != EXPECTED_CLASSIFIER_SHA:
        raise RuntimeError("Source checkpoint identity mismatch")
    paired_path = find_file("paired_train_manifest.parquet", EXPECTED_PAIRED_SHA)
    val_path = find_file("historical_val_manifest.parquet", EXPECTED_VAL_SHA)
    paired = pd.read_parquet(paired_path); val = pd.read_parquet(val_path)
    if len(paired) != EXPECTED_ROWS or len(val) != EXPECTED_VAL_ROWS: raise RuntimeError("Manifest row mismatch")
    if paired["global_row_index"].tolist() != list(range(EXPECTED_ROWS)): raise RuntimeError("Manifest order mismatch")
    if paired.groupby("global_step_index").size().nunique() != 1: raise RuntimeError("Ragged global batches")
    human = load_human_cache()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(Path(merge_root, "merged_encoder"))
    synthetic = load_synthetic_cache(tokenizer)
    human_ids = set(paired["c1_id1"]).union(paired["c1_id2"]).union(val["id1"]).union(val["id2"])
    human_ids.update(paired.loc[~paired["t1_syn1"], "t1_id1"])
    human_ids.update(paired.loc[~paired["t1_syn2"], "t1_id2"])
    missing = human_ids.difference(human["id_to_row"])
    if missing: raise RuntimeError(f"Human cache misses {len(missing)} ids")
    synthetic_ids = set(paired.loc[paired["t1_syn1"], "t1_id1"]).union(
        paired.loc[paired["t1_syn2"], "t1_id2"]
    )
    if synthetic_ids.difference(synthetic["id_to_row"]):
        raise RuntimeError("Synthetic cache misses referenced ids")
    _SHARED.update(merge_root=merge_root, paired=paired, val=val, human=human, synthetic=synthetic,
                   cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id)
    atomic_json(OUT / "preflight.json", {"status":"complete", "arm":ARM, "paired_sha256":EXPECTED_PAIRED_SHA,
                                         "val_sha256":EXPECTED_VAL_SHA, "rows":len(paired), "steps":EXPECTED_STEPS,
                                         "global_batch":GLOBAL_BATCH, "pair_max_length":PAIR_MAX_LENGTH,
                                         "source_identity":input_identity})


def replica(index):
    rank, world = xr.global_ordinal(), xr.world_size(); device = torch_xla.device()
    model, model_meta = build_model(device, _SHARED["merge_root"], rank)
    train_loader, eval_loader = make_loaders(rank, world)
    lora_params = [p for n,p in model.named_parameters() if p.requires_grad and "encoder" in n]
    head_params = [p for n,p in model.named_parameters() if p.requires_grad and "head" in n]
    optimizer = torch.optim.AdamW([{"params":lora_params,"lr":LORA_LR},
                                   {"params":head_params,"lr":CLASSIFIER_LR}], weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                              num_warmup_steps=max(1,int(EXPECTED_STEPS*WARMUP_RATIO)),
                              num_training_steps=EXPECTED_STEPS)
    if rank == 0:
        atomic_json(OUT / "training_meta.json", {"status":"running", "arm":ARM, "world_size":world,
                    "steps":EXPECTED_STEPS, "eval_steps":EVAL_STEPS, "per_core_batch":PER_CORE_TRAIN_BATCH,
                    "global_batch":GLOBAL_BATCH, "replay_weight":0.5, "pos_weight":POS_WEIGHT,
                    "lora_lr":LORA_LR, "classifier_lr":CLASSIFIER_LR, "weight_decay":WEIGHT_DECAY,
                    "model_init_seed":MODEL_INIT_SEED, "data_seed":DATA_SEED, **model_meta})
    initial = evaluate(model, eval_loader, device, rank, 0); save_checkpoint(model, optimizer, scheduler, 0, initial, rank)
    train_start=time.perf_counter(); last_safety=train_start; history=[{"step":0,**initial}]
    mp = pl_xla.MpDeviceLoader(train_loader, device)
    for step, batch in enumerate(mp, start=1):
        ids, mask, types, labels, weights, _cats, _valid = batch
        optimizer.zero_grad(); logits=model(ids,mask,types)
        per_row=F.binary_cross_entropy_with_logits(logits,labels,
                    pos_weight=torch.tensor(POS_WEIGHT,device=device),reduction="none")
        loss=(per_row*weights).mean(); loss.backward(); xm.optimizer_step(optimizer); scheduler.step()
        if step % 25 == 0: log(rank,"TRAIN",f"step={step}/{EXPECTED_STEPS} loss={loss.item():.5f}")
        if step in EVAL_STEPS:
            try:
                result=evaluate(model,eval_loader,device,rank,step); history.append({"step":step,**result})
                save_checkpoint(model,optimizer,scheduler,step,result,rank)
            except Exception as exc:
                log(rank,"EVAL",f"step={step} failed but training continues: {exc}",force=True)
                save_checkpoint(model,optimizer,scheduler,step,None,rank,"eval_failed")
                model.train()
        now=time.perf_counter()
        if now-last_safety >= SAFETY_SAVE_SECONDS:
            save_checkpoint(model,optimizer,scheduler,step,None,rank,"safety"); last_safety=now
        if now-train_start > MAX_TRAIN_SECONDS:
            raise TimeoutError(f"Training safety cutoff before complete step={step}")
    if rank==0:
        meta=json.loads((OUT/"training_meta.json").read_text()); meta.update(status="complete",history=history,
             final_step=EXPECTED_STEPS,elapsed_seconds=time.perf_counter()-START); atomic_json(OUT/"training_meta.json",meta)
        atomic_json(OUT/"run_status.json",{"status":"complete","arm":ARM,"final_step":EXPECTED_STEPS,
                                            "elapsed_seconds":time.perf_counter()-START})
    log(rank,"DONE",f"arm={ARM} complete",force=True)


def main():
    if ARM not in {"C1","T1"}: raise RuntimeError(f"Invalid ARM={ARM}")
    os.environ.pop("TPU_PROCESS_ADDRESSES",None); os.environ.pop("CLOUD_TPU_TASK_ID",None)
    atomic_json(OUT/"run_status.json",{"status":"running","arm":ARM,"phase":"preflight"})
    load_shared()
    torch_xla.launch(replica,args=(),start_method="fork")


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        atomic_json(OUT/"failure.json",{"status":"error","arm":ARM,"type":type(exc).__name__,
                    "message":str(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.perf_counter()-START})
        atomic_json(OUT/"run_status.json",{"status":"error","arm":ARM,"message":str(exc)})
        raise
