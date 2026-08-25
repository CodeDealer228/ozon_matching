"""TPU LoRA distillation of the formatting-v4 FRIDA teacher into BERTA.

Reads the immutable public output of berta-format-v4-tokenize-teacher.  The
real targets and per-sample weighting are the continue_c_llm3m recipe; teacher
probabilities are cached once there and contribute an equal focal-BCE term.
"""
import os
import sys
import gc
import json
import math
import time
import subprocess

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "peft", "transformers==5.15.0", "tokenizers==0.22.2"])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_scheduler, set_seed
from sklearn.metrics import average_precision_score
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla

SEED = 42
MODEL = "sergeyzh/BERTA"
ITEM_MAX_LENGTH = PAIR_MAX_LENGTH = 384
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
FOCAL_GAMMA = 2.0
HUMAN_WEIGHT = 5.86
PER_CORE_TRAIN_BATCH = 32
PER_CORE_EVAL_BATCH = 64
MAX_TRAIN_SECONDS = 30_000
QUIET_SAVE_SECONDS = 3_600
EXPERIMENT = "berta_format_v4_frida_distill_384"
WORK = "/kaggle/working"
OUT = os.path.join(WORK, EXPERIMENT)
_START = time.perf_counter()
_SHARED = {}


def log(rank, stage, msg, force=False):
    if rank == 0 or force:
        print(f"[{time.perf_counter() - _START:8.1f}s][rank{rank}] {stage}: {msg}", flush=True)


def find_root():
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"cache_meta.json", "train_pairs.parquet", "teacher_scores.bin"} <= set(files):
            return root
    raise RuntimeError("BERTA v4 token/teacher cache not found")


def allocate(len1, len2, budget):
    half = budget // 2
    if len1 <= half and len2 <= half: return len1, len2
    if len1 <= half: return len1, budget - len1
    if len2 <= half: return budget - len2, len2
    return half, budget - half


class PairDataset(Dataset):
    def __init__(self, frame, id_to_row, scores=None, indices=None, valid=None):
        self.r1 = frame["id1"].map(id_to_row).to_numpy(np.int64)
        self.r2 = frame["id2"].map(id_to_row).to_numpy(np.int64)
        self.target = frame["true_target"].to_numpy(np.float32)
        self.category = frame["category_id"].to_numpy(np.int64)
        self.scope = frame["scope"].to_numpy(np.int64)
        self.teacher = np.zeros(len(frame), np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
        self.indices = np.arange(len(frame)) if indices is None else indices
        self.valid = np.ones(len(self.indices), np.float32) if valid is None else valid
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        return self.r1[j], self.r2[j], self.target[j], self.teacher[j], self.category[j], self.scope[j], self.valid[i]


def collate(cache, cls_id, sep_id, pad_id):
    ids, lengths = cache["ids"], cache["lengths"]
    def fn(batch):
        n = len(batch); x = np.full((n, PAIR_MAX_LENGTH), pad_id, np.int64); m = np.zeros_like(x); typ = np.zeros_like(x)
        target, teacher, category, scope, valid = [], [], [], [], []
        for i, (r1, r2, y, t, c, s, v) in enumerate(batch):
            a, b = ids[r1, :lengths[r1]], ids[r2, :lengths[r2]]
            k1, k2 = allocate(len(a), len(b), PAIR_MAX_LENGTH - 3)
            a, b = a[:k1], b[:k2]; n1 = 1 + len(a) + 1; end = n1 + len(b) + 1
            x[i, 0] = cls_id; x[i, 1:1 + len(a)] = a; x[i, 1 + len(a)] = sep_id
            x[i, n1:n1 + len(b)] = b; x[i, n1 + len(b)] = sep_id; m[i, :end] = 1; typ[i, n1:end] = 1
            target.append(y); teacher.append(t); category.append(c); scope.append(s); valid.append(v)
        return tuple(map(torch.from_numpy, (x, m, typ, np.asarray(target, np.float32), np.asarray(teacher, np.float32), np.asarray(category, np.int64), np.asarray(scope, np.int64), np.asarray(valid, np.float32))))
    return fn


class Head(nn.Module):
    def __init__(self, dim): super().__init__(); self.dropout = nn.Dropout(0.1); self.linear = nn.Linear(dim, 1)
    def forward(self, x): return self.linear(self.dropout(x)).view(-1)


class Berta(nn.Module):
    def __init__(self, encoder, head): super().__init__(); self.encoder, self.head = encoder, head
    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype); return self.head(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float())


def build_model(device):
    base = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16)
    config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "key", "value", "dense"], bias="none")
    encoder = get_peft_model(base, config)
    for name, p in encoder.named_parameters():
        if "lora_" in name: p.data = p.data.float()
    if not any(p.requires_grad for n, p in encoder.named_parameters() if "lora_" in n):
        raise RuntimeError("LoRA is frozen")
    return Berta(encoder, Head(base.config.hidden_size)).to(device)


def focal(logits, targets):
    p = torch.sigmoid(logits); ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * p + (1 - targets) * (1 - p)
    return ((1 - pt).clamp(min=1e-6) ** FOCAL_GAMMA) * ce


def loss_fn(logits, targets, teacher, cats, scope, category_weight, category_pos_weight):
    real = focal(logits, targets); distilled = focal(logits, teacher)
    w_pos = 1 + (category_pos_weight[cats] - 1) * targets
    w = category_weight[cats] * torch.where(scope == 0, torch.full_like(real, HUMAN_WEIGHT), torch.ones_like(real)) * w_pos
    return (((real + distilled) * 0.5) * w).sum() / w.sum()


def make_eval_loader(frame, cache, id_to_row, cls_id, sep_id, pad_id, rank, world):
    dataset = PairDataset(frame, id_to_row)
    per_rank = math.ceil(len(dataset) / (world * PER_CORE_EVAL_BATCH)) * PER_CORE_EVAL_BATCH
    all_idx = np.pad(np.arange(len(dataset)), (0, per_rank * world - len(dataset)), mode="edge")
    valid = np.zeros(per_rank * world, np.float32); valid[:len(dataset)] = 1
    start = rank * per_rank
    shard = PairDataset(frame, id_to_row, indices=all_idx[start:start + per_rank], valid=valid[start:start + per_rank])
    return DataLoader(shard, batch_size=PER_CORE_EVAL_BATCH, drop_last=True, shuffle=False, collate_fn=collate(cache, cls_id, sep_id, pad_id), num_workers=2)


def evaluate(model, loader, device, categories):
    model.eval(); logits_all=[]; y_all=[]; c_all=[]; valid_all=[]
    with torch.no_grad():
        for x, m, typ, y, _t, cat, _s, valid in pl_xla.MpDeviceLoader(loader, device):
            logits_all.append(model(x, m, typ)); y_all.append(y); c_all.append(cat); valid_all.append(valid)
    logits = xm.all_gather(torch.cat(logits_all), dim=0); y = xm.all_gather(torch.cat(y_all), dim=0); c = xm.all_gather(torch.cat(c_all), dim=0); valid = xm.all_gather(torch.cat(valid_all), dim=0)
    xm.mark_step(); keep = valid.cpu().numpy() > .5; p = torch.sigmoid(logits).cpu().numpy()[keep]; y = y.cpu().numpy()[keep]; c = c.cpu().numpy()[keep]
    per = {name: float(average_precision_score(y[c == i], p[c == i])) for i, name in enumerate(categories) if len(np.unique(y[c == i])) > 1}
    model.train(); return float(np.mean(list(per.values()))), per


def save(model, optimizer, scheduler, epoch, step, metric, label, rank, result):
    if rank: return
    root = os.path.join(OUT, label); os.makedirs(os.path.join(root, "adapter"), exist_ok=True); xm.mark_step()
    sd = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model.encoder).items()}; save_file(sd, os.path.join(root, "adapter", "adapter_model.safetensors")); model.encoder.peft_config["default"].save_pretrained(os.path.join(root, "adapter"))
    torch.save({k: v.detach().cpu() for k, v in model.head.state_dict().items()}, os.path.join(root, "classifier.pt"))
    xm.save(optimizer.state_dict(), os.path.join(root, "optimizer.pt")); torch.save(scheduler.state_dict(), os.path.join(root, "scheduler.pt"))
    state = {"experiment": EXPERIMENT, "epoch": epoch, "global_step": step, "val_macro_pr_auc": metric, "max_pair_length": PAIR_MAX_LENGTH, "loss": "0.5 real focal-BCE + 0.5 cached FRIDA focal-BCE"}
    with open(os.path.join(root, "train_state.json"), "w") as f: json.dump(state, f, indent=2)
    result["last"] = state
    with open(os.path.join(OUT, "run_results.json"), "w") as f: json.dump(result, f, ensure_ascii=False, indent=2)


def load_shared():
    root = find_root()
    meta = json.load(open(os.path.join(root, "cache_meta.json")))
    if meta["max_length"] != ITEM_MAX_LENGTH or meta.get("add_special_tokens") is not False: raise RuntimeError("wrong BERTA cache contract")
    n = int(meta["n_items"]); ids = np.memmap(os.path.join(root, "ids.bin"), np.int64, mode="r", shape=(n,)); tok = np.memmap(os.path.join(root, "input_ids.bin"), np.int32, mode="r", shape=(n, ITEM_MAX_LENGTH)); attn = np.memmap(os.path.join(root, "attention_mask.bin"), np.int8, mode="r", shape=(n, ITEM_MAX_LENGTH))
    train = pd.read_parquet(os.path.join(root, "train_pairs.parquet")); val = pd.read_parquet(os.path.join(root, "human_val_pairs.parquet")); scores = np.memmap(os.path.join(root, "teacher_scores.bin"), np.float32, mode="r", shape=(len(train),))
    if not np.isfinite(scores).all(): raise RuntimeError("teacher score cache has NaN")
    cats = json.load(open(os.path.join(root, "categories.json")))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    count = np.bincount(train["category_id"], minlength=len(cats)).astype(float); pos = np.bincount(train["category_id"], weights=train["true_target"], minlength=len(cats)); cat_w = (count.sum() / len(cats)) / np.maximum(count, 1); pos_w = (count - pos) / np.maximum(pos, 1e-6)
    _SHARED.update(cache={"ids": tok, "lengths": np.asarray(attn).sum(1).astype(np.int32)}, id_to_row={int(x): i for i, x in enumerate(ids)}, train=train, val=val, scores=scores, categories=cats, cat_w=cat_w, pos_w=pos_w, cls=tokenizer.cls_token_id, sep=tokenizer.sep_token_id, pad=tokenizer.pad_token_id)
    os.makedirs(OUT, exist_ok=True); log(0, "DATA", f"cache={n:,} items train={len(train):,} val={len(val):,}")


def worker(_index):
    rank, world, device = xr.global_ordinal(), xr.world_size(), torch_xla.device(); set_seed(SEED + rank)
    model = build_model(device); params = [p for p in model.parameters() if p.requires_grad]; total_batch = PER_CORE_TRAIN_BATCH * world; steps_per_epoch = len(_SHARED["train"]) // total_batch; scheduler_steps = steps_per_epoch * EPOCHS
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY); scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=max(1, int(scheduler_steps * WARMUP_RATIO)), num_training_steps=scheduler_steps)
    cat_w = torch.tensor(_SHARED["cat_w"], device=device); pos_w = torch.tensor(_SHARED["pos_w"], device=device)
    val_loader = make_eval_loader(_SHARED["val"], _SHARED["cache"], _SHARED["id_to_row"], _SHARED["cls"], _SHARED["sep"], _SHARED["pad"], rank, world)
    result = {"config": {"epochs": EPOCHS, "lr": LR, "pair_max_length": PAIR_MAX_LENGTH, "train_rows": len(_SHARED["train"]), "teacher_cached": True}, "history": []}; best=-1.; step=0; started=time.perf_counter(); last_save=0.
    log(rank, "CONFIG", f"world={world} batch/core={PER_CORE_TRAIN_BATCH} seq={PAIR_MAX_LENGTH} epochs={EPOCHS} LR={LR}", True)
    for epoch in range(1, EPOCHS + 1):
        dataset = PairDataset(_SHARED["train"], _SHARED["id_to_row"], _SHARED["scores"]); sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=SEED, drop_last=True); sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=PER_CORE_TRAIN_BATCH, sampler=sampler, drop_last=True, collate_fn=collate(_SHARED["cache"], _SHARED["cls"], _SHARED["sep"], _SHARED["pad"]), num_workers=2)
        for x, m, typ, y, teacher, cats, scope, _valid in pl_xla.MpDeviceLoader(loader, device):
            optimizer.zero_grad(set_to_none=True); logits=model(x,m,typ); loss=loss_fn(logits,y,teacher,cats,scope,cat_w,pos_w)
            if not torch.isfinite(loss): raise RuntimeError("non-finite distillation loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); xm.optimizer_step(optimizer); scheduler.step(); step += 1
            elapsed=time.perf_counter()-started
            if step % 100 == 0 and rank == 0: xm.mark_step(); log(rank,"TRAIN",f"epoch={epoch} step={step} loss={loss.item():.5f} lr={scheduler.get_last_lr()[0]:.2e}")
            if elapsed-last_save >= QUIET_SAVE_SECONDS: save(model,optimizer,scheduler,epoch,step,None,"last",rank,result); last_save=elapsed
            if elapsed >= MAX_TRAIN_SECONDS: save(model,optimizer,scheduler,epoch,step,None,"last",rank,result); return
        metric, per = evaluate(model,val_loader,device,_SHARED["categories"]); improved=metric>best; best=max(best,metric); result["history"].append({"epoch":epoch,"global_step":step,"val_macro_pr_auc":metric,"per_category":per}); save(model,optimizer,scheduler,epoch,step,metric,f"checkpoint_epoch{epoch:02d}",rank,result)
        if improved: save(model,optimizer,scheduler,epoch,step,metric,"best",rank,result)
        log(rank,"METRICS",f"epoch={epoch} val={metric:.5f} best={best:.5f}",True)
        if metric >= .76: return


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None); os.environ.pop("CLOUD_TPU_TASK_ID", None); load_shared(); torch_xla.launch(worker, args=(), start_method="fork")


if __name__ == "__main__": main()
