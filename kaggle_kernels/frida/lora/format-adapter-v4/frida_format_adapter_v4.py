"""LoRA-адаптация FRIDA continue_c `last` к компактному item-format v4.

Base encoder и FullSftHead загружаются из continue_c_llm3m_2ep/last.
Обучаются только небольшие q/v LoRA; head заморожен и всегда работает в eval.
Данные и loss совпадают с experiment C: human item-disjoint train + ровно
3M LLM train (seed=42, category/target stratified), focal gamma=2,
human_weight=5.86, category и within-category positive-rate weights.
"""

import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# TPU v5e-8 host: 224 vCPU shared by 8 rank processes -> 28 threads/rank.
os.environ.setdefault("OMP_NUM_THREADS", "28")
os.environ.setdefault("MKL_NUM_THREADS", "28")
os.environ.setdefault("RAYON_NUM_THREADS", "28")

# Kaggle TPU image does not include these packages. Install before their
# first import; internet is enabled in kernel-metadata.json.
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "polars", "peft"]
)

import numpy as np
import polars as pl
import torch
import torch.nn as nn

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as error:
    print(f"[WARN] torchao patch unavailable: {error}", flush=True)

from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file as save_safetensors
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Config, T5EncoderModel, get_scheduler, set_seed

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl_xla
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla import runtime as xr


SEED = 42
MODEL_NAME = "ai-forever/FRIDA"
EXPERIMENT_NAME = "frida_format_adapter_v4_c_last"
SOURCE_EXPERIMENT = "continue_c_llm3m_2ep"
SOURCE_EXPECTED_STEP = 13_474

ITEM_SPLIT_FILE = "matches_item_disjoint_70_30_seed42.parquet"
LLM_SPLIT_FILE = "llm_train_val_split.parquet"
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 224

HEAD_HIDDEN = (1024, 512, 128)
HEAD_DROPOUT = 0.2

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "v"]
LR = 7e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
FOCAL_GAMMA = 2.0
HUMAN_WEIGHT = 5.86

LLM_TRAIN_SUBSAMPLE_SIZE = 3_000_000
EPOCHS = 1
# Smoke-search bounds. The fallback 16/32 is compiled first, then at most
# three binary-search shapes. On the first OOM the search stops
# because failed XLA programs can leave HBM fragmented/resident.
MIN_PER_CORE_TRAIN_BATCH = 16
MAX_PER_CORE_TRAIN_BATCH = 48
MIN_PER_CORE_EVAL_BATCH = 32
MAX_PER_CORE_EVAL_BATCH = 96
BATCH_SMOKE_ITERATIONS = 3
HBM_LOG_EVERY_STEPS = 500
VALIDATE_EVERY_ROWS = 100_000
QUALIFIED_SCORE = 0.758
LOW_SCORE = 0.700
MAX_TRAIN_SECONDS = 30_600
LOG_EVERY_STEPS = 50

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_format_adapter_v4")

_SHARED = {}
_WRAPPED_MODEL = None
_START = time.perf_counter()


def log(rank, stage, message, force=False):
    if rank == 0 or force:
        print(f"[{time.perf_counter() - _START:8.1f}s][rank{rank}] {stage}: {message}", flush=True)


def find_file(filename):
    if not os.path.isdir("/kaggle/input"):
        raise FileNotFoundError("/kaggle/input is unavailable")
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            found.append(os.path.join(root, filename))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {filename!r}, found {found}")
    return found[0]


def find_source_checkpoint():
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"encoder.pt", "head.pt", "train_state.json"} <= set(files):
            with open(os.path.join(root, "train_state.json"), encoding="utf-8") as f:
                state = json.load(f)
            if (state.get("experiment") == SOURCE_EXPERIMENT and
                    int(state.get("global_step", -1)) == SOURCE_EXPECTED_STEP and
                    os.path.basename(root) == "last"):
                candidates.append((root, state))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one c_last step={SOURCE_EXPECTED_STEP}, got {candidates}")
    return candidates[0]


def find_resume_checkpoint():
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "train_state.json" not in files:
            continue
        with open(os.path.join(root, "train_state.json"), encoding="utf-8") as f:
            state = json.load(f)
        adapter = os.path.join(root, "adapter")
        if (state.get("experiment") == EXPERIMENT_NAME and
                os.path.isfile(os.path.join(adapter, "adapter_model.safetensors"))):
            candidates.append((root, adapter, state))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (int(item[2].get("global_step", 0)),
                                      os.path.basename(item[0]) == "last"), reverse=True)
    return candidates[0]


def find_cache(scope):
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"cache_meta.json", "ids.bin", "input_ids.bin", "attention_mask.bin"} <= set(files):
            with open(os.path.join(root, "cache_meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("format_version") == 4 and meta.get("scope") == scope:
                candidates.append((root, meta))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one v4 cache for {scope}, got {candidates}")
    root, meta = candidates[0]
    n = int(meta["n_items"])
    max_len = int(meta["max_length"])
    if max_len != ITEM_MAX_LENGTH or meta.get("prefix") != "" or meta.get("category_in_text") is not False:
        raise RuntimeError(f"Wrong v4 cache contract: {meta}")
    ids = np.memmap(os.path.join(root, "ids.bin"), np.int64, "r", shape=(n,))
    input_ids = np.memmap(os.path.join(root, "input_ids.bin"), np.int32, "r", shape=(n, max_len))
    mask = np.memmap(os.path.join(root, "attention_mask.bin"), np.int8, "r", shape=(n, max_len))
    lengths = np.asarray(mask.sum(axis=1), dtype=np.int16)
    order = np.argsort(ids)
    sorted_ids = np.asarray(ids)[order]
    if len(np.unique(sorted_ids)) != n:
        raise RuntimeError(f"Duplicate ids in {scope} v4 cache")
    candidates_pad = np.flatnonzero(lengths < max_len)
    if len(candidates_pad) == 0:
        raise RuntimeError(f"Cannot derive pad id from {scope} cache")
    sample = candidates_pad[::max(1, len(candidates_pad) // 200)][:200]
    pads = {int(input_ids[i, int(lengths[i])]) for i in sample}
    if len(pads) != 1:
        raise RuntimeError(f"Inconsistent pad ids in {scope}: {pads}")
    return {"scope": scope, "ids": ids, "sorted_ids": sorted_ids, "order": order,
            "input_ids": input_ids, "lengths": lengths, "pad_id": pads.pop(), "meta": meta}


def lookup_rows(cache, item_ids):
    item_ids = np.asarray(item_ids, dtype=np.int64)
    pos = np.searchsorted(cache["sorted_ids"], item_ids)
    valid = pos < len(cache["sorted_ids"])
    valid[valid] &= cache["sorted_ids"][pos[valid]] == item_ids[valid]
    rows = np.full(len(item_ids), -1, dtype=np.int64)
    rows[valid] = cache["order"][pos[valid]]
    return rows


def allocate_pair_budget(len1, len2, budget):
    half = budget // 2
    if len1 <= half:
        return len1, min(len2, budget - len1)
    if len2 <= half:
        return min(len1, budget - len2), len2
    return half, budget - half


class PairDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame

    def __len__(self):
        return len(self.frame["label"])

    def __getitem__(self, index):
        return tuple(self.frame[key][index] for key in ("scope", "row1", "row2", "label", "category"))


def make_collate(caches, pad_id):
    def collate(batch):
        scopes, rows1, rows2, labels, categories = zip(*batch)
        ids = np.full((len(batch), PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), PAIR_MAX_LENGTH), dtype=np.int64)
        for i, (scope, row1, row2) in enumerate(zip(scopes, rows1, rows2)):
            cache = caches[int(scope)]
            len1, len2 = int(cache["lengths"][row1]), int(cache["lengths"][row2])
            keep1, keep2 = allocate_pair_budget(len1, len2, PAIR_MAX_LENGTH)
            ids[i, :keep1] = cache["input_ids"][row1, :keep1]
            ids[i, keep1:keep1 + keep2] = cache["input_ids"][row2, :keep2]
            mask[i, :keep1 + keep2] = 1
        return (torch.from_numpy(ids), torch.from_numpy(mask),
                torch.tensor(labels, dtype=torch.float32),
                torch.tensor(categories, dtype=torch.int64),
                torch.tensor(scopes, dtype=torch.int64))
    return collate


def make_train_loader(frame, caches, pad_id, world_size, rank, per_core_batch):
    dataset = PairDataset(frame)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=SEED, drop_last=True)
    sampler.set_epoch(1)
    loader = DataLoader(dataset, batch_size=per_core_batch, sampler=sampler, drop_last=True,
                        collate_fn=make_collate(caches, pad_id), num_workers=2)
    return loader, sampler


def make_eval_loader(frame, caches, pad_id, world_size, rank, per_core_batch):
    n = len(frame["label"])
    per_rank = max(per_core_batch, math.ceil(n / (world_size * per_core_batch)) * per_core_batch)
    total = per_rank * world_size
    indices = np.arange(total) % n
    valid = np.arange(total) < n
    start = rank * per_rank
    selected = indices[start:start + per_rank]
    shard = {key: values[selected] for key, values in frame.items()}
    loader = DataLoader(PairDataset(shard), batch_size=per_core_batch, shuffle=False,
                        drop_last=True, collate_fn=make_collate(caches, pad_id), num_workers=2)
    return loader, valid[start:start + per_rank]


class FullSftHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        blocks, width = [], hidden_dim
        for hidden in HEAD_HIDDEN:
            blocks.append(nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden),
                                        nn.GELU(), nn.Dropout(HEAD_DROPOUT)))
            width = hidden
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


class FormatAdapterModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()
        return self

    def forward(self, input_ids, attention_mask):
        with torch.autocast("xla", dtype=torch.bfloat16):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.classifier(output.last_hidden_state[:, 0, :])
        return logits.float()


def build_model_cpu(checkpoint_dir, resume):
    config = T5Config.from_pretrained(MODEL_NAME)
    encoder = T5EncoderModel(config).to(dtype=torch.bfloat16)
    encoder_state = torch.load(os.path.join(checkpoint_dir, "encoder.pt"), map_location="cpu")
    encoder.load_state_dict(encoder_state, strict=True)
    del encoder_state
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    if resume:
        encoder = PeftModel.from_pretrained(encoder, resume[1], is_trainable=True)
    else:
        lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGET_MODULES, bias="none",
                          task_type="FEATURE_EXTRACTION", init_lora_weights=True)
        encoder = get_peft_model(encoder, lora)
    for name, parameter in encoder.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()

    head = FullSftHead(config.d_model)
    head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "head.pt"), map_location="cpu"), strict=True)
    head = head.to(dtype=torch.bfloat16)
    for parameter in head.parameters():
        parameter.requires_grad = False
    model = FormatAdapterModel(encoder, head)
    model.train()
    return model


def focal_bce(logits, targets):
    probabilities = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * probabilities + (1 - targets) * (1 - probabilities)
    return ((1 - pt).clamp(min=1e-6) ** FOCAL_GAMMA) * ce


def compute_metrics(probabilities, labels, categories, names):
    per_category = {}
    for category_id, name in enumerate(names):
        selected = categories == category_id
        if selected.sum() >= 2 and len(np.unique(labels[selected])) == 2:
            per_category[name] = float(average_precision_score(labels[selected], probabilities[selected]))
    macro = float(np.mean(list(per_category.values()))) if per_category else float("nan")
    return macro, per_category


def evaluate(model, loader, valid, device, category_names, rank, hbm_tracker):
    model.eval()
    logits, labels, categories = [], [], []
    with torch.no_grad():
        for ids, mask, batch_labels, batch_categories, _scope in pl_xla.MpDeviceLoader(loader, device):
            logits.append(model(ids, mask))
            labels.append(batch_labels.to(device))
            categories.append(batch_categories.to(device))
    gathered_logits = xm.all_gather(torch.cat(logits), dim=0)
    gathered_labels = xm.all_gather(torch.cat(labels), dim=0)
    gathered_categories = xm.all_gather(torch.cat(categories), dim=0)
    gathered_valid = xm.all_gather(torch.as_tensor(valid, device=device, dtype=torch.float32), dim=0)
    xm.mark_step()
    log_hbm(device, rank, "full human validation", hbm_tracker)
    keep = gathered_valid.cpu().numpy() > 0.5
    probabilities = torch.sigmoid(gathered_logits).cpu().numpy()[keep]
    labels_np = gathered_labels.cpu().numpy()[keep]
    categories_np = gathered_categories.cpu().numpy().astype(np.int64)[keep]
    macro, per_category = compute_metrics(probabilities, labels_np, categories_np, category_names)
    model.train()
    return {"macro_pr_auc": macro, "per_category": per_category, "n": int(keep.sum())}


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_checkpoint(model, optimizer, scheduler, label, state, rank):
    if rank != 0:
        return
    xm.mark_step()
    path = os.path.join(OUTPUT_DIR, label)
    adapter_path = os.path.join(path, "adapter")
    os.makedirs(adapter_path, exist_ok=True)
    adapter_state = {key: value.detach().cpu() for key, value in
                     get_peft_model_state_dict(model.encoder).items()}
    save_safetensors(adapter_state, os.path.join(adapter_path, "adapter_model.safetensors"))
    model.encoder.peft_config["default"].save_pretrained(adapter_path)
    torch.save(cpu_state(optimizer.state_dict()), os.path.join(path, "optimizer.pt"))
    torch.save(cpu_state(scheduler.state_dict()), os.path.join(path, "scheduler.pt"))
    with open(os.path.join(path, "train_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {label}: rows={state['trained_rows']:,} step={state['global_step']} "
          f"val={state.get('val_macro_pr_auc')}", flush=True)


def log_hbm(device, rank, label, tracker):
    """Best-effort allocator HBM telemetry. torch_xla does not expose the
    full compiled-program peak, so RESOURCE_EXHAUSTED remains authoritative."""
    try:
        info = xm.get_memory_info(device)
        if "kb_total" in info:
            total = int(info["kb_total"]) * 1024
            free = int(info["kb_free"]) * 1024
            used = total - free
        else:
            total = int(info.get("bytes_limit", 0))
            used = int(info.get("bytes_used", 0))
            free = total - used
        tracker["reported_peak_bytes"] = max(tracker.get("reported_peak_bytes", 0), used)
        if rank == 0:
            print(f"[HBM] {label}: reported_used={used / 2**30:.2f}GiB "
                  f"reported_free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB "
                  f"reported_peak={tracker['reported_peak_bytes'] / 2**30:.2f}GiB "
                  f"(compiled-program memory may be invisible)", flush=True)
    except Exception as error:
        if rank == 0:
            print(f"[HBM] {label}: unavailable: {type(error).__name__}: {error}", flush=True)


def smoke_pair(model, optimizer, device, rank, world_size, train_batch, eval_batch,
               category_weight, category_pos_weight, tracker):
    """Exact production paths: one real MpDeviceLoader train batch and one
    real MpDeviceLoader eval batch, each followed by blocking .item()."""
    log(rank, "SMOKE", f"probe train/eval={train_batch}/{eval_batch}, seq={PAIR_MAX_LENGTH}",
        force=True)
    train_loader, _sampler = make_train_loader(
        _SHARED["train_frame"], _SHARED["caches"], _SHARED["pad_id"],
        world_size, rank, train_batch)
    train_mp = pl_xla.MpDeviceLoader(train_loader, device)
    train_iterator = iter(train_mp)
    optimizer.zero_grad(set_to_none=True)
    ids, mask, labels, categories, scopes = next(train_iterator)
    logits = model(ids, mask)
    per_sample = focal_bce(logits, labels)
    pos_weight = 1.0 + (category_pos_weight[categories] - 1.0) * labels
    human_weight = torch.where(scopes == 0, torch.full_like(per_sample, HUMAN_WEIGHT),
                               torch.ones_like(per_sample))
    weights = category_weight[categories] * pos_weight * human_weight
    loss = (per_sample * weights).sum() / weights.sum()
    loss.backward()
    xm.mark_step()
    loss_value = loss.item()  # blocking: surfaces lazy XLA execution errors
    optimizer.zero_grad(set_to_none=True)
    log_hbm(device, rank, f"smoke train batch={train_batch}", tracker)
    del train_iterator, train_mp, train_loader, _sampler
    del ids, mask, labels, categories, scopes, logits, per_sample
    del pos_weight, human_weight, weights, loss
    gc.collect()

    eval_loader, _valid = make_eval_loader(
        _SHARED["val_frame"], _SHARED["caches"], _SHARED["pad_id"],
        world_size, rank, eval_batch)
    model.eval()
    with torch.no_grad():
        eval_mp = pl_xla.MpDeviceLoader(eval_loader, device)
        eval_iterator = iter(eval_mp)
        ids, mask, _labels, _categories, _scopes = next(eval_iterator)
        eval_value = model(ids, mask).float().mean().item()  # blocking
    model.train()
    log_hbm(device, rank, f"smoke eval batch={eval_batch}", tracker)
    del eval_iterator, eval_mp, eval_loader, _valid
    del ids, mask, _labels, _categories, _scopes
    gc.collect()
    if not math.isfinite(loss_value) or not math.isfinite(eval_value):
        raise RuntimeError(f"Non-finite smoke values: train={loss_value}, eval={eval_value}")
    log(rank, "SMOKE", f"PASS {train_batch}/{eval_batch}: loss={loss_value:.5f}", force=True)


def select_batches(model, optimizer, device, rank, world_size,
                   category_weight, category_pos_weight, tracker):
    """Compile the safe fallback, then run up to three binary refinements.
    Any OOM ends the search immediately: continuing after a failed XLA
    program is unsafe."""
    safe_train, safe_eval = MIN_PER_CORE_TRAIN_BATCH, MIN_PER_CORE_EVAL_BATCH
    smoke_pair(model, optimizer, device, rank, world_size, safe_train, safe_eval,
               category_weight, category_pos_weight, tracker)
    high = MAX_PER_CORE_TRAIN_BATCH
    for iteration in range(BATCH_SMOKE_ITERATIONS):
        candidate = (safe_train + high + 1) // 2
        candidate_eval = min(MAX_PER_CORE_EVAL_BATCH, candidate * 2)
        local_status = 1  # 1=pass, 0=OOM, -1=other failure
        local_error = None
        try:
            smoke_pair(model, optimizer, device, rank, world_size, candidate, candidate_eval,
                       category_weight, category_pos_weight, tracker)
        except RuntimeError as error:
            message = str(error)
            local_error = message
            local_status = (0 if "RESOURCE_EXHAUSTED" in message or
                            "out of memory" in message.lower() else -1)
        statuses = xm.all_gather(torch.tensor([local_status], device=device, dtype=torch.float32))
        xm.mark_step()
        statuses = statuses.cpu().numpy().astype(np.int64)
        if (statuses < 0).any():
            raise RuntimeError(f"Non-OOM smoke failure on at least one rank at "
                               f"{candidate}/{candidate_eval}; local_error={local_error}")
        if (statuses == 0).any():
            log(rank, "SMOKE", f"OOM at {candidate}/{candidate_eval}; stop probing and use "
                                f"last safe {safe_train}/{safe_eval}; local_error={local_error}",
                force=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            break
        safe_train, safe_eval = candidate, candidate_eval
    log(rank, "SMOKE", f"selected train/eval={safe_train}/{safe_eval}", force=True)
    return safe_train, safe_eval


def build_frame(id1, id2, labels, categories, scope, caches):
    row1 = lookup_rows(caches[scope], id1)
    row2 = lookup_rows(caches[scope], id2)
    missing = int((row1 < 0).sum() + (row2 < 0).sum())
    if missing:
        raise RuntimeError(f"{missing} endpoints missing from v4 {caches[scope]['scope']} cache")
    return {"scope": np.full(len(row1), scope, dtype=np.int8), "row1": row1, "row2": row2,
            "label": np.asarray(labels, dtype=np.float32), "category": np.asarray(categories, dtype=np.int64)}


def stratified_llm_sample(target, category, size):
    rng = np.random.RandomState(SEED)
    keep = []
    per_category = size // len(np.unique(category))
    for category_id in np.unique(category):
        rows = np.flatnonzero(category == category_id)
        take = min(per_category, len(rows))
        levels, counts = np.unique(target[rows], return_counts=True)
        quota = np.floor(counts / counts.sum() * take).astype(np.int64)
        deficit = take - int(quota.sum())
        if deficit:
            quota[np.argsort(-counts)[:deficit]] += 1
        picked = []
        for level, amount in zip(levels, quota):
            pool = rows[target[rows] == level]
            picked.append(rng.choice(pool, size=min(int(amount), len(pool)), replace=False))
        keep.append(np.concatenate(picked))
    selected = np.concatenate(keep)
    if len(selected) != size:
        raise RuntimeError(f"Expected {size:,} LLM rows, sampled {len(selected):,}")
    return selected


def load_shared_data():
    human_cache, llm_cache = find_cache("human"), find_cache("llm")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError("Human and LLM caches have different pad ids")
    caches = [human_cache, llm_cache]

    human = pl.read_parquet(find_file(ITEM_SPLIT_FILE)).to_pandas()
    category_names = sorted(human["category"].unique().tolist())
    category_to_id = {name: index for index, name in enumerate(category_names)}
    human["cat_id"] = human["category"].map(category_to_id).astype(np.int64)
    human_train = human[human["split"] == "train"].reset_index(drop=True)
    human_val = human[human["split"] == "val"].reset_index(drop=True)
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(human_val["id1"], human_val["id2"]))).encode()
    ).hexdigest()[:16]
    human_train_frame = build_frame(human_train.id1, human_train.id2, human_train.label,
                                    human_train.cat_id, 0, caches)
    val_frame = build_frame(human_val.id1, human_val.id2, human_val.label,
                            human_val.cat_id, 0, caches)

    llm = pl.read_parquet(find_file(LLM_SPLIT_FILE)).to_pandas()
    llm = llm[llm["split"] == "train"].reset_index(drop=True)
    llm["cat_id"] = llm["category"].map(category_to_id).astype(np.int64)
    selected = stratified_llm_sample(llm.target.to_numpy(), llm.cat_id.to_numpy(),
                                     LLM_TRAIN_SUBSAMPLE_SIZE)
    llm = llm.iloc[selected].reset_index(drop=True)
    llm_frame = build_frame(llm.id1, llm.id2, llm.target, llm.cat_id, 1, caches)
    train_frame = {key: np.concatenate((human_train_frame[key], llm_frame[key]))
                   for key in human_train_frame}

    counts = np.maximum(np.bincount(train_frame["category"], minlength=len(category_names)), 1)
    category_weight = (counts.sum() / len(category_names)) / counts
    positive_mass = np.bincount(train_frame["category"], weights=train_frame["label"],
                                minlength=len(category_names))
    category_pos_weight = (counts - positive_mass) / np.maximum(positive_mass, 1e-6)
    checkpoint_dir, source_state = find_source_checkpoint()
    resume = find_resume_checkpoint()
    print(f"[SETUP] train={len(train_frame['label']):,} human={len(human_train_frame['label']):,} "
          f"llm={len(llm_frame['label']):,} val={len(val_frame['label']):,} "
          f"val_fingerprint={val_fingerprint} source={checkpoint_dir}", flush=True)
    _SHARED.update(caches=caches, pad_id=human_cache["pad_id"], train_frame=train_frame,
                   val_frame=val_frame, category_names=category_names,
                   category_weight=category_weight.astype(np.float32),
                   category_pos_weight=category_pos_weight.astype(np.float32),
                   checkpoint_dir=checkpoint_dir, source_state=source_state,
                   val_fingerprint=val_fingerprint, resume=resume)
    del human, human_train, human_val, human_train_frame
    del llm, llm_frame, selected
    gc.collect()


def make_state(global_step, trained_rows, score, per_category, val_at_rows, best_score,
               stopped_reason, next_validation, qualified, train_batch, eval_batch,
               reported_peak_bytes):
    return {
        "experiment": EXPERIMENT_NAME,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_global_step": SOURCE_EXPECTED_STEP,
        "epoch": 1,
        "global_step": global_step,
        "trained_rows": trained_rows,
        "val_macro_pr_auc": score,
        "val_per_category": per_category,
        "val_at_trained_rows": val_at_rows,
        "best_val_macro_pr_auc": best_score,
        "next_validation_rows": next_validation,
        "qualified_for_decrease_stop": qualified,
        "stopped_reason": stopped_reason,
        "val_fingerprint": _SHARED["val_fingerprint"],
        "item_format_version": 4,
        "item_max_length": ITEM_MAX_LENGTH,
        "pair_max_length": PAIR_MAX_LENGTH,
        "per_core_train_batch": train_batch,
        "per_core_eval_batch": eval_batch,
        "reported_peak_hbm_gib": reported_peak_bytes / 2**30,
        "llm_train_subsample_size": LLM_TRAIN_SUBSAMPLE_SIZE,
        "human_weight": HUMAN_WEIGHT,
        "focal_gamma": FOCAL_GAMMA,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGET_MODULES},
        "head_frozen": True,
    }


def mp_fn(_index):
    rank, world_size = xr.global_ordinal(), xr.world_size()
    device = torch_xla.device()
    set_seed(SEED + rank)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = _WRAPPED_MODEL.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.classifier.parameters()):
        raise RuntimeError("FullSftHead must stay frozen")
    log(rank, "MODEL", f"LoRA trainable={sum(p.numel() for p in trainable):,}; head frozen", force=True)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    resume = _SHARED["resume"]
    resume_state = resume[2] if resume else {}
    category_weight = torch.tensor(_SHARED["category_weight"], device=device)
    category_pos_weight = torch.tensor(_SHARED["category_pos_weight"], device=device)
    hbm_tracker = {"reported_peak_bytes": 0}

    if resume:
        train_batch = int(resume_state["per_core_train_batch"])
        eval_batch = int(resume_state["per_core_eval_batch"])
    else:
        train_batch, eval_batch = select_batches(
            model, optimizer, device, rank, world_size,
            category_weight, category_pos_weight, hbm_tracker)

    total_batch = train_batch * world_size
    steps_per_epoch = len(_SHARED["train_frame"]["label"]) // total_batch
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                              num_warmup_steps=max(1, int(steps_per_epoch * WARMUP_RATIO)),
                              num_training_steps=steps_per_epoch)
    if resume:
        optimizer_path = os.path.join(resume[0], "optimizer.pt")
        scheduler_path = os.path.join(resume[0], "scheduler.pt")
        if not os.path.isfile(optimizer_path) or not os.path.isfile(scheduler_path):
            raise RuntimeError(f"Incomplete exact-resume checkpoint: {resume[0]}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        log(rank, "RESUME", f"restored adapter+optimizer+scheduler from {resume[0]}", force=True)
        smoke_pair(model, optimizer, device, rank, world_size, train_batch, eval_batch,
                   category_weight, category_pos_weight, hbm_tracker)
    train_loader, sampler = make_train_loader(_SHARED["train_frame"], _SHARED["caches"],
                                              _SHARED["pad_id"], world_size, rank, train_batch)
    val_loader, val_valid = make_eval_loader(_SHARED["val_frame"], _SHARED["caches"],
                                             _SHARED["pad_id"], world_size, rank, eval_batch)

    global_step = int(resume_state.get("global_step", 0))
    trained_rows = int(resume_state.get("trained_rows", global_step * total_batch))
    next_validation = int(resume_state.get("next_validation_rows", VALIDATE_EVERY_ROWS))
    best_score = float(resume_state.get("best_val_macro_pr_auc", -1.0))
    previous_score = resume_state.get("val_macro_pr_auc")
    previous_per_category = resume_state.get("val_per_category")
    last_val_rows = resume_state.get("val_at_trained_rows")
    qualified = bool(resume_state.get("qualified_for_decrease_stop", False))
    stopped_reason = None
    start = time.perf_counter()

    def run_validation(reason):
        nonlocal best_score, previous_score, previous_per_category, last_val_rows
        nonlocal qualified, stopped_reason, next_validation
        metrics = evaluate(model, val_loader, val_valid, device, _SHARED["category_names"],
                           rank, hbm_tracker)
        score = metrics["macro_pr_auc"]
        improved = score > best_score
        if improved:
            best_score = score
        log(rank, "VAL", f"{reason}: rows={trained_rows:,} score={score:.6f} best={best_score:.6f}", force=True)
        if rank == 0:
            print(f"[METRICS] per_category: {json.dumps(metrics['per_category'], ensure_ascii=False)}", flush=True)
        if qualified and previous_score is not None and score < previous_score:
            stopped_reason = f"qualified score decreased: {previous_score:.6f} -> {score:.6f}"
        qualified = qualified or score >= QUALIFIED_SCORE
        previous_score = score
        previous_per_category = metrics["per_category"]
        last_val_rows = trained_rows
        next_validation += VALIDATE_EVERY_ROWS * (2 if score < LOW_SCORE else 1)
        state = make_state(global_step, trained_rows, score, previous_per_category, last_val_rows,
                           best_score, stopped_reason, next_validation, qualified,
                           train_batch, eval_batch, hbm_tracker["reported_peak_bytes"])
        save_checkpoint(model, optimizer, scheduler, "last", state, rank)
        if improved:
            save_checkpoint(model, optimizer, scheduler, "best", state, rank)
        model.train()
        return score

    log(rank, "TRAIN", f"rows={len(_SHARED['train_frame']['label']):,} steps={steps_per_epoch:,} "
                        f"per_core_train/eval={train_batch}/{eval_batch} "
                        f"global_batch={total_batch} val_every={VALIDATE_EVERY_ROWS:,} rows", force=True)
    batch_iterator = iter(pl_xla.MpDeviceLoader(train_loader, device))
    if global_step:
        log(rank, "RESUME", f"skipping {global_step:,} deterministic epoch-1 batches", force=True)
        for _ in range(global_step):
            next(batch_iterator)
    for ids, mask, labels, categories, scopes in batch_iterator:
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mask)
        per_sample = focal_bce(logits, labels)
        pos_weight = 1.0 + (category_pos_weight[categories] - 1.0) * labels
        human_weight = torch.where(scopes == 0, torch.full_like(per_sample, HUMAN_WEIGHT),
                                   torch.ones_like(per_sample))
        weights = category_weight[categories] * pos_weight * human_weight
        loss = (per_sample * weights).sum() / weights.sum()
        loss.backward()
        xm.optimizer_step(optimizer)
        scheduler.step()
        global_step += 1
        trained_rows = global_step * total_batch

        if global_step % HBM_LOG_EVERY_STEPS == 0:
            xm.mark_step()
            _ = loss.item()
            log_hbm(device, rank, f"train step={global_step}", hbm_tracker)

        if global_step % LOG_EVERY_STEPS == 0:
            xm.mark_step()
            if rank == 0:
                print(f"[TRAIN] step={global_step:,}/{steps_per_epoch:,} rows={trained_rows:,} "
                      f"loss={loss.item():.5f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if trained_rows >= next_validation:
            run_validation("interval")
            if stopped_reason:
                break
        if time.perf_counter() - start >= MAX_TRAIN_SECONDS:
            stopped_reason = "TPU wall-clock safety cutoff"
            break

    expected_trained_rows = steps_per_epoch * total_batch
    if not stopped_reason and trained_rows < expected_trained_rows:
        stopped_reason = "loader ended before all non-dropped rows"
    if stopped_reason is None and trained_rows != last_val_rows:
        run_validation("final")
    final_state = make_state(global_step, trained_rows, previous_score, previous_per_category,
                             last_val_rows, best_score, stopped_reason or "one epoch complete",
                             next_validation, qualified, train_batch, eval_batch,
                             hbm_tracker["reported_peak_bytes"])
    save_checkpoint(model, optimizer, scheduler, "last", final_state, rank)
    log(rank, "DONE", final_state["stopped_reason"], force=True)


def main():
    global _WRAPPED_MODEL
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)
    load_shared_data()
    model = build_model_cpu(_SHARED["checkpoint_dir"], _SHARED["resume"])
    _WRAPPED_MODEL = xmp.MpModelWrapper(model)
    print("[MODEL] built once pre-fork and wrapped with MpModelWrapper", flush=True)
    torch_xla.launch(mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
