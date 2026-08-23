# %% [code]
"""
Targeted, cheap TPU diagnostic for frida-percat-lora's chronic NaN problem:
across MULTIPLE independent real runs (the original run that produced
frida-with-merged-lora-and-8-10-14-cat-loras, and today's v8/v9 runs, all
using the SAME "fixed" config -- HEAD_LR_MULTIPLIER=3.0, HEAD_WARMUP_STEPS=20,
GRAD_CLIP_NORM=1.0), only categories 07/10/14 have ever survived from a fresh
init. 8 of the first 10 categories in the most recent real run diverged to
NaN within 2-40 steps. That "fix" clips gradients (a BACKWARD-pass lever) and
ramps the head's LR -- but the training loop's own check
(`torch.isfinite(loss)`) fires BEFORE `loss.backward()`, meaning the loss is
already non-finite from a bad FORWARD pass. Grad clipping cannot address
that; it was only ever mitigating a different failure mode.

This kernel isolates the real question instead of tuning LR/warmup/clip a
FOURTH time: is the FROZEN MERGED BACKBONE itself, under
`torch.autocast("xla", dtype=torch.bfloat16)`, numerically unstable on
SPECIFIC categories' real content -- and does a few steps of real gradient
update on a freshly-reset LoRA (the exact production path) reproduce the
exact same divergence, cheaply, in isolation? Mirrors frida-nan-diagnose's
own method (per-block finiteness hooks, real data, compare precisions) --
that kernel diagnosed a DIFFERENT bug (DataParallel + autocast thread
propagation on MiniLM/CUDA) already fixed and not relevant here; this one
is TPU/XLA-specific and needs a real torch_xla.autocast to mean anything.

SCOPE: human-only real data (matches.parquet + items_human.parquet). The
real per-category training mix is ~30% human + ~70% LLM soft-labelled rows,
but the hypothesis under test -- category/content-dependent bf16 forward
instability in the merged backbone -- doesn't need the LLM half to be
tested; if it reproduces on pure human content for the SAME categories that
fail in production, that's already strong signal. Extending to LLM content
is a natural follow-up if this comes back inconclusive.

SINGLE TPU CORE, NOT torch_xla.launch: this is five categories x ~50 steps
of forward+backward on a small (batch=24, seq=192) LoRA-adapted model, not a
distributed training run -- torch_xla.device() works standalone without the
multi-process spawn machinery, and a single core is sufficient to reproduce
a genuine forward-pass NaN (each production core runs its own independent
model copy too; xm.optimizer_step's all-reduce is the only thing single
-core doesn't reproduce, and the failure this is chasing happens before
backward() ever runs).

Constants, LoRA config, LR schedule, and every data-plumbing helper below
are copied byte-for-byte from frida-percat-lora/frida_percat_lora.py
(SEED, LORA_R/ALPHA/DROPOUT/TARGET_MODULES, LR, HEAD_LR_MULTIPLIER,
HEAD_WARMUP_STEPS, GRAD_CLIP_NORM, category-label-token-based pair
assembly, ...) so a real divergence here means the SAME divergence, not an
artifact of a simplified repro -- exactly the same reasoning
frida-nan-diagnose already used ("mirrors frida_batch_probe.py's
philosophy... not import, single-file Kaggle script kernel").

CATEGORIES TESTED, from the most recent real run's own log:
  01 Аптека              -- diverged at step 2/336 (fastest failure)
  09 Мебель              -- diverged at step 3/336
  00 Автотовары          -- diverged at step 40/339
  05 Детские товары      -- diverged at step 171/333 (latest failure, still
                             failed eventually)
  07 Канцелярские товары -- the ONLY category that has ever fully trained
                             and validated, across every real run so far
                             (val_pr_auc=0.7308 most recently) -- the
                             control this diagnostic needs: if IT also
                             shows instability here, the hypothesis is
                             wrong and the real cause is elsewhere.
"""
import json
import math
import os
import re
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch_xla
import torch_xla.core.xla_model as xm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import T5EncoderModel

# =====================================================================
# CONSTANTS -- copied from frida_percat_lora.py, do not drift from it
# =====================================================================

SEED = 42

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/human_scope"
EXTERNAL_CHECKPOINTS_HINT = (
    "/kaggle/input/datasets/maxamsecret/"
    "frida-with-merged-lora-and-8-10-14-cat-loras/frida_percat_lora"
)

ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 192
CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30
EXPECTED_N_ITEMS_HUMAN = 711_304

HEAD_DROPOUT = 0.1
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

LR = 2e-4
HEAD_LR_MULTIPLIER = 3.0
HEAD_WARMUP_STEPS = 20
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01
LR_MIN_RATIO = 0.05
PER_CORE_TRAIN_BATCH = 24

# Representative of a real category (most categories: ~333-339 steps/epoch,
# warmup=33) -- exact enough for the LR trajectory over the first N_STEPS
# tested here to match production closely; the schedule's SHAPE, not this
# specific total, is what matters for reproducing early-training divergence.
REPRESENTATIVE_TOTAL_STEPS = 333
REPRESENTATIVE_WARMUP_STEPS = 33

N_DIAGNOSTIC_STEPS = 50
BATCHES_PER_CATEGORY = 12  # 12 x 24 = 288 distinct real rows/category, cycled

TARGET_CATEGORIES = ["Аптека", "Мебель", "Автотовары", "Детские товары",
                     "Канцелярские товары"]

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def _log(msg):
    print(f"[DIAG] {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _find_merged_base_dir():
    if os.path.isfile(os.path.join(EXTERNAL_CHECKPOINTS_HINT, "merged_base", "config.json")):
        return os.path.join(EXTERNAL_CHECKPOINTS_HINT, "merged_base")
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _files in os.walk("/kaggle/input"):
            if os.path.basename(root) == "merged_base" and os.path.isfile(
                    os.path.join(root, "config.json")):
                return root
    raise RuntimeError("[DIAG] no merged_base/ found under /kaggle/input -- "
                       "is frida-with-merged-lora-and-8-10-14-cat-loras mounted?")


# =====================================================================
# ITEM CACHE -- byte-identical logic to frida_percat_lora.py's load_v1_cache
# =====================================================================

def _pattern_starts_chunked(input_ids, n_items, tok_a, tok_b, chunk=500_000):
    starts = np.full(n_items, -1, dtype=np.int64)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        block = np.asarray(input_ids[lo:hi])
        match = (block[:, :-1] == tok_a) & (block[:, 1:] == tok_b)
        rows, cols = np.nonzero(match)
        first = {}
        for r, c in zip(rows, cols):
            if r not in first:
                first[r] = c
        for r, c in first.items():
            starts[lo + r] = c
    return starts


def _lengths_chunked(attn, n_items, chunk=500_000):
    lengths = np.zeros(n_items, dtype=np.int64)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        lengths[lo:hi] = np.asarray(attn[lo:hi]).sum(axis=1)
    return lengths


def _derive_pad_id(input_ids, lengths, max_length):
    candidates = np.where(lengths < max_length)[0]
    sample = candidates[:: max(1, len(candidates) // 500)][:500]
    pads = {int(input_ids[r, int(lengths[r])]) for r in sample}
    if len(pads) != 1:
        raise RuntimeError(f"[DIAG] inconsistent pad token: {sorted(pads)}")
    return pads.pop()


def _find_human_cache_root():
    if os.path.isdir(HUMAN_CACHE_HINT) and os.path.isfile(
            os.path.join(HUMAN_CACHE_HINT, "cache_meta.json")):
        return HUMAN_CACHE_HINT
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "input_ids.bin" in files:
                with open(os.path.join(root, "cache_meta.json")) as f:
                    meta = json.load(f)
                if meta.get("n_items") == EXPECTED_N_ITEMS_HUMAN:
                    return root
    raise RuntimeError(f"[DIAG] no human item cache with "
                       f"n_items={EXPECTED_N_ITEMS_HUMAN:,} found")


def load_human_cache():
    root = _find_human_cache_root()
    with open(os.path.join(root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    _log(f"item cache: {root} n_items={n_items:,} max_length={max_length}")
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"[DIAG] max_length={max_length} != {ITEM_MAX_LENGTH}")

    ids = np.asarray(np.memmap(os.path.join(root, "ids.bin"), dtype=np.int64,
                               mode="r", shape=(n_items,)))
    input_ids = np.memmap(os.path.join(root, "input_ids.bin"), dtype=np.int32,
                          mode="r", shape=(n_items, max_length))
    attn = np.memmap(os.path.join(root, "attention_mask.bin"), dtype=np.int8,
                     mode="r", shape=(n_items, max_length))

    lengths = _lengths_chunked(attn, n_items)
    del attn
    cat_start = _pattern_starts_chunked(input_ids, n_items, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _pattern_starts_chunked(input_ids, n_items, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    pad_id = _derive_pad_id(input_ids, lengths, max_length)

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    return dict(input_ids=input_ids, lengths=lengths, cat_start=cat_start,
                attr_start=attr_start, order=order, sorted_ids=sorted_ids, pad_id=pad_id)


def lookup_rows(cache, item_ids):
    pos = np.searchsorted(cache["sorted_ids"], item_ids)
    pos_clipped = np.clip(pos, 0, len(cache["sorted_ids"]) - 1)
    hit = cache["sorted_ids"][pos_clipped] == item_ids
    return np.where(hit, cache["order"][pos_clipped], -1).astype(np.int64)


def allocate_pair_budget(len1, len2, budget):
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def build_compact_ids(cache, row):
    item_length = int(cache["lengths"][row])
    category_start = int(cache["cat_start"][row])
    attr_start = int(cache["attr_start"][row])
    ids = cache["input_ids"]
    if category_start < 0 or attr_start < 0:
        return np.asarray(ids[row, :item_length])
    head = np.asarray(ids[row, :category_start])
    tail = np.asarray(ids[row, attr_start:item_length])
    return np.concatenate((head, tail))


def build_pair_batch(cache, row1_ids, row2_ids, labels, device):
    bsz = len(row1_ids)
    input_ids = np.full((bsz, PAIR_MAX_LENGTH), cache["pad_id"], dtype=np.int64)
    attention_mask = np.zeros((bsz, PAIR_MAX_LENGTH), dtype=np.int64)
    for i in range(bsz):
        c1 = build_compact_ids(cache, int(row1_ids[i]))
        c2 = build_compact_ids(cache, int(row2_ids[i]))
        l1, l2 = len(c1), len(c2)
        if l1 + l2 <= PAIR_MAX_LENGTH:
            k1, k2 = l1, l2
        else:
            k1, k2 = allocate_pair_budget(l1, l2, PAIR_MAX_LENGTH)
        combined = np.concatenate((c1[:k1], c2[:k2]))
        n = len(combined)
        input_ids[i, :n] = combined
        attention_mask[i, :n] = 1
    return (torch.from_numpy(input_ids).to(device),
            torch.from_numpy(attention_mask).to(device),
            torch.tensor(labels, dtype=torch.float32, device=device))


# =====================================================================
# MODEL -- same LoRA config as production; autocast is a TOGGLE here
# (production's FridaCrossEncoderClassifier hardcodes it on) so fp32-vs-
# bf16 can actually be compared at the same weights.
# =====================================================================

class DiagModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, use_autocast):
        if use_autocast:
            with torch.autocast("xla", dtype=torch.bfloat16):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :].float()
        return self.classifier(cls).view(-1).float()


def build_model(device, merged_dir):
    try:
        base = T5EncoderModel.from_pretrained(merged_dir, dtype=torch.bfloat16,
                                              attn_implementation="eager")
    except (TypeError, ValueError):
        base = T5EncoderModel.from_pretrained(merged_dir, dtype=torch.bfloat16)
    hidden_dim = base.config.d_model

    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGET_MODULES, bias="none",
                          task_type=TaskType.FEATURE_EXTRACTION)
    peft_model = get_peft_model(base, lora_cfg)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            p.requires_grad = True

    model = DiagModel(peft_model, LinearHead(hidden_dim)).to(device)
    return model


class LinearHead(nn.Module):
    def __init__(self, hidden_dim, dropout=HEAD_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.out(self.dropout(x)).view(-1)


def reset_specialist_(model, seed):
    torch.manual_seed(seed)
    for name, p in model.encoder.named_parameters():
        if "lora_A" in name:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in name:
            nn.init.zeros_(p)
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)


def make_param_groups(model):
    lora_params = [p for n, p in model.encoder.named_parameters()
                   if "lora_" in n and p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return [{"params": lora_params, "lr": LR, "name": "lora"},
            {"params": head_params, "lr": LR * HEAD_LR_MULTIPLIER, "name": "head"}]


def _make_head_lr_lambda(warmup_steps, total_steps, min_ratio):
    head_warmup = max(1, min(HEAD_WARMUP_STEPS, warmup_steps))

    def lr_lambda(step):
        if step < head_warmup:
            return step / head_warmup
        if step < warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _make_lr_lambda(warmup_steps, total_steps, min_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# =====================================================================
# PER-BLOCK FINITENESS HOOKS -- same method as frida-nan-diagnose
# =====================================================================

def run_hooked_forward(model, input_ids, attention_mask, use_autocast, grad_enabled):
    block_status = {}

    def make_hook(name):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            has_nan = bool(torch.isnan(hs).any())
            has_inf = bool(torch.isinf(hs).any())
            block_status[name] = (has_nan, has_inf,
                                  float(hs.abs().max()) if not (has_nan or has_inf) else float("nan"))
        return hook

    handles = []
    for name, module in model.named_modules():
        if re.fullmatch(r".*block\.\d+", name):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                          use_autocast=use_autocast)
    finally:
        for h in handles:
            h.remove()

    first_bad = None
    for name in sorted(block_status.keys(),
                       key=lambda n: int(n.rsplit(".", 1)[1])):
        has_nan, has_inf, _ = block_status[name]
        if (has_nan or has_inf) and first_bad is None:
            first_bad = name
    return logits, first_bad, block_status


# =====================================================================
# MAIN
# =====================================================================

def main():
    device = torch_xla.device()
    _log(f"device: {device}")

    merged_dir = _find_merged_base_dir()
    _log(f"merged base: {merged_dir}")
    cache = load_human_cache()

    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"]).dropna()
    matches["label"] = (matches["target"].astype(float) >= 0.5).astype(np.float32)
    items = pd.read_parquet(items_path, columns=["id", "category"]).dropna()
    id_to_category = dict(zip(items["id"], items["category"]))
    category_vocab = sorted(items["category"].unique().tolist())
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    _log(f"{len(category_vocab)} categories total")

    matches["category"] = matches["id1"].map(id_to_category)
    matches = matches.dropna(subset=["category"])

    _log("building model (fresh LoRA over the merged base) ...")
    model = build_model(device, merged_dir)

    summary = []
    for cat_name in TARGET_CATEGORIES:
        if cat_name not in category_to_id:
            _log(f"WARNING: category {cat_name!r} not found in items_human.parquet -- skipping")
            continue
        cat_id = category_to_id[cat_name]
        cat_rows = matches[matches["category"] == cat_name]
        n_needed = BATCHES_PER_CATEGORY * PER_CORE_TRAIN_BATCH
        have_enough = len(cat_rows) >= n_needed
        if not have_enough:
            _log(f"WARNING: only {len(cat_rows)} real rows for {cat_name!r}, wanted {n_needed} "
                f"-- batches will repeat")
        cat_rows = cat_rows.sample(n=n_needed, replace=not have_enough, random_state=SEED)
        row1 = lookup_rows(cache, cat_rows["id1"].to_numpy(dtype=np.int64))
        row2 = lookup_rows(cache, cat_rows["id2"].to_numpy(dtype=np.int64))
        keep = (row1 >= 0) & (row2 >= 0)
        row1, row2 = row1[keep], row2[keep]
        labels_all = cat_rows["label"].to_numpy()[keep]
        n_batches_available = max(1, len(row1) // PER_CORE_TRAIN_BATCH)
        pos_rate = float(labels_all.mean())
        pos_weight_val = (1.0 - pos_rate) / max(pos_rate, 1e-9)

        _log(f"\n{'=' * 90}\n[{cat_id:02d}] {cat_name}: {len(row1)} real human rows available "
            f"({n_batches_available} batches), pos_rate={pos_rate:.3f}, "
            f"pos_weight={pos_weight_val:.3f}\n{'=' * 90}")

        reset_specialist_(model, SEED + cat_id)
        optimizer = torch.optim.AdamW(make_param_groups(model), weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
            _make_lr_lambda(REPRESENTATIVE_WARMUP_STEPS, REPRESENTATIVE_TOTAL_STEPS, LR_MIN_RATIO),
            _make_head_lr_lambda(REPRESENTATIVE_WARMUP_STEPS, REPRESENTATIVE_TOTAL_STEPS, LR_MIN_RATIO),
        ])
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_val, device=device))

        diverged_at = None
        first_bad_block = None
        for step in range(1, N_DIAGNOSTIC_STEPS + 1):
            b = (step - 1) % n_batches_available
            lo, hi = b * PER_CORE_TRAIN_BATCH, (b + 1) * PER_CORE_TRAIN_BATCH
            input_ids, attention_mask, batch_labels = build_pair_batch(
                cache, row1[lo:hi], row2[lo:hi], labels_all[lo:hi], device)

            optimizer.zero_grad()
            logits, bad_block, _status = run_hooked_forward(
                model, input_ids, attention_mask, use_autocast=True, grad_enabled=True)
            loss = loss_fn(logits, batch_labels)
            # Matches production's own check exactly (frida_percat_lora.py's
            # step loop): .item() alone is the sync point, no separate
            # mark_step -- and the per-block hooks above already forced
            # their own sync as each block completed, so there's no risk of
            # this deferring a large graph the way frida-full-sft's eval did.
            finite = bool(torch.isfinite(loss).item())

            if step <= 5 or step % 10 == 0 or not finite:
                _log(f"  step {step:3d}: loss={loss.item():.4f} finite={finite} "
                    f"lr_lora={scheduler.get_last_lr()[0]:.2e} "
                    f"first_bad_block={bad_block}")

            if not finite:
                diverged_at = step
                first_bad_block = bad_block
                _log(f"  [{cat_id:02d}] {cat_name}: DIVERGED at step {step} -- "
                    f"first non-finite block: {bad_block}")
                # Same batch, same (now-corrupted) weights, fp32 no-autocast:
                # isolates whether bf16 specifically is the trigger, or the
                # weights themselves are already bad regardless of precision.
                _logits_fp32, bad_block_fp32, _ = run_hooked_forward(
                    model, input_ids, attention_mask, use_autocast=False, grad_enabled=False)
                _log(f"  [{cat_id:02d}] {cat_name}: SAME batch/weights under fp32 "
                    f"no-autocast -- first bad block: {bad_block_fp32} "
                    f"(None means fp32 stayed finite -- bf16-specific)")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP_NORM)
            xm.optimizer_step(optimizer)
            scheduler.step()

        summary.append(dict(category=cat_name, category_id=cat_id,
                            n_real_rows=int(len(row1)), pos_rate=pos_rate,
                            diverged_at_step=diverged_at, first_bad_block=first_bad_block,
                            survived_all_steps=diverged_at is None))

    _log(f"\n{'=' * 90}\n[SUMMARY]\n{'=' * 90}")
    for row in summary:
        status = (f"DIVERGED step {row['diverged_at_step']} at {row['first_bad_block']}"
                  if not row["survived_all_steps"]
                  else f"survived all {N_DIAGNOSTIC_STEPS} steps")
        _log(f"  [{row['category_id']:02d}] {row['category']:<24} "
            f"n={row['n_real_rows']:>4} pos_rate={row['pos_rate']:.3f}  {status}")

    with open(os.path.join(WORKING_DIR, "nan_diagnose_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _log("done -- nan_diagnose_summary.json written")


if __name__ == "__main__":
    main()
