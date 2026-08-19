"""
Minimal, fast, targeted GPU diagnostic for the NaN-loss-on-every-step bug in
frida-lora-sft: local CPU repro (same real data, same real model, plain fp32
AND torch.autocast(cpu, bfloat16)) came back completely clean -- finite
logits, sane BCE loss, zero degenerate rows across a 52K-row scan of the real
collate output. Forcing attn_implementation="eager" (bypassing SDPA's fused
CUDA kernels, the leading suspect) did NOT fix it either -- still 100% NaN
from step 0 on a real T4 run. So the bug is real-CUDA-specific but NOT
(solely) about which attention kernel gets picked.

This script isolates the remaining untested variables directly on a real GPU,
cheaply (a handful of forward passes, not a training loop -- should finish in
well under a minute of GPU time):
  1. Are the loaded weights themselves already NaN (bad download/cache)?
  2. Does PLAIN fp32 on GPU (no autocast at all) also NaN? If yes, this is
     not a precision/autocast issue at all -- something more fundamental
     (driver, specific CUDA kernel, corrupted weights, T4-specific bug).
  3. If fp32 is clean but fp16/bf16 autocast NaNs: per-block hooks on all 24
     T5EncoderModel blocks pinpoint the EXACT layer where NaN first appears,
     for each precision separately.
  4. Single GPU vs DataParallel (2 GPUs) -- the real training run always
     wraps in DataParallel; never tested in isolation until now.

Self-contained, mirrors frida_batch_probe.py's philosophy (no import of
frida_lora_sft.py, single-file Kaggle script kernel).
"""
import json
import os
import re
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

import numpy as np
import torch
import torch.nn as nn

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

from peft import LoraConfig, get_peft_model
from transformers import T5EncoderModel

MODEL_NAME = "ai-forever/FRIDA"
PAIR_MAX_LENGTH = 128
BATCH_SIZE = 32   # matches the real training run's probe-selected micro-batch

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

ITEM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-item-tokens-human"
MATCHES_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"


def _find_item_cache_dir(hint):
    if os.path.isdir(hint) and os.path.exists(os.path.join(hint, "ids.bin")):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "ids.bin" in files:
                return root
    return hint


def load_real_batch(device):
    """Real item ids from matches.parquet, real cached tokens, simple
    truncate-to-128 concat (not the full fair-share/category-excision
    policy -- that exact policy was already validated NaN-free on CPU; this
    just needs REAL content, not a byte-identical collate)."""
    import pandas as pd
    cache_dir = _find_item_cache_dir(ITEM_CACHE_HINT)
    with open(os.path.join(cache_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    ids = np.memmap(os.path.join(cache_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    item_input_ids = np.memmap(os.path.join(cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                                shape=(n_items, max_length))
    item_attention_mask = np.memmap(os.path.join(cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                                     shape=(n_items, max_length))
    id_to_row = {int(item_id): i for i, item_id in enumerate(ids)}
    item_lengths = np.asarray(item_attention_mask).sum(axis=1)

    matches = pd.read_parquet(MATCHES_PATH, columns=["id1", "id2"]).dropna().head(BATCH_SIZE * 3)
    pad_id = 0  # FRIDA/RobertaTokenizer pad id, verified in frida_lora_sft.py's own tokenizer load

    input_ids = np.full((BATCH_SIZE, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    attention_mask = np.zeros((BATCH_SIZE, PAIR_MAX_LENGTH), dtype=np.int64)
    n_filled = 0
    for _, row in matches.iterrows():
        if n_filled >= BATCH_SIZE:
            break
        r1, r2 = id_to_row.get(int(row["id1"])), id_to_row.get(int(row["id2"]))
        if r1 is None or r2 is None:
            continue
        l1, l2 = int(item_lengths[r1]), int(item_lengths[r2])
        seq1 = np.asarray(item_input_ids[r1, :l1])
        seq2 = np.asarray(item_input_ids[r2, :l2])
        combined = np.concatenate([seq1, seq2])[:PAIR_MAX_LENGTH]
        n = len(combined)
        input_ids[n_filled, :n] = combined
        attention_mask[n_filled, :n] = 1
        n_filled += 1

    assert n_filled == BATCH_SIZE, f"only found {n_filled}/{BATCH_SIZE} valid pairs"
    labels = torch.randint(0, 2, (BATCH_SIZE,), dtype=torch.float32, device=device)
    return (torch.from_numpy(input_ids).to(device), torch.from_numpy(attention_mask).to(device), labels)


class FridaCrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :]).view(-1)


def build_model(device, attn_implementation):
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation=attn_implementation)
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                              target_modules=LORA_TARGET_MODULES, bias="none")
    peft_model = get_peft_model(base_model, lora_config)
    classifier_head = nn.Linear(base_model.config.d_model, 1)
    model = FridaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


def check_weights_finite(model):
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
    return bad


def run_with_block_hooks(model, input_ids, attention_mask, dtype_label, autocast_dtype, device):
    print(f"\n{'=' * 90}\n[{dtype_label}] forward pass with per-block NaN hooks\n{'=' * 90}")

    block_status = {}

    def make_hook(name):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            has_nan = bool(torch.isnan(hs).any())
            has_inf = bool(torch.isinf(hs).any())
            block_status[name] = (has_nan, has_inf, float(hs.abs().max()) if not (has_nan or has_inf) else float("nan"))
        return hook

    handles = []
    for name, module in model.named_modules():
        if re.fullmatch(r".*block\.\d+", name):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    first_nan_block = None
    for name in sorted(block_status.keys(), key=lambda n: int(n.rsplit(".", 1)[1])):
        has_nan, has_inf, abs_max = block_status[name]
        flag = "NaN!" if has_nan else ("Inf!" if has_inf else "ok")
        print(f"  {name}: {flag} (abs_max={abs_max:.4g})" if flag == "ok" else f"  {name}: {flag}")
        if (has_nan or has_inf) and first_nan_block is None:
            first_nan_block = name

    print(f"[{dtype_label}] first bad block: {first_nan_block!r}")
    print(f"[{dtype_label}] final logits: {logits.float().tolist()}")
    print(f"[{dtype_label}] any NaN in final logits: {bool(torch.isnan(logits).any())}")
    return first_nan_block, logits


def main():
    if not torch.cuda.is_available():
        print("[DIAG] No GPU available. Exiting.")
        return

    device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} x{n_gpu}")
    print(f"[INFO] compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"[INFO] torch: {torch.__version__} | torch.version.cuda: {torch.version.cuda} | "
          f"cudnn: {torch.backends.cudnn.version()}")
    import transformers, peft as peft_module
    print(f"[INFO] transformers: {transformers.__version__} | peft: {peft_module.__version__}")
    print(f"[INFO] torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")

    print("\n[DATA] loading a real batch (real ids, real cached tokens) ...")
    input_ids, attention_mask, labels = load_real_batch(device)
    print(f"[DATA] input_ids shape: {input_ids.shape} | attention_mask row sums: "
          f"{attention_mask.sum(dim=1).tolist()}")

    print("\n[MODEL] building (eager attention, single GPU) ...")
    model = build_model(device, attn_implementation="eager")
    model.eval()

    bad_weights = check_weights_finite(model)
    print(f"[WEIGHTS] non-finite params right after load: {len(bad_weights)} "
          f"{'(' + ', '.join(bad_weights[:5]) + '...)' if bad_weights else ''}")

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 1) plain fp32, no autocast at all -- the critical missing test
    fb, logits_fp32 = run_with_block_hooks(model, input_ids, attention_mask, "FP32-no-autocast", None, device)
    if not torch.isnan(logits_fp32).any():
        loss_fp32 = loss_fn(logits_fp32, labels)
        print(f"[FP32-no-autocast] loss: {float(loss_fp32)}")

    # 2) fp16 autocast
    fb16, logits_fp16 = run_with_block_hooks(model, input_ids, attention_mask, "FP16-autocast", torch.float16, device)
    if not torch.isnan(logits_fp16.float()).any():
        loss_fp16 = loss_fn(logits_fp16.float(), labels)
        print(f"[FP16-autocast] loss: {float(loss_fp16)}")

    # 3) bf16 autocast (current script's config)
    fbbf16, logits_bf16 = run_with_block_hooks(model, input_ids, attention_mask, "BF16-autocast", torch.bfloat16, device)
    if not torch.isnan(logits_bf16.float()).any():
        loss_bf16 = loss_fn(logits_bf16.float(), labels)
        print(f"[BF16-autocast] loss: {float(loss_bf16)}")

    # 4) DataParallel, bf16 -- isolate whether multi-GPU wrapping itself is
    # the missing variable (real training always wraps, this diagnostic
    # hasn't tested it at all until now).
    if n_gpu > 1:
        print(f"\n{'=' * 90}\n[DataParallel] wrapping across {n_gpu} GPUs, bf16 autocast, no hooks "
              f"(DataParallel replicas break simple forward-hook capture) ...\n{'=' * 90}")
        model_dp = nn.DataParallel(model, device_ids=list(range(n_gpu)))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_dp = model_dp(input_ids=input_ids, attention_mask=attention_mask)
        print(f"[DataParallel-bf16] logits: {logits_dp.float().tolist()}")
        print(f"[DataParallel-bf16] any NaN: {bool(torch.isnan(logits_dp.float()).any())}")

    print("\n" + "=" * 90)
    print("[SUMMARY]")
    print(f"  weights finite at load: {len(bad_weights) == 0}")
    print(f"  FP32 no-autocast NaN:   {bool(torch.isnan(logits_fp32).any())}")
    print(f"  FP16 autocast NaN:      {bool(torch.isnan(logits_fp16.float()).any())} (first bad block: {fb16})")
    print(f"  BF16 autocast NaN:      {bool(torch.isnan(logits_bf16.float()).any())} (first bad block: {fbbf16})")
    print("=" * 90)


if __name__ == "__main__":
    main()
