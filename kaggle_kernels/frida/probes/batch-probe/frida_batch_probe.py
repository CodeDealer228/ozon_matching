"""
Standalone, fast pre-flight check: finds the largest micro-batch size that
fits in GPU memory for FRIDA LoRA SFT (same model/LoRA config/PAIR_MAX_LENGTH
as frida_lora_sft.py), WITHOUT loading any real data or running any real
training -- just the model + synthetic input_ids, so this finishes in
minutes and answers "what batch size can this GPU actually handle" before
committing to the full 10-epoch run.

Self-contained (does not import frida_lora_sft.py -- Kaggle script kernels
are single-file) but keeps the exact same LoRA config, PAIR_MAX_LENGTH, and
fp16 choice as the real training script, so its answer is a real answer for
that script, not a rough guess:
  - target_modules=["q","k","v","o","wi_0","wi_1","wo"] (T5's real module
    names, verified against the loaded checkpoint -- see frida_lora_sft.py's
    README for how).
  - PAIR_MAX_LENGTH=512 (2x256, FRIDA's native context cap -- the pre-built
    item cache maxamsecret/frida-item-tokens-human uses 256 tokens/item with
    bos/eos already included, so a full pair is up to 512).
  - fp16 + GradScaler, not bf16 (T4 has no real bf16 tensor cores; see
    frida_lora_sft.py's module docstring for the CLAUDE.md-documented
    gotcha this avoids).

Unlike frida_lora_sft.py's embedded run_vram_probe (which stops once it
finds a batch under a fixed 12.5GB/GPU target with margin, since that's
"good enough" for a training run that then also needs headroom for the eval
loader / bookkeeping tensors), THIS script pushes all the way to a REAL OOM
at each doubling step first, then binary-searches the exact boundary --
"до OOM" (until OOM), a genuine ceiling measurement, not a conservative
pick. Report both: the true ceiling AND a recommended (margined) value to
actually train with.
"""

import gc
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

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
PAIR_MAX_LENGTH = 128   # matches frida_lora_sft.py's active config -- a
                         # deliberate cost/quota-over-completeness choice
                         # (~94% of real pairs get truncated at this budget,
                         # vs the data-driven near-lossless 384 where only
                         # 0.97% do -- see that script's PAIR_MAX_LENGTH
                         # comment / this kernel's README).

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

START_BATCH = 4
MAX_BATCH_CAP = 1024
ROUND_TO = 4
SAFETY_MARGIN_GB = 1.5      # subtracted from the true ceiling for the
                             # recommended training batch -- real training
                             # also runs the eval loop, data loader pinning,
                             # etc. alongside, not just one isolated step.


class FridaCrossEncoderClassifier(nn.Module):
    # autocast lives INSIDE forward(), not wrapped around calls to this
    # module -- matches frida_lora_sft.py's own fix. Root cause: DataParallel
    # runs each replica's forward() in its own thread, and an autocast
    # context entered by the CALLER (as this probe previously did) doesn't
    # propagate into those threads -- PyTorch's own docs prescribe entering
    # autocast inside forward() for exactly this reason. Confirmed via a
    # dedicated GPU diagnostic (kaggle_kernels/frida-nan-diagnose): wrapping
    # outside produced bit-for-bit identical NaN output under fp16 AND bf16
    # once DataParallel was involved, while a single-GPU bf16 run (no
    # DataParallel) was completely clean.
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :]).view(-1).float()


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _probe_step(batch_size, n_gpu, model, loss_fn, trainable_params, device):
    input_ids = torch.randint(0, 1000, (batch_size, PAIR_MAX_LENGTH), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, PAIR_MAX_LENGTH), dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)

    for p in trainable_params:
        p.grad = None
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)

    # bf16 autocast happens INSIDE FridaCrossEncoderClassifier.forward() now
    # (see that class for why), not wrapped here.
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = loss_fn(logits, labels)
    loss.backward()

    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))

    for p in trainable_params:
        p.grad = None
    del input_ids, attention_mask, labels, logits, loss
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def main():
    if not torch.cuda.is_available():
        print("[PROBE] No GPU available -- this probe needs one to mean anything. Exiting.")
        return

    device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[INFO] GPU: {gpu_name} x{n_gpu} | total VRAM/GPU: {total_vram_gb:.1f}GB")
    print(f"[INFO] PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}, LoRA target_modules={LORA_TARGET_MODULES}")

    print(f"[MODEL] loading {MODEL_NAME} (T5EncoderModel) + fresh LoRA ...")
    # attn_implementation="eager", matching frida_lora_sft.py's own fix -- keeps
    # this probe's memory footprint representative of what actually trains
    # (see that script's build_fresh_model for the full root-cause writeup:
    # sdpa's fused CUDA kernels have documented bf16/fp16 NaN edge cases with
    # padding masks on older GPUs like T4).
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    classifier_head = nn.Linear(base_model.config.d_model, 1)
    model = FridaCrossEncoderClassifier(peft_model, classifier_head).to(device)

    if n_gpu > 1:
        print(f"[GPU] wrapping in DataParallel across {n_gpu} GPUs (probe measures peak across all of them)")
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("=" * 90)
    print(f"[PROBE] doubling from batch={START_BATCH} until a REAL OOM (not stopping at a target threshold) ...")
    print("=" * 90)

    results = []
    candidate = START_BATCH
    last_good, last_good_gb = None, None
    oom_at = None

    while candidate <= MAX_BATCH_CAP:
        try:
            peak_gb = _probe_step(candidate, n_gpu, model, loss_fn, trainable_params, device)
            print(f"[PROBE] batch={candidate:5d} -> peak={peak_gb:.2f}GB/GPU "
                  f"({100 * peak_gb / total_vram_gb:.0f}% of {total_vram_gb:.1f}GB)")
            results.append((candidate, peak_gb))
            last_good, last_good_gb = candidate, peak_gb
            candidate *= 2
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] batch={candidate:5d} -> REAL OOM (this is the ceiling doubling found)")
            oom_at = candidate
            break
        gc.collect()

    if oom_at is None:
        print(f"[PROBE] never hit OOM up to MAX_BATCH_CAP={MAX_BATCH_CAP} -- either this GPU has a LOT of "
              f"headroom at PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}, or MAX_BATCH_CAP should be raised.")
        true_ceiling = last_good
    else:
        print("=" * 90)
        print(f"[PROBE] binary-searching the exact boundary between {last_good} (fits) and {oom_at} (OOMs) ...")
        print("=" * 90)
        lo, hi = last_good, oom_at
        while hi - lo > ROUND_TO:
            mid = ((lo + hi) // 2 // ROUND_TO) * ROUND_TO
            if mid <= lo:
                break
            try:
                peak_gb = _probe_step(mid, n_gpu, model, loss_fn, trainable_params, device)
                print(f"[PROBE] batch={mid:5d} -> peak={peak_gb:.2f}GB/GPU (fits)")
                results.append((mid, peak_gb))
                lo, last_good, last_good_gb = mid, mid, peak_gb
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                print(f"[PROBE] batch={mid:5d} -> OOM")
                hi = mid
            gc.collect()
        true_ceiling = last_good

    print("=" * 90)
    print("[RESULT]")
    print(f"  True ceiling (largest batch that actually fits): {true_ceiling} "
          f"(peak {last_good_gb:.2f}GB/GPU of {total_vram_gb:.1f}GB)" if last_good_gb else f"  True ceiling: {true_ceiling}")

    if last_good_gb:
        # Recommend a value with SAFETY_MARGIN_GB of headroom subtracted,
        # rounded down to ROUND_TO -- real training also runs the eval
        # loop, DataLoader pinned-memory buffers, etc. alongside single-step
        # peak, so leave room rather than shipping the exact measured edge.
        target_gb = total_vram_gb - SAFETY_MARGIN_GB
        if last_good_gb <= target_gb:
            recommended = true_ceiling
        else:
            # scale down proportionally from the measured peak, then round
            scale = target_gb / last_good_gb
            recommended = max(ROUND_TO, (int(true_ceiling * scale) // ROUND_TO) * ROUND_TO)
        print(f"  Recommended MICRO_BATCH_SIZE for real training (with {SAFETY_MARGIN_GB:.1f}GB safety margin): "
              f"{recommended}")
        print(f"  -> set MICRO_BATCH_SIZE={recommended} in frida_lora_sft.py's CONFIG, or trust its own "
              f"embedded run_vram_probe to rediscover the same number automatically.")
    print("=" * 90)
    print("[DONE] all measurements:")
    for b, gb in results:
        print(f"    batch={b:5d}  peak={gb:.2f}GB/GPU")


if __name__ == "__main__":
    main()
