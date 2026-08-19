"""
Computes per-ITEM embeddings (mean-pooled, NOT per-pair) for every one of
the 711,304 items_human.parquet items, using sergeyzh/rubert-mini-frida as
a proper bi-encoder: one independent embedding per item, comparable via
cosine to any other item's -- unlike sergeyzh/BERTA elsewhere in this
project, which is used as a cross-encoder (pair jointly attended, no
per-item vector exists). No LoRA adapter here -- rubert-mini-frida is used
as-is, base weights only.

Motivation: kaggle_kernels/retrieval-recall-hard-negative-mining measured
TF-IDF cosine's recall@k at reproducing organizers' own implicit
retrieval -- weak (recall@10 ranged 0.015-0.35 across categories, nowhere
near the ~0.9 target). Pure lexical overlap can't capture paraphrase/
synonym/word-order variation; a real trained embedding model should get
much closer. This kernel produces the embeddings a follow-up nearest-
neighbor pass needs to test that.

===========================================================================
TWO-KERNEL SPLIT: tokenization lives in ../rubert-mini-frida-tokenize, not
here
===========================================================================
kaggle_kernels/berta-lora-pair-cache's own v3 got SIGKILL (OOM) tokenizing
this same 711,304-item set on a T4 instance. Rather than just chunking
tighter on the same cramped machine (which IS still done here as a second
line of defense, but isn't the primary fix), tokenization runs as its own
kernel on a TPU-shaped Kaggle instance instead -- nothing there touches
the TPU chips, it's plain CPU tokenization, the point is the much larger
host RAM that instance shape comes with. This kernel just mounts that
kernel's already-tokenized memmap cache (published as a dataset by hand --
see this kernel's README) and does pure GPU-bound forward passes, with
nothing CPU/RAM-heavy competing against it.

===========================================================================
LAUNCH MECHANISM + BATCH SIZE -- lessons from berta-lora-pair-cache, applied
from the start instead of rediscovered
===========================================================================
torch.distributed.run --standalone --nproc_per_node=N (not
torch.multiprocessing.spawn, which crashed there with no clean traceback;
not DataParallel/DDP-wrapped, since this is pure inference with no
backward pass for either to actually help with). Each rank independently
owns half the items (contiguous split, no shared/redundant work needed --
unlike the pair-cache case, no rank needs the OTHER rank's items at all
here, so there's no tokenize-once-on-rank-0-then-share step required).

Batch size: berta-lora-pair-cache's own VRAM probe was capped at 1024
(BATCH_SIZE_CAP=1024) and stopped there measuring only 5.19GB/GPU peak --
nowhere near the ~11GB threshold, meaning the probe had real headroom left
and was stopped by that arbitrary cap, not by an actual VRAM limit. This
script's cap is deliberately much higher (8192) so the probe can find the
REAL ceiling instead of an arbitrary one -- "big batch but no OOM" per the
actual ask, discovered, not guessed.
"""

import os
import sys
import json
import math
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoModel

MODEL_NAME = "sergeyzh/rubert-mini-frida"
# hidden_dim is discovered at runtime from the loaded model's own config
# (encoder.config.hidden_size) -- not assumed/hardcoded here.

TOKEN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/rubert-mini-frida-item-tokenized/rubert_mini_frida_item_tokens"
ITEM_MAX_LENGTH = 256  # MUST match ../rubert-mini-frida-tokenize's own ITEM_MAX_LENGTH

BATCH_SIZE_START = 512
BATCH_SIZE_CAP = 8192      # was implicitly 1024 in berta-lora-pair-cache and
                            # never actually hit its VRAM ceiling there (peak
                            # 5.19GB at batch=1024, target threshold ~11GB) --
                            # let the probe find the real number this time.
TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 2.0
PROBE_ROUND_TO = 32

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SHARD_DIR = os.path.join(WORKING_DIR, "shards")
OUTPUT_DIR = os.path.join(WORKING_DIR, "rubert_mini_frida_item_embeddings")


def _elapsed(t0):
    return time.perf_counter() - t0


def _log(rank, stage, msg):
    print(f"[rank {rank}] {stage}: {msg}", flush=True)


def _find_dataset_dir(hint):
    if os.path.isdir(hint) and os.path.exists(os.path.join(hint, "cache_meta.json")):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "ids.bin" in files:
                return root
    raise RuntimeError(
        "No token cache found under /kaggle/input -- expected "
        "rubert-mini-frida-tokenize's output mounted as a dataset "
        "(dataset_sources in kernel-metadata.json)."
    )


class MeanPoolingEncoder(nn.Module):
    """Mean pooling (mask-weighted average over last_hidden_state) --
    matches sergeyzh/BERTA's own convention elsewhere in this project
    (same author/distillation lineage); the most common, robust default
    for a general-purpose sentence-embedding model absent a stated
    alternative. autocast INSIDE forward() -- not because this script uses
    DataParallel (it doesn't, see module docstring), but for consistency
    with every other model wrapper in this project and zero downside."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).float()


def _probe_peak_gb(batch_size, seq_len, probe_model, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    del dummy_input_ids, dummy_attention_mask
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(model, device, rank):
    if not torch.cuda.is_available():
        return BATCH_SIZE_START

    threshold = TARGET_VRAM_GB - SAFETY_MARGIN_GB
    candidate = BATCH_SIZE_START
    last_good, last_good_gb = None, None

    while candidate <= BATCH_SIZE_CAP:
        try:
            peak_gb = _probe_peak_gb(candidate, ITEM_MAX_LENGTH, model, device)
            _log(rank, "PROBE", f"batch={candidate} -> peak={peak_gb:.2f}GB")
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            _log(rank, "PROBE", f"batch={candidate} -> OOM")
            break
        last_good, last_good_gb = candidate, peak_gb
        if peak_gb >= threshold:
            break
        candidate *= 2

    if last_good is None:
        return max(PROBE_ROUND_TO, BATCH_SIZE_START // 2)

    lo, hi = last_good, min(candidate, BATCH_SIZE_CAP)
    while hi - lo > PROBE_ROUND_TO:
        mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
        if mid <= lo:
            break
        try:
            peak_gb = _probe_peak_gb(mid, ITEM_MAX_LENGTH, model, device)
            _log(rank, "PROBE", f"batch={mid} -> peak={peak_gb:.2f}GB")
            if peak_gb >= threshold:
                hi = mid
            else:
                lo = mid
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            hi = mid

    _log(rank, "PROBE", f"selected batch_size={lo} (last measured peak={last_good_gb:.2f}GB "
                        f"at batch={last_good}, target threshold={threshold:.1f}GB)")
    return lo


def worker():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cache_dir = _find_dataset_dir(TOKEN_CACHE_HINT)
    with open(os.path.join(cache_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    assert max_length == ITEM_MAX_LENGTH, (
        f"token cache max_length={max_length} != this script's ITEM_MAX_LENGTH={ITEM_MAX_LENGTH}")
    if rank == 0:
        _log(rank, "CONFIG", f"model={MODEL_NAME!r} n_items={n_items:,} "
                             f"item_max_length={ITEM_MAX_LENGTH} world_size={world_size}")
        _log(rank, "CONFIG", f"token cache meta: {meta}")

    all_ids = np.memmap(os.path.join(cache_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    all_input_ids = np.memmap(os.path.join(cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                               shape=(n_items, max_length))
    all_attn = np.memmap(os.path.join(cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                          shape=(n_items, max_length))

    # Sequential load across ranks -- avoids double-materializing the same
    # checkpoint simultaneously (established lesson from every other
    # multi-GPU kernel in this project).
    model = None
    for turn in range(world_size):
        if rank == turn:
            _log(rank, "MODEL", f"loading {MODEL_NAME} ...")
            encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
            hidden_dim = encoder.config.hidden_size
            _log(rank, "MODEL", f"loaded, hidden_size={hidden_dim}, "
                                f"config.model_type={getattr(encoder.config, 'model_type', '?')}")
            model = MeanPoolingEncoder(encoder).to(device)
            model.eval()
        dist.barrier()

    hidden_dim = model.encoder.config.hidden_size
    batch_size = run_vram_probe(model, device, rank)
    dist.barrier()

    # Contiguous half-split -- each rank needs only ITS OWN items, no
    # cross-rank sharing required at all (unlike berta-lora-pair-cache,
    # where a pair could reference any two items anywhere).
    chunk = math.ceil(n_items / world_size)
    lo = rank * chunk
    hi = min(n_items, lo + chunk)
    _log(rank, "SHARD", f"rows [{lo}:{hi}) of {n_items:,} total, batch_size={batch_size}")

    t0 = time.perf_counter()
    embeddings = np.empty((hi - lo, hidden_dim), dtype=np.float16)
    with torch.no_grad():
        for start in range(lo, hi, batch_size):
            end = min(hi, start + batch_size)
            input_ids = torch.from_numpy(np.asarray(all_input_ids[start:end])).long().to(device, non_blocking=True)
            attention_mask = torch.from_numpy(np.asarray(all_attn[start:end])).long().to(device, non_blocking=True)
            pooled = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings[start - lo:end - lo] = pooled.cpu().numpy().astype(np.float16)
            if (start - lo) % (batch_size * 20) == 0:
                _log(rank, "EMBED", f"{end - lo:,}/{hi - lo:,} ({_elapsed(t0):.0f}s elapsed)")

    _log(rank, "EMBED", f"shard done: {hi - lo:,} items, {_elapsed(t0):.0f}s")

    os.makedirs(SHARD_DIR, exist_ok=True)
    np.save(os.path.join(SHARD_DIR, f"embeddings_rank{rank}.npy"), embeddings)
    np.save(os.path.join(SHARD_DIR, f"ids_rank{rank}.npy"), np.asarray(all_ids[lo:hi]))
    dist.barrier()

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        emb_parts, id_parts = [], []
        for r in range(world_size):
            emb_parts.append(np.load(os.path.join(SHARD_DIR, f"embeddings_rank{r}.npy")))
            id_parts.append(np.load(os.path.join(SHARD_DIR, f"ids_rank{r}.npy")))
        merged_emb = np.concatenate(emb_parts, axis=0)
        merged_ids = np.concatenate(id_parts, axis=0)
        np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), merged_emb)
        np.save(os.path.join(OUTPUT_DIR, "ids.npy"), merged_ids)
        with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_NAME, "hidden_dim": int(hidden_dim), "n_items": int(len(merged_ids)),
                "item_max_length": ITEM_MAX_LENGTH, "pooling": "mean (mask-weighted), bf16 autocast",
                "embedding_dtype": "float16", "batch_size_used": batch_size, "world_size": world_size,
            }, f, ensure_ascii=False, indent=2)
        shutil.rmtree(SHARD_DIR, ignore_errors=True)
        _log(0, "DONE", f"merged {len(merged_ids):,} item embeddings (dim={hidden_dim}) -> {OUTPUT_DIR}")

    dist.barrier()
    dist.destroy_process_group()


def main_launcher():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available -- this kernel needs at least 1 GPU (2 expected).")
    world_size = torch.cuda.device_count()
    print(f"[MAIN] {world_size} GPU(s) visible: "
          f"{[torch.cuda.get_device_name(i) for i in range(world_size)]}", flush=True)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={world_size}",
        os.path.abspath(__file__),
    ]
    print(f"[MAIN] launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"torch.distributed.run exited with code {result.returncode}")
    print("[MAIN] all ranks finished.", flush=True)


if __name__ == "__main__":
    if "RANK" in os.environ:
        worker()
    else:
        main_launcher()
