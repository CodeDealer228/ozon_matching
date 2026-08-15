"""
Computes per-ITEM embeddings (mean-pooled, bi-encoder style: one item at a
time, [CLS] item [SEP]) using the CURRENT BEST BERTA+LoRA checkpoint
(maxamsecret/berta-lora-sft-5epoch, val_macro_pr_auc=0.7111) -- not raw
BERTA, the task-fine-tuned adapter. Unlike kaggle_kernels/berta-lora-pair-cache,
which uses this same checkpoint as a CROSS-encoder (pair jointly attended,
one vector per PAIR), this treats it as a bi-encoder (one independent
vector per ITEM) -- purely for the retrieval/hard-negative-mining line of
work, not something the model was end-to-end trained to do. Rationale:
LoRA has seen real task supervision (unlike sergeyzh/rubert-mini-frida,
generic pretrained), so even used in a mode it wasn't explicitly trained
for, its features may carry more task-relevant signal for retrieval than
a generic embedding model -- this kernel's own recall@k output (via a
follow-up mining kernel, mirroring ../rubert-mini-frida-recall-mining) is
the actual test of that, not assumed here.

===========================================================================
REUSES THE EXISTING TOKEN CACHE -- NO NEW TOKENIZATION KERNEL NEEDED
===========================================================================
maxamsecret/berta-lora-sft-5epoch already contains a full item-token cache
at berta_item_tokens/{ids,input_ids,attention_mask}.bin -- verified by
file size before writing any code here: attention_mask.bin is exactly
711,304 x 128 bytes (int8), input_ids.bin exactly 711,304 x 128 x 4 bytes
(int32), ids.bin exactly 711,304 x 8 bytes (int64) -- same item count and
ITEM_MAX_LENGTH=128 this project's other BERTA scripts already use. This
was tokenized WITHOUT special tokens (add_special_tokens=False, same
convention as berta-lora-pair-cache's own item cache -- CLS/SEP are added
per-pair at collate time there). Bi-encoder usage here needs its own
[CLS] + content + [SEP] framing instead (see build_single_item_batch),
built fresh from the cached raw content -- no re-tokenization, just
reassembly.

Category is excised from item content here too (_find_token_pattern_starts
+ build_compact_ids, ported verbatim from berta-lora-pair-cache), NOT
because it's uninformative for retrieval across categories (it isn't -- it
distinguishes 20 different catalog slices) but for consistency with what
the LoRA adapter actually saw during fine-tuning: pair assembly there
always excises the Category span before the encoder ever sees it, so
feeding it Category-included content now would be an out-of-distribution
input format relative to what shaped these particular weights.

===========================================================================
LESSONS FROM berta-lora-pair-cache's OWN debugging history, applied from
the start instead of rediscovered
===========================================================================
- peft version pin (transformers==4.57.6/peft==0.17.1/huggingface-hub==0.36.0,
  --no-deps --force-reinstall): v2 there hit ImportError: EmbeddingParallel
  from an unpinned `pip install peft` pulling PyPI's latest against this
  Kaggle image's older pre-installed transformers.
- torch.distributed.run --standalone --nproc_per_node=N launch, not
  torch.multiprocessing.spawn (crashed there with no clean traceback) and
  not DataParallel/DDP-wrapped (pure inference, nothing to sync).
- Sequential model loading across ranks (rank 0 first, barrier, then rank 1).
- VRAM probe cap raised to 8192 -- berta-lora-pair-cache's own probe was
  capped at 1024 and never found its real ceiling (peak was only 5.19GB/GPU
  against an ~11GB target at that cap).
- No redundant per-rank tokenization needed at all here (unlike the pair
  cache, where a PAIR can reference any two items anywhere and both ranks
  needed the same shared vocabulary) -- each rank just needs its OWN half
  of the already-cached items, no cross-rank dependency, so no
  tokenize-once-on-rank-0-then-share step required (mirrors
  ../rubert-mini-frida-item-embeddings's simpler split for the same reason).

OUTPUT ONLY -- no dataset created/published from inside this kernel.
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

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sergeyzh/BERTA"

ITEM_MAX_LENGTH = 128           # matches the cache's own memmap shape
CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]
SEQ_LEN = ITEM_MAX_LENGTH + 2   # [CLS] + up to 128 content + [SEP]

BATCH_SIZE_START = 512
BATCH_SIZE_CAP = 8192           # was implicitly 1024 in berta-lora-pair-cache
                                 # and never actually hit its VRAM ceiling
                                 # there (peak 5.19GB at batch=1024, target
                                 # ~11GB) -- let the probe find the real
                                 # number this time.
TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 2.0
PROBE_ROUND_TO = 32

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SHARD_DIR = os.path.join(WORKING_DIR, "shards")
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_lora_item_embeddings")


def _import_peft():
    """Lazy, called from worker() only -- see main_launcher()'s pinned
    install, done once in the parent before torch.distributed.run spawns
    children (mirrors berta-lora-pair-cache's own _import_peft exactly,
    same reasoning: avoids a redundant/racy reinstall in each child)."""
    try:
        import peft.import_utils
        import peft.tuners.lora.torchao
        peft.import_utils.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception as e:
        print(f"[WARN] torchao patch failed (likely fine): {e}")
    from peft import PeftModel
    return PeftModel


def _elapsed(t0):
    return time.perf_counter() - t0


def _log(rank, stage, msg):
    print(f"[rank {rank}] {stage}: {msg}", flush=True)


def _find_token_cache_dir():
    """Searches by file CO-OCCURRENCE (all three .bin files present in the
    same directory), not by the berta_item_tokens/ folder name or dataset
    name -- same robust-to-remounting convention already established in
    this project (e.g. _find_adapter_dir below, which searches for
    adapter_config.json rather than assuming a path)."""
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if {"ids.bin", "input_ids.bin", "attention_mask.bin"}.issubset(set(files)):
                return root
    raise RuntimeError(
        "No item-token cache (ids.bin/input_ids.bin/attention_mask.bin) found "
        "under /kaggle/input -- expected maxamsecret/berta-lora-sft-5epoch's "
        "berta_item_tokens/ mounted (dataset_sources)."
    )


def _find_adapter_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    raise RuntimeError("No adapter_config.json found under /kaggle/input.")


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


def build_compact_ids(item_input_ids_mm, row, item_length, category_start, attr_start):
    if category_start < 0 or attr_start < 0:
        return item_input_ids_mm[row, :item_length]
    head = item_input_ids_mm[row, :category_start]
    tail = item_input_ids_mm[row, attr_start:item_length]
    return np.concatenate((head, tail))


def build_single_item_batch(item_input_ids_mm, item_lengths, category_start, attr_start,
                             rows, cls_id, sep_id, pad_id):
    bsz = len(rows)
    input_ids = np.full((bsz, SEQ_LEN), pad_id, dtype=np.int64)
    attention_mask = np.zeros((bsz, SEQ_LEN), dtype=np.int64)
    token_type_ids = np.zeros((bsz, SEQ_LEN), dtype=np.int64)
    for i, row in enumerate(rows):
        compact = build_compact_ids(item_input_ids_mm, row, int(item_lengths[row]),
                                     int(category_start[row]), int(attr_start[row]))
        content_len = min(len(compact), SEQ_LEN - 2)
        input_ids[i, 0] = cls_id
        input_ids[i, 1:1 + content_len] = compact[:content_len]
        input_ids[i, 1 + content_len] = sep_id
        attention_mask[i, :2 + content_len] = 1
    return (torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
            torch.from_numpy(token_type_ids))


class BertaPoolingEncoder(nn.Module):
    """Mean pooling (mask-weighted average) -- identical to
    berta-lora-pair-cache's own wrapper, ported verbatim; mean pooling is
    agnostic to whether the sequence holds one item or two."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).float()


def _probe_peak_gb(batch_size, probe_model, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, SEQ_LEN), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, SEQ_LEN), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros((batch_size, SEQ_LEN), dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask,
                    token_type_ids=dummy_token_type_ids)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids
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
            peak_gb = _probe_peak_gb(candidate, model, device)
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
            peak_gb = _probe_peak_gb(mid, model, device)
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
    PeftModel = _import_peft()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    token_cache_dir = _find_token_cache_dir()
    n_items = os.path.getsize(os.path.join(token_cache_dir, "ids.bin")) // 8
    if rank == 0:
        _log(rank, "CONFIG", f"model={MODEL_NAME!r} n_items={n_items:,} "
                             f"item_max_length={ITEM_MAX_LENGTH} seq_len={SEQ_LEN} world_size={world_size}")
        _log(rank, "CONFIG", f"token cache dir: {token_cache_dir}")

    all_ids = np.memmap(os.path.join(token_cache_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    all_input_ids = np.memmap(os.path.join(token_cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                               shape=(n_items, ITEM_MAX_LENGTH))
    all_attn = np.memmap(os.path.join(token_cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                          shape=(n_items, ITEM_MAX_LENGTH))
    item_lengths = np.asarray(all_attn).sum(axis=1).astype(np.int32)
    category_start = _find_token_pattern_starts(np.asarray(all_input_ids), CATEGORY_LABEL_PATTERN)
    attr_start = _find_token_pattern_starts(np.asarray(all_input_ids), ATTRIBUTES_LABEL_PATTERN)

    # Sequential load across ranks -- avoids double-materializing the same
    # checkpoint simultaneously (established lesson).
    tokenizer = None
    model = None
    for turn in range(world_size):
        if rank == turn:
            _log(rank, "MODEL", f"loading {MODEL_NAME} + adapter ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
            adapter_dir = _find_adapter_dir()
            _log(rank, "MODEL", f"adapter dir: {adapter_dir}")
            peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
            model = BertaPoolingEncoder(peft_model).to(device)
            model.eval()
            _log(rank, "MODEL", "loaded")
        dist.barrier()

    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    hidden_dim = model.encoder.config.hidden_size

    batch_size = run_vram_probe(model, device, rank)
    dist.barrier()

    chunk = math.ceil(n_items / world_size)
    lo = rank * chunk
    hi = min(n_items, lo + chunk)
    _log(rank, "SHARD", f"rows [{lo}:{hi}) of {n_items:,} total, batch_size={batch_size}")

    t0 = time.perf_counter()
    embeddings = np.empty((hi - lo, hidden_dim), dtype=np.float16)
    with torch.no_grad():
        for start in range(lo, hi, batch_size):
            end = min(hi, start + batch_size)
            rows = list(range(start, end))
            input_ids, attention_mask, token_type_ids = build_single_item_batch(
                all_input_ids, item_lengths, category_start, attr_start, rows, cls_id, sep_id, pad_id)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)
            pooled = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
                "model": MODEL_NAME, "adapter": "maxamsecret/berta-lora-sft-5epoch",
                "source_val_macro_pr_auc": 0.7111, "hidden_dim": int(hidden_dim),
                "n_items": int(len(merged_ids)), "seq_len": SEQ_LEN, "item_max_length": ITEM_MAX_LENGTH,
                "pooling": "mean (mask-weighted), bf16 autocast", "category_excised": True,
                "embedding_dtype": "float16", "batch_size_used": batch_size, "world_size": world_size,
            }, f, ensure_ascii=False, indent=2)
        shutil.rmtree(SHARD_DIR, ignore_errors=True)
        _log(0, "DONE", f"merged {len(merged_ids):,} item embeddings (dim={hidden_dim}) -> {OUTPUT_DIR}")

    dist.barrier()
    dist.destroy_process_group()


def main_launcher():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
        "huggingface-hub==0.36.0", "transformers==4.57.6", "peft==0.17.1",
    ])

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
