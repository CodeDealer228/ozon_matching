"""
GPU-only, single-process eval kernel: computes the REAL val_macro_pr_auc
(human) and llm_val_macro_pr_auc, both with full per-category breakdowns,
for continue_c_llm3m_2ep's `last` checkpoint (global_step=13,474) -- the
quiet, unvalidated safety-net save that was NEVER actually evaluated during
training (val_macro_pr_auc: null in its own train_state.json), yet scored
**0.48999 on the real judging stand** -- the new project-best test score,
beating that SAME run's own validated `best`/checkpoint_epoch01 (val
0.7595, test 0.487). This kernel exists specifically to fill in the missing
val number for `last`, now that its test score is known to matter.

No torch_xla anywhere -- this is plain single-GPU PyTorch inference, not a
training kernel, so none of frida_full_sft_continue.py's TPU-multiprocess
machinery applies. Data-loading/pair-assembly/metric functions below are
copied verbatim from that file's own CPU-only helpers (load_v1_cache,
lookup_rows, allocate_pair_budget, build_compact_ids, PairIndexDataset,
make_pair_collate_fn, compute_metrics, _build_frame) -- these were already
XLA-agnostic (pure numpy/CPU), so "verbatim" really does mean unchanged,
not "adapted". Only the model's own forward() and the eval loop are
GPU-native instead of the training file's TPU-sharded/all_gathered version
(see evaluate() there) -- a single GPU needs no sharding across replicas
at all, the val sets (109,648 human + 100,000 llm) fit as one straight
pass through a DataLoader.

PAIR_MAX_LENGTH=192 -- the TRAINING length (frida_full_sft_continue.py's
own value), NOT the 188 the submission serves at. This kernel's whole
purpose is a number directly comparable to every other checkpoint's own
val_macro_pr_auc in this project's history (all measured at the training
length), not a submission-serving dry run.
"""
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score
from transformers import T5EncoderModel

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

CHECKPOINT_HINT = ("/kaggle/input/datasets/maxamsecret/frida-sft-experiment-c-llm-3m-output/"
                   "frida_full_sft/last")
ITEM_SPLIT_PATH_HINT = ("/kaggle/input/datasets/maxamsecret/frida-item-disjoint-split/"
                        "matches_item_disjoint_70_30_seed42.parquet")
LLM_SPLIT_PATH_HINT = ("/kaggle/input/datasets/maxamsecret/frida-llm-train-val-split/"
                       "llm_train_val_split.parquet")
HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/human_scope"
LLM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/llm_scope"

ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 192  # training length -- see module docstring

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

HEAD_HIDDEN = (1024, 512, 128)
HEAD_DROPOUT = 0.2

EVAL_BATCH = 128

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():8.1f}s][{stage}] {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _find_checkpoint_dir():
    if os.path.isfile(os.path.join(CHECKPOINT_HINT, "encoder.pt")):
        return CHECKPOINT_HINT
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "encoder.pt" in files and "head.pt" in files and os.path.basename(root) == "last":
                return root
    raise RuntimeError(f"[EVAL] no last/ checkpoint found under /kaggle/input "
                       f"(hint was {CHECKPOINT_HINT})")


# =====================================================================
# ITEM CACHES -- verbatim from frida_full_sft_continue.py (pure CPU/numpy,
# XLA-agnostic already)
# =====================================================================

EXPECTED_N_ITEMS = {"human": 711_304, "llm": 12_384_610}


def _pattern_starts_chunked(input_ids, n_items, tok_a, tok_b, chunk=500_000):
    starts = np.full(n_items, -1, dtype=np.int32)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        block = np.asarray(input_ids[lo:hi])
        mask = (block[:, :-1] == tok_a) & (block[:, 1:] == tok_b)
        rows, cols = np.where(mask)
        if len(rows):
            uniq_rows, first_idx = np.unique(rows, return_index=True)
            starts[lo + uniq_rows] = cols[first_idx]
        del block, mask, rows, cols
    return starts


def _lengths_chunked(attn, n_items, chunk=500_000):
    lengths = np.empty(n_items, dtype=np.int32)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        lengths[lo:hi] = np.asarray(attn[lo:hi]).sum(axis=1)
    return lengths


def _find_v1_cache_root(hint, expect_n_items):
    for candidate in (hint,):
        meta_path = os.path.join(candidate, "cache_meta.json")
        if os.path.isdir(candidate) and os.path.exists(meta_path):
            with open(meta_path) as f:
                if json.load(f).get("n_items") == expect_n_items:
                    return candidate
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "input_ids.bin" in files:
                with open(os.path.join(root, "cache_meta.json")) as f:
                    meta = json.load(f)
                if meta.get("n_items") == expect_n_items:
                    _log("CACHE", f"hint {hint} missed -- matched n_items={expect_n_items:,} at {root}")
                    return root
    raise RuntimeError(f"[CACHE] no item cache with n_items={expect_n_items:,} found (hint was {hint})")


def _derive_pad_id(input_ids, lengths, max_length, name):
    candidates = np.where(lengths < max_length)[0]
    if len(candidates) == 0:
        raise RuntimeError(f"[CACHE:{name}] every row is at max_length -- cannot derive pad_id")
    sample = candidates[:: max(1, len(candidates) // 500)][:500]
    pads = {int(input_ids[r, int(lengths[r])]) for r in sample}
    if len(pads) != 1:
        raise RuntimeError(f"[CACHE:{name}] inconsistent pad token: {sorted(pads)}")
    pad = pads.pop()
    _log("CACHE", f"{name}: pad_id={pad} (derived from the cache's own padding)")
    return pad


def load_v1_cache(hint, name):
    root = _find_v1_cache_root(hint, EXPECTED_N_ITEMS[name])
    with open(os.path.join(root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    _log("CACHE", f"{name}: {root}: n_items={n_items:,} max_length={max_length} prefix={meta.get('prefix')!r}")
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"[CACHE:{name}] max_length={max_length} != {ITEM_MAX_LENGTH}")

    ids_path = os.path.join(root, "ids.bin")
    input_ids_path = os.path.join(root, "input_ids.bin")
    attn_path = os.path.join(root, "attention_mask.bin")
    for fname, want in ((ids_path, n_items * 8),
                        (input_ids_path, n_items * max_length * 4),
                        (attn_path, n_items * max_length)):
        got = os.path.getsize(fname)
        if got != want:
            raise RuntimeError(f"[CACHE:{name}] {fname} size mismatch: got {got:,}, want {want:,}")

    ids = np.asarray(np.memmap(ids_path, dtype=np.int64, mode="r", shape=(n_items,)))
    input_ids = np.memmap(input_ids_path, dtype=np.int32, mode="r", shape=(n_items, max_length))
    attn = np.memmap(attn_path, dtype=np.int8, mode="r", shape=(n_items, max_length))

    t0 = time.perf_counter()
    lengths = _lengths_chunked(attn, n_items)
    del attn
    cat_start = _pattern_starts_chunked(input_ids, n_items, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _pattern_starts_chunked(input_ids, n_items, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    _log("CACHE", f"{name}: lengths + label positions scanned ({time.perf_counter() - t0:.1f}s)")

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    pad_id = _derive_pad_id(input_ids, lengths, max_length, name)
    return dict(name=name, n_items=n_items, input_ids=input_ids, lengths=lengths,
                cat_start=cat_start, attr_start=attr_start, order=order,
                sorted_ids=sorted_ids, pad_id=pad_id)


def lookup_rows(cache, item_ids):
    pos = np.searchsorted(cache["sorted_ids"], item_ids)
    pos_clipped = np.clip(pos, 0, len(cache["sorted_ids"]) - 1)
    hit = cache["sorted_ids"][pos_clipped] == item_ids
    rows = np.where(hit, cache["order"][pos_clipped], -1)
    return rows.astype(np.int64)


# =====================================================================
# PAIR ASSEMBLY -- verbatim
# =====================================================================

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


class PairIndexDataset(Dataset):
    def __init__(self, scope, row1, row2, labels, categories):
        self.scope = scope
        self.row1 = row1
        self.row2 = row2
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, i):
        return self.scope[i], self.row1[i], self.row2[i], self.labels[i], self.categories[i]


def make_pair_collate_fn(caches, pad_id, pair_max_length):
    def collate_fn(batch):
        scope, row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)
        for i in range(bsz):
            cache = caches[int(scope[i])]
            c1 = build_compact_ids(cache, int(row1[i]))
            c2 = build_compact_ids(cache, int(row2[i]))
            l1, l2 = len(c1), len(c2)
            if l1 + l2 <= pair_max_length:
                k1, k2 = l1, l2
            else:
                k1, k2 = allocate_pair_budget(l1, l2, pair_max_length)
            combined = np.concatenate((c1[:k1], c2[:k2]))
            n = len(combined)
            input_ids[i, :n] = combined
            attention_mask[i, :n] = 1
        return (torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
                torch.tensor(labels, dtype=torch.float32),
                np.asarray(categories, dtype=np.int64))
    return collate_fn


def _build_frame(id1, id2, label, category, scope, caches):
    cache = caches[scope]
    row1 = lookup_rows(cache, np.asarray(id1, dtype=np.int64))
    row2 = lookup_rows(cache, np.asarray(id2, dtype=np.int64))
    missing = int((row1 < 0).sum() + (row2 < 0).sum())
    if missing:
        raise RuntimeError(f"[SETUP] {missing:,} endpoints missing from the "
                           f"{cache['name']} item cache -- wrong cache scope?")
    return dict(scope=np.full(len(row1), scope, dtype=np.int8), row1=row1, row2=row2,
                label=np.asarray(label, dtype=np.float32),
                category=np.asarray(category, dtype=np.int64))


# =====================================================================
# MODEL -- verbatim architecture (FullSftHead/FridaFullSftClassifier), GPU
# forward instead of the training file's XLA autocast
# =====================================================================

class FullSftHead(nn.Module):
    def __init__(self, hidden_dim, layer_sizes=HEAD_HIDDEN, dropout=HEAD_DROPOUT):
        super().__init__()
        blocks = []
        d = hidden_dim
        for h in layer_sizes:
            blocks.append(nn.Sequential(
                nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


class FridaFullSftClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :].float()
        return self.classifier(cls)


# =====================================================================
# METRICS
# =====================================================================

def compute_metrics(probs, labels, categories, category_names):
    per_cat = {}
    for cat_id, cat_name in enumerate(category_names):
        mask = categories == cat_id
        if mask.sum() < 2:
            continue
        y_true = labels[mask]
        if len(np.unique(y_true)) < 2:
            continue
        per_cat[cat_name] = average_precision_score(y_true, probs[mask])
    macro = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return macro, per_cat


@torch.inference_mode()
def run_eval(model, frame, caches, pad_id, category_vocab, device, label):
    dataset = PairIndexDataset(frame["scope"], frame["row1"], frame["row2"],
                               frame["label"], frame["category"])
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False, drop_last=False,
                        collate_fn=make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH),
                        num_workers=2)
    all_probs, all_labels, all_categories = [], [], []
    t0 = time.perf_counter()
    n_done = 0
    for input_ids, attention_mask, batch_labels, batch_categories in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch_labels.numpy())
        all_categories.append(batch_categories)
        n_done += len(probs)
        if n_done % (EVAL_BATCH * 50) < EVAL_BATCH:
            elapsed = time.perf_counter() - t0
            _log("EVAL", f"{label}: {n_done:,}/{len(frame['label']):,} "
                        f"({elapsed:.1f}s, {n_done / max(elapsed, 1e-9):.1f} pairs/s)")
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    categories = np.concatenate(all_categories)
    macro, per_cat = compute_metrics(probs, labels, categories, category_vocab)
    _log("EVAL", f"{label}: DONE macro_pr_auc={macro:.4f} n={len(labels):,} "
                f"({time.perf_counter() - t0:.1f}s)")
    return macro, per_cat, len(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        _log("WARN", "no CUDA device visible -- this kernel needs a GPU accelerator enabled")
    _log("SETUP", f"device={device}")
    if device.type == "cuda":
        # REAL CRASH (2026-08-23): two straight runs died with "CUDA error:
        # no kernel image is available for execution on the device" at the
        # very first real op (a plain nn.Embedding lookup) -- AFTER weights
        # had already loaded and .to(device) had already succeeded (memory
        # copies don't need compiled kernels; compute ops do). That error
        # means the installed torch build has no compiled kernel for this
        # GPU's actual compute capability -- logged loudly here, before the
        # ~270s of cache-loading, so a third failure is diagnosable in
        # seconds instead of minutes.
        _log("SETUP", f"torch={torch.__version__} torch.version.cuda={torch.version.cuda} "
                      f"device_name={torch.cuda.get_device_name(0)!r} "
                      f"capability={torch.cuda.get_device_capability(0)} "
                      f"device_count={torch.cuda.device_count()} "
                      f"arch_list={torch.cuda.get_arch_list()}")
        # Fail in SECONDS, not ~270s (the cache-loading cost paid twice
        # already), if this build genuinely cannot execute any real kernel
        # on this GPU -- a trivial elementwise add is as basic an op as
        # torch has, same failure class as the embedding lookup that has
        # killed both prior runs.
        try:
            _ = (torch.zeros(4, device=device) + 1).sum().item()
            _log("SETUP", "CUDA smoke op OK (tiny elementwise add + reduce)")
        except Exception as e:
            _log("SETUP", f"CUDA SMOKE OP FAILED: {type(e).__name__}: {e} -- this build cannot "
                          f"execute ANY real kernel on this GPU, not specific to the model. "
                          f"Stopping now instead of repeating the ~270s cache-load to hit the "
                          f"same class of failure again.")
            raise

    ckpt_dir = _find_checkpoint_dir()
    _log("SETUP", f"checkpoint: {ckpt_dir}")
    with open(os.path.join(ckpt_dir, "train_state.json")) as f:
        source_train_state = json.load(f)
    _log("SETUP", f"source train_state.json: {source_train_state}")

    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches: "
                           f"{human_cache['pad_id']} vs {llm_cache['pad_id']}")
    caches = [human_cache, llm_cache]
    pad_id = human_cache["pad_id"]

    # ---- human val (item-disjoint 70/30 split) ----
    split_path = _find_dataset_file(ITEM_SPLIT_PATH_HINT, "matches_item_disjoint_70_30_seed42.parquet")
    split_df = pl.read_parquet(split_path).to_pandas()
    category_vocab = sorted(split_df["category"].unique().tolist())
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    split_df["cat_id"] = split_df["category"].map(category_to_id).astype(np.int64)
    human_val = split_df[split_df["split"] == "val"].reset_index(drop=True)
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(human_val["id1"], human_val["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    val_frame = _build_frame(human_val["id1"], human_val["id2"], human_val["label"],
                             human_val["cat_id"], 0, caches)
    _log("SETUP", f"human val={len(human_val):,} val_fingerprint={val_fingerprint} "
                 f"categories={len(category_vocab)}")
    del split_df, human_val
    gc.collect()

    # ---- llm val (no-leak split) ----
    llm_split_path = _find_dataset_file(LLM_SPLIT_PATH_HINT, "llm_train_val_split.parquet")
    llm_split_df = pl.read_parquet(llm_split_path).to_pandas()
    llm_split_df["cat_id"] = llm_split_df["category"].map(category_to_id).astype(np.int64)
    llm_val = llm_split_df[llm_split_df["split"] == "val"].reset_index(drop=True)
    llm_val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(llm_val["id1"], llm_val["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    llm_val_frame = _build_frame(llm_val["id1"], llm_val["id2"], llm_val["label"],
                                 llm_val["cat_id"].to_numpy().astype(np.int64), 1, caches)
    _log("SETUP", f"llm val={len(llm_val):,} val_fingerprint={llm_val_fingerprint}")
    del llm_split_df, llm_val
    gc.collect()

    # ---- model ----
    _log("MODEL", f"building {MODEL_NAME} encoder + FullSftHead ...")
    encoder = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    hidden_dim = encoder.config.d_model
    encoder_sd = torch.load(os.path.join(ckpt_dir, "encoder.pt"), map_location="cpu")
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"[MODEL] encoder state_dict mismatch -- missing={missing} "
                           f"unexpected={unexpected}")
    del encoder_sd

    head = FullSftHead(hidden_dim)
    head_sd = torch.load(os.path.join(ckpt_dir, "head.pt"), map_location="cpu")
    head.load_state_dict(head_sd)
    del head_sd

    model = FridaFullSftClassifier(encoder, head).to(device)
    model.eval()
    _log("MODEL", "weights loaded and verified (strict key match)")

    # ---- eval ----
    val_macro, val_per_cat, n_val = run_eval(model, val_frame, caches, pad_id, category_vocab,
                                             device, "human_val")
    llm_val_macro, llm_val_per_cat, n_llm_val = run_eval(model, llm_val_frame, caches, pad_id,
                                                         category_vocab, device, "llm_val")

    result = {
        "checkpoint_dir": ckpt_dir,
        "source_train_state": source_train_state,
        "val_macro_pr_auc": val_macro,
        "val_per_category": val_per_cat,
        "val_fingerprint": val_fingerprint,
        "n_val": n_val,
        "llm_val_macro_pr_auc": llm_val_macro,
        "llm_val_per_category": llm_val_per_cat,
        "llm_val_fingerprint": llm_val_fingerprint,
        "n_llm_val": n_llm_val,
        "pair_max_length": PAIR_MAX_LENGTH,
        "eval_kernel": "frida_eval_c_last_checkpoint.py",
    }
    out_path = os.path.join(WORKING_DIR, "eval_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log("DONE", f"val_macro_pr_auc={val_macro:.4f} llm_val_macro_pr_auc={llm_val_macro:.4f} "
                f"-> {out_path}")
    _log("DONE", f"total wall-clock {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
