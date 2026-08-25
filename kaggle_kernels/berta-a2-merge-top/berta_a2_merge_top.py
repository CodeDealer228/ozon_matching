"""Merge Max's exact test-top BERTA LoRA and verify inference equivalence.

Inputs are immutable and hash checked.  The kernel also prepares a compact
token cache for the 10,123 train-clean A2 variants and produces the frozen
BERTA old/new diagnostic.  No model training happens here.
"""
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from sklearn.metrics import average_precision_score
from transformers import AutoModel, AutoTokenizer

# Kaggle's current GPU image ships torchao 0.10.0 while the bundled PEFT
# dispatcher accepts only >=0.16.  This model is ordinary fp32/fp16 BERTA and
# does not use torchao at all, so disable only that optional dispatcher.  This
# is the same compatibility patch used by the established BERTA kernels here.
try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as exc:
    print(f"[WARN] optional torchao dispatcher patch failed: {exc}", flush=True)


START = time.perf_counter()
OUT = Path("/kaggle/working/berta_a2_merge_top")
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_ADAPTER_SHA = "26b294123b2445fa7bb46e0c36c2ee5ad259c218ea80e357661f49729800cd34"
EXPECTED_CLASSIFIER_SHA = "70ac231c64ad2f70ac73592afbce76c3ee9f0cb804ecb16e8b0f4258325d43fa"
EXPECTED_SYNTH_SHA = "37cd340ba6d5205d8358e6825fc579b3a7f45a9d6d09079641fbf547be060c1a"
EXPECTED_VAL_SHA = "a48993ca6f54d0b7dde7fbe3f6e637a737ee6cb0c55f33b21e4970c40b6e8bb9"
EXPECTED_A2_ROWS = 10_123
EXPECTED_VAL_ROWS = 73_131
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 256
PREFIX = "paraphrase: "
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 442
CATEGORY_PATTERN = np.asarray([14948, 30], dtype=np.int32)
ATTR_PATTERN = np.asarray([4816, 37519, 521, 30], dtype=np.int32)
FP32_MAX_TOL = 1e-4
FP32_MEAN_TOL = 1e-5
RUNTIME_MAX_TOL = 2e-2
RUNTIME_MEAN_TOL = 2e-3
AP_TOL = 1e-4
INFER_BATCH = 32
MAX_SECONDS = 39_600


def log(stage, message):
    print(f"[{time.perf_counter()-START:8.1f}s][{stage}] {message}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def find_file(name, expected_sha=None, near=None):
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files and (near is None or near.lower() in root.lower()):
            path = os.path.join(root, name)
            if expected_sha is None or sha256_file(path) == expected_sha:
                found.append(path)
    if len(found) != 1:
        raise RuntimeError(f"Expected one {name} near={near!r}, found {found}")
    return found[0]


def find_base_root():
    found = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "merge_meta.json" in files and "merged_encoder" in dirs:
            found.append(root)
    if len(found) != 1:
        raise RuntimeError(f"Expected one first-stage merged BERTA root, found {found}")
    return found[0]


def find_human_scope():
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "cache_meta.json" in files and "human_scope" in root.lower():
            found.append(root)
    if len(found) != 1:
        raise RuntimeError(f"Expected one human_scope cache, found {found}")
    return found[0]


def find_dataset_file(name):
    matches = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            matches.append(os.path.join(root, name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}, found {matches}")
    return matches[0]


def pattern_starts(arr, pattern):
    plen = len(pattern)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    for offset in range(arr.shape[1] - plen + 1):
        pending = starts < 0
        if not pending.any():
            break
        rows = np.flatnonzero(pending)
        view = arr[rows, offset:offset + plen]
        matched = np.all(view == pattern[None, :], axis=1)
        starts[rows[matched]] = offset
    return starts


def load_human_cache():
    root = find_human_scope()
    meta = json.loads(Path(root, "cache_meta.json").read_text(encoding="utf-8"))
    if meta["n_items"] <= 0 or meta["max_length"] != ITEM_MAX_LENGTH or meta["prefix"] != PREFIX:
        raise RuntimeError(f"Human cache identity mismatch: {meta}")
    n = int(meta["n_items"])
    ids = np.memmap(Path(root, "ids.bin"), dtype=np.int64, mode="r", shape=(n,))
    tokens = np.memmap(Path(root, "input_ids.bin"), dtype=np.int32, mode="r", shape=(n, ITEM_MAX_LENGTH))
    attn = np.memmap(Path(root, "attention_mask.bin"), dtype=np.int8, mode="r", shape=(n, ITEM_MAX_LENGTH))
    lengths = np.asarray(attn).sum(axis=1).astype(np.int16)
    cat_start = pattern_starts(tokens, CATEGORY_PATTERN)
    attr_start = pattern_starts(tokens, ATTR_PATTERN)
    categories = np.load(Path(root, "categories.npy"))
    category_vocab = json.loads(Path(root, "category_vocab.json").read_text(encoding="utf-8"))
    id_to_row = {int(item_id): row for row, item_id in enumerate(ids)}
    log("CACHE", f"human rows={n:,} root={root}")
    return dict(root=root, ids=ids, tokens=tokens, lengths=lengths, cat_start=cat_start,
                attr_start=attr_start, categories=categories, category_vocab=category_vocab,
                id_to_row=id_to_row, meta=meta)


def compact(cache, row):
    length = int(cache["lengths"][row])
    cs, ats = int(cache["cat_start"][row]), int(cache["attr_start"][row])
    if cs < 0 or ats < 0:
        return cache["tokens"][row, :length]
    return np.concatenate((cache["tokens"][row, :cs], cache["tokens"][row, ats:length]))


def allocate_pair_budget(len1, len2, budget):
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def make_pair_batch(row1, row2, cache1, cache2, cls_id, sep_id, pad_id):
    bsz = len(row1)
    ids = np.full((bsz, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    mask = np.zeros((bsz, PAIR_MAX_LENGTH), dtype=np.int64)
    types = np.zeros((bsz, PAIR_MAX_LENGTH), dtype=np.int64)
    budget = PAIR_MAX_LENGTH - 3
    for i, (r1, r2) in enumerate(zip(row1, row2)):
        a, b = compact(cache1, int(r1)), compact(cache2, int(r2))
        k1, k2 = allocate_pair_budget(len(a), len(b), budget)
        n1 = 1 + k1 + 1
        total = n1 + k2 + 1
        ids[i, 0] = cls_id
        ids[i, 1:1+k1] = a[:k1]
        ids[i, 1+k1] = sep_id
        ids[i, n1:n1+k2] = b[:k2]
        ids[i, n1+k2] = sep_id
        mask[i, :total] = 1
        types[i, n1:total] = 1
    return ids, mask, types


class Head(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dropout = nn.Dropout(0.10)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.linear(self.dropout(x))


class Model(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)).float()
        return self.head(pooled).view(-1).float()


def load_head(path, hidden):
    head = Head(hidden)
    state = torch.load(path, map_location="cpu")
    if "linear.weight" in state:
        head.load_state_dict(state)
    else:
        head.linear.load_state_dict(state)
    return head


@torch.no_grad()
def score(model, row1, row2, cache1, cache2, tokenizer, device, autocast):
    model.eval()
    out = []
    for start in range(0, len(row1), INFER_BATCH):
        stop = min(start + INFER_BATCH, len(row1))
        arr = make_pair_batch(row1[start:stop], row2[start:stop], cache1, cache2,
                              tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
        tensors = [torch.from_numpy(x).to(device) for x in arr]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast):
            logits = model(*tensors)
        out.append(logits.float().cpu().numpy())
        if start and start % (INFER_BATCH * 200) == 0:
            log("SCORE", f"{start:,}/{len(row1):,}")
        if time.perf_counter() - START > MAX_SECONDS:
            raise TimeoutError("merge/equivalence wall-clock safety stop")
    return np.concatenate(out)


def metrics(logits, labels, cat_ids, category_vocab):
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    per_cat = {}
    for cat_id, name in enumerate(category_vocab):
        m = cat_ids == cat_id
        if m.sum() and np.unique(labels[m]).size == 2:
            per_cat[name] = float(average_precision_score(labels[m], probs[m]))
    return {
        "macro_ap": float(np.mean(list(per_cat.values()))),
        "micro_ap": float(average_precision_score(labels, probs)),
        "per_category_ap": per_cat,
    }


def build_synthetic_cache(a2, items, tokenizer):
    needed = set(int(x) for x in a2["id2"])
    subset = items[items["id"].isin(needed)][["id", "category", "attributes"]].copy()
    if subset["id"].nunique() != len(needed):
        raise RuntimeError("Could not resolve all A2 item-B rows in items_human")
    joined = a2.merge(subset, left_on="id2", right_on="id", how="left", validate="many_to_one")
    texts = (
        "Name: " + joined["name2_prime"].fillna("").astype(str).str.slice(0, MAX_NAME_CHARS)
        + " Category: " + joined["category_y"].fillna("").astype(str).str.slice(0, MAX_CATEGORY_CHARS)
        + " Attributes: " + joined["attributes"].fillna("").astype(str).str.slice(0, MAX_ATTR_CHARS)
    )
    enc = tokenizer([PREFIX + x for x in texts.tolist()], add_special_tokens=False,
                    padding="max_length", truncation=True, max_length=ITEM_MAX_LENGTH,
                    return_tensors="np")
    tokens = enc["input_ids"].astype(np.int32)
    lengths = enc["attention_mask"].sum(1).astype(np.int16)
    cache = {
        "tokens": tokens,
        "lengths": lengths,
        "cat_start": pattern_starts(tokens, CATEGORY_PATTERN),
        "attr_start": pattern_starts(tokens, ATTR_PATTERN),
    }
    np.save(OUT / "synthetic_input_ids.npy", tokens)
    np.save(OUT / "synthetic_lengths.npy", lengths)
    np.save(OUT / "synthetic_category_start.npy", cache["cat_start"])
    np.save(OUT / "synthetic_attr_start.npy", cache["attr_start"])
    index = joined[["synthetic_key", "id1", "id2", "category_x", "model", "candidate_index"]].copy()
    index = index.rename(columns={"category_x": "category"})
    index["synthetic_row"] = np.arange(len(index), dtype=np.int32)
    index.to_parquet(OUT / "synthetic_cache_index.parquet", index=False)
    atomic_json(OUT / "synthetic_cache_meta.json", {
        "status": "complete", "rows": len(index), "max_length": ITEM_MAX_LENGTH,
        "prefix": PREFIX, "text_formula": "Name/Category/Attributes 140/32/442",
        "index_sha256": sha256_file(OUT / "synthetic_cache_index.parquet"),
    })
    return cache, index


def main():
    progress = {"status": "running", "phase": "preflight"}
    atomic_json(OUT / "run_status.json", progress)
    device = torch.device("cuda")
    if torch.cuda.get_device_capability()[0] < 7:
        raise RuntimeError("Unsupported GPU")

    adapter = find_file("adapter_model.safetensors", EXPECTED_ADAPTER_SHA, "top-checkpoint")
    adapter_dir = os.path.dirname(adapter)
    classifier_path = find_file("classifier.pt", EXPECTED_CLASSIFIER_SHA, "top-checkpoint")
    a2_path = find_file("a2_train_clean.parquet", EXPECTED_SYNTH_SHA)
    val_path = find_file("historical_val_manifest.parquet", EXPECTED_VAL_SHA)
    base_root = find_base_root()
    base_dir = os.path.join(base_root, "merged_encoder")
    items_path = find_dataset_file("items_human.parquet")
    identities = {
        "adapter_sha256": sha256_file(adapter), "classifier_sha256": sha256_file(classifier_path),
        "a2_sha256": sha256_file(a2_path), "val_sha256": sha256_file(val_path),
        "base_root": base_root, "base_merge_meta": json.loads(Path(base_root, "merge_meta.json").read_text()),
    }
    atomic_json(OUT / "input_identity.json", identities)

    a2 = pd.read_parquet(a2_path)
    val = pd.read_parquet(val_path)
    if len(a2) != EXPECTED_A2_ROWS or len(val) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Row identity mismatch a2={len(a2)} val={len(val)}")
    human = load_human_cache()
    missing = set(val["id1"]).union(val["id2"]).difference(human["id_to_row"])
    if missing:
        raise RuntimeError(f"Human token cache misses {len(missing)} validation ids")
    row1 = np.asarray([human["id_to_row"][int(x)] for x in val["id1"]], dtype=np.int64)
    row2 = np.asarray([human["id_to_row"][int(x)] for x in val["id2"]], dtype=np.int64)
    labels = val["label"].to_numpy(np.int8)
    cat_ids = human["categories"][row1].astype(np.int64)

    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    base = AutoModel.from_pretrained(base_dir, torch_dtype=torch.float32)
    peft = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    head = load_head(classifier_path, base.config.hidden_size)
    source = Model(peft, head).to(device).eval()

    probe_n = 64
    probe_old_fp32 = score(source, row1[:probe_n], row2[:probe_n], human, human,
                           tokenizer, device, autocast=False)
    source_val = score(source, row1, row2, human, human, tokenizer, device, autocast=True)
    source_metrics = metrics(source_val, labels, cat_ids, human["category_vocab"])
    np.save(OUT / "val_logits_unmerged.npy", source_val)
    progress.update(phase="merge", source_metrics=source_metrics)
    atomic_json(OUT / "run_status.json", progress)

    merged_encoder = source.encoder.merge_and_unload()
    merged_model = Model(merged_encoder, source.head).to(device).eval()
    probe_new_fp32 = score(merged_model, row1[:probe_n], row2[:probe_n], human, human,
                           tokenizer, device, autocast=False)
    merged_val = score(merged_model, row1, row2, human, human, tokenizer, device, autocast=True)
    merged_metrics = metrics(merged_val, labels, cat_ids, human["category_vocab"])
    np.save(OUT / "val_logits_merged.npy", merged_val)

    fp32_diff = np.abs(probe_old_fp32 - probe_new_fp32)
    runtime_diff = np.abs(source_val - merged_val)
    report = {
        "status": "complete",
        "fp32_probe_rows": probe_n,
        "fp32_max_abs_logit_diff": float(fp32_diff.max()),
        "fp32_mean_abs_logit_diff": float(fp32_diff.mean()),
        "runtime_dtype": "cuda_fp16_autocast",
        "runtime_max_abs_logit_diff": float(runtime_diff.max()),
        "runtime_mean_abs_logit_diff": float(runtime_diff.mean()),
        "source_metrics": source_metrics,
        "merged_metrics": merged_metrics,
        "macro_ap_abs_delta": abs(source_metrics["macro_ap"] - merged_metrics["macro_ap"]),
        "micro_ap_abs_delta": abs(source_metrics["micro_ap"] - merged_metrics["micro_ap"]),
    }
    report["gate_passed"] = bool(
        report["fp32_max_abs_logit_diff"] <= FP32_MAX_TOL
        and report["fp32_mean_abs_logit_diff"] <= FP32_MEAN_TOL
        and report["runtime_max_abs_logit_diff"] <= RUNTIME_MAX_TOL
        and report["runtime_mean_abs_logit_diff"] <= RUNTIME_MEAN_TOL
        and report["macro_ap_abs_delta"] <= AP_TOL
        and report["micro_ap_abs_delta"] <= AP_TOL
    )
    atomic_json(OUT / "equivalence_report.json", report)
    if not report["gate_passed"]:
        raise RuntimeError(f"Merge equivalence gate failed: {report}")

    merged_dir = OUT / "merged_encoder"
    merged_encoder.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    torch.save({k: v.detach().cpu() for k, v in source.head.state_dict().items()}, OUT / "classifier.pt")
    del source_val, merged_val, probe_old_fp32, probe_new_fp32
    gc.collect()

    items = pd.read_parquet(items_path, columns=["id", "category", "attributes"])
    synthetic_cache, synthetic_index = build_synthetic_cache(a2, items, tokenizer)
    del items
    gc.collect()

    a2_row1 = np.asarray([human["id_to_row"][int(x)] for x in a2["id1"]], dtype=np.int64)
    a2_row2 = np.asarray([human["id_to_row"][int(x)] for x in a2["id2"]], dtype=np.int64)
    syn_rows = synthetic_index["synthetic_row"].to_numpy(np.int64)
    old_logits = score(merged_model, a2_row1, a2_row2, human, human, tokenizer, device, True)
    new_logits = score(merged_model, a2_row1, syn_rows, human, synthetic_cache, tokenizer, device, True)
    diagnostic = synthetic_index.copy()
    diagnostic["old_logit"] = old_logits
    diagnostic["new_logit"] = new_logits
    diagnostic["delta_logit"] = new_logits - old_logits
    diagnostic["old_probability"] = 1.0 / (1.0 + np.exp(-old_logits))
    diagnostic["new_probability"] = 1.0 / (1.0 + np.exp(-new_logits))
    diagnostic.to_parquet(OUT / "a2_frozen_rescore.parquet", index=False)
    atomic_json(OUT / "a2_frozen_rescore_summary.json", {
        "status": "complete", "rows": len(diagnostic),
        "positive_delta_fraction": float((diagnostic["delta_logit"] > 0).mean()),
        "delta_mean": float(diagnostic["delta_logit"].mean()),
        "delta_median": float(diagnostic["delta_logit"].median()),
        "filtered_by_hardness": False,
    })

    output_identity = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"run_status.json"}:
            output_identity[str(path.relative_to(OUT))] = {"bytes": path.stat().st_size,
                                                           "sha256": sha256_file(path)}
    atomic_json(OUT / "output_identity.json", output_identity)
    atomic_json(OUT / "run_status.json", {"status": "complete", "phase": "done",
                                            "equivalence_gate_passed": True,
                                            "elapsed_seconds": time.perf_counter() - START})
    log("DONE", f"complete in {time.perf_counter()-START:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(OUT / "failure.json", {"status": "error", "type": type(exc).__name__,
                                            "message": str(exc), "traceback": traceback.format_exc(),
                                            "elapsed_seconds": time.perf_counter() - START})
        atomic_json(OUT / "run_status.json", {"status": "error", "phase": "failed",
                                                "message": str(exc)})
        raise
