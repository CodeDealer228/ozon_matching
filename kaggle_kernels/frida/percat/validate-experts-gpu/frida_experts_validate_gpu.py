"""Validate 20 formatting-v4 FRIDA specialists on the item-disjoint holdout."""
import gc
import json
import math
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                       "peft", "safetensors"])

import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import average_precision_score
from torch import nn
from transformers import T5Config, T5EncoderModel

MODEL_NAME = "ai-forever/FRIDA"
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 224
HEAD_HIDDEN = (1024, 512, 128)
HEAD_DROPOUT = 0.2
EXPECTED_N_ITEMS = 711_304
EXPECTED_VAL = 109_648
EXPECTED_FINGERPRINT = "0fd633cbb571c59a"
START_BATCH = 128
OUT = "/kaggle/working/frida_experts_validation.json"
EXPECTED_CATEGORIES = [
    "Автотовары", "Аптека", "Бытовая техника", "Бытовая химия",
    "Галантерея и аксессуары", "Детские товары", "Дом и сад",
    "Канцелярские товары", "Красота и гигиена", "Мебель",
    "Музыкальные инструменты", "Обувь", "Одежда", "Продукты питания",
    "Спорт и отдых", "Строительство и ремонт", "Товары для животных",
    "Хобби и творчество", "Электроника", "Ювелирные изделия",
]


def find_file(filename):
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            found.append(os.path.join(root, filename))
    if len(found) != 1:
        raise RuntimeError(f"expected one {filename}, found {found}")
    return found[0]


def find_human_cache():
    for root, _dirs, files in os.walk("/kaggle/input"):
        if not {"cache_meta.json", "ids.bin", "input_ids.bin",
                "attention_mask.bin"} <= set(files):
            continue
        with open(os.path.join(root, "cache_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("format_version") == 4 and meta.get("scope") == "human"
                and meta.get("n_items") == EXPECTED_N_ITEMS
                and meta.get("max_length") == ITEM_MAX_LENGTH
                and meta.get("prefix") == ""
                and meta.get("category_in_text") is False):
            return root, meta
    raise RuntimeError("Tokenize Items v4 human cache not found")


def find_checkpoint_roots():
    experts = []
    base = []
    formatting = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "cat00_last" in _dirs:
            experts.append(root)
        state_path = os.path.join(root, "train_state.json")
        if not os.path.isfile(state_path):
            continue
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        if (state.get("experiment") == "continue_c_llm3m_2ep"
                and int(state.get("global_step", -1)) == 13_474
                and os.path.basename(root) == "last"
                and {"encoder.pt", "head.pt"} <= set(files)):
            base.append(root)
        if (state.get("experiment") == "frida_format_adapter_v4_c_last"
                and os.path.basename(root) == "best"
                and os.path.isfile(os.path.join(root, "adapter", "adapter_model.safetensors"))):
            formatting.append(root)
    if not (len(experts) == len(base) == len(formatting) == 1):
        raise RuntimeError(
            f"artifact discovery failed: experts={experts} base={base} formatting={formatting}")
    for cat_id in range(20):
        checkpoint = os.path.join(experts[0], f"cat{cat_id:02d}_last")
        required = [os.path.join(checkpoint, "adapter", "adapter_model.safetensors"),
                    os.path.join(checkpoint, "head.pt"),
                    os.path.join(checkpoint, "train_state.json")]
        if not all(os.path.isfile(path) for path in required):
            raise RuntimeError(f"incomplete expert checkpoint: {checkpoint}")
    return experts[0], base[0], formatting[0]


class FullSftHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        blocks = []
        d = hidden_dim
        for h in HEAD_HIDDEN:
            blocks.append(nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h),
                                        nn.GELU(), nn.Dropout(HEAD_DROPOUT)))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = FullSftHead(encoder.config.d_model)

    def forward(self, input_ids, attention_mask):
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(hidden.float()).view(-1)


def load_model(base_dir, formatting_dir, first_expert, device):
    print(f"[MODEL] c-base={base_dir}", flush=True)
    base = T5EncoderModel(T5Config.from_pretrained(MODEL_NAME))
    state = torch.load(os.path.join(base_dir, "encoder.pt"), map_location="cpu",
                       weights_only=True)
    incompatible = base.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"c-base mismatch: {incompatible}")
    del state
    print(f"[MODEL] merge formatting adapter={formatting_dir}", flush=True)
    base = PeftModel.from_pretrained(base, os.path.join(formatting_dir, "adapter"),
                                     is_trainable=False).merge_and_unload()
    base.to(dtype=torch.float16)
    config = PeftConfig.from_pretrained(os.path.join(first_expert, "adapter"))
    model = Classifier(get_peft_model(base, config)).eval().to(device)
    return model


def load_expert(model, checkpoint):
    adapter = load_safetensors(os.path.join(checkpoint, "adapter", "adapter_model.safetensors"))
    if any(not torch.isfinite(value).all() for value in adapter.values()):
        raise RuntimeError(f"non-finite adapter: {checkpoint}")
    result = set_peft_model_state_dict(model.encoder, adapter)
    missing_lora = [key for key in (getattr(result, "missing_keys", None) or [])
                    if "lora_" in key]
    unexpected = getattr(result, "unexpected_keys", None) or []
    if missing_lora or unexpected:
        raise RuntimeError(f"adapter mismatch: missing={missing_lora} unexpected={unexpected}")
    head = torch.load(os.path.join(checkpoint, "head.pt"), map_location="cpu",
                      weights_only=True)
    model.classifier.load_state_dict(head, strict=True)
    if any(not torch.isfinite(value).all() for value in head.values()):
        raise RuntimeError(f"non-finite head: {checkpoint}")


def allocate_pair_budget(len1, len2):
    half = PAIR_MAX_LENGTH // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, PAIR_MAX_LENGTH - len1
    if len2 <= half:
        return PAIR_MAX_LENGTH - len2, len2
    return half, PAIR_MAX_LENGTH - half


def make_batch(cache, row1, row2, lengths, pad_id):
    n_rows = len(row1)
    ids = np.full((n_rows, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    mask = np.zeros((n_rows, PAIR_MAX_LENGTH), dtype=np.int64)
    for i, (r1, r2) in enumerate(zip(row1, row2)):
        len1, len2 = int(lengths[r1]), int(lengths[r2])
        keep1, keep2 = allocate_pair_budget(len1, len2)
        combined = np.concatenate((cache[r1, :keep1], cache[r2, :keep2]))
        ids[i, :len(combined)] = combined
        mask[i, :len(combined)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


def validate_category(model, rows1, rows2, labels, cache, lengths, pad_id, device):
    batch_size = START_BATCH
    offset = 0
    predictions = []
    while offset < len(labels):
        stop = min(offset + batch_size, len(labels))
        ids, mask = make_batch(cache, rows1[offset:stop], rows2[offset:stop], lengths, pad_id)
        try:
            with torch.inference_mode():
                logits = model(ids.to(device, non_blocking=True),
                               mask.to(device, non_blocking=True))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            offset = stop
        except torch.OutOfMemoryError:
            del ids, mask
            torch.cuda.empty_cache()
            batch_size //= 2
            if batch_size < 4:
                raise
            print(f"[OOM] retry batch={batch_size}", flush=True)
    probabilities = np.concatenate(predictions)
    if len(probabilities) != len(labels) or not np.isfinite(probabilities).all():
        raise RuntimeError("invalid predictions")
    return float(average_precision_score(labels, probabilities)), batch_size


def main():
    started = time.time()
    assert allocate_pair_budget(20, 300) == (20, 204)
    assert allocate_pair_budget(300, 20) == (204, 20)
    assert sum(allocate_pair_budget(300, 300)) == PAIR_MAX_LENGTH
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda")
    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)
    experts_root, base_dir, formatting_dir = find_checkpoint_roots()
    cache_root, cache_meta = find_human_cache()

    split = pd.read_parquet(find_file("matches_item_disjoint_70_30_seed42.parquet"))
    val = split.loc[split["split"].eq("val"), ["id1", "id2", "label", "category"]].copy()
    categories = sorted(split["category"].dropna().unique().tolist())
    if len(val) != EXPECTED_VAL or categories != EXPECTED_CATEGORIES:
        raise RuntimeError(f"wrong validation contract: n={len(val)} categories={categories}")
    import hashlib
    fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val["id1"], val["id2"]))).encode()
    ).hexdigest()[:16]
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"wrong val fingerprint: {fingerprint}")

    n_items = cache_meta["n_items"]
    ids = np.memmap(os.path.join(cache_root, "ids.bin"), dtype=np.int64, mode="r",
                    shape=(n_items,))
    order = np.argsort(ids, kind="stable")
    sorted_ids = np.asarray(ids)[order]
    cache = np.memmap(os.path.join(cache_root, "input_ids.bin"), dtype=np.int32, mode="r",
                      shape=(n_items, ITEM_MAX_LENGTH))
    attention = np.memmap(os.path.join(cache_root, "attention_mask.bin"), dtype=np.int8,
                          mode="r", shape=(n_items, ITEM_MAX_LENGTH))

    def lookup(values):
        values = np.asarray(values, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, values)
        clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        if not np.all(sorted_ids[clipped] == values):
            raise RuntimeError("validation item missing from v4 cache")
        return order[clipped]

    row1 = lookup(val["id1"])
    row2 = lookup(val["id2"])
    referenced = np.unique(np.concatenate((row1, row2)))
    lengths = np.full(n_items, -1, dtype=np.int16)
    for start in range(0, len(referenced), 100_000):
        selected = referenced[start:start + 100_000]
        lengths[selected] = np.asarray(attention[selected]).sum(axis=1).astype(np.int16)
    if np.any(lengths[referenced] < 2):
        raise RuntimeError("invalid cached token length")
    padded = referenced[lengths[referenced] < ITEM_MAX_LENGTH]
    pad_id = int(cache[padded[0], int(lengths[padded[0]])])
    val["row1"] = row1
    val["row2"] = row2

    model = load_model(base_dir, formatting_dir,
                       os.path.join(experts_root, "cat00_last"), device)
    results = {}
    for cat_id, category in enumerate(categories):
        checkpoint = os.path.join(experts_root, f"cat{cat_id:02d}_last")
        load_expert(model, checkpoint)
        subset = val.loc[val["category"].eq(category)]
        score, final_batch = validate_category(
            model, subset["row1"].to_numpy(), subset["row2"].to_numpy(),
            subset["label"].to_numpy(dtype=np.float32), cache, lengths, pad_id, device)
        results[category] = {"cat_id": cat_id, "n": len(subset),
                             "val_pr_auc": score, "batch_size": final_batch,
                             "checkpoint": f"cat{cat_id:02d}_last"}
        print(f"[VAL] {cat_id:02d} {category}: n={len(subset):,} PR-AUC={score:.6f}",
              flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    macro = float(np.mean([entry["val_pr_auc"] for entry in results.values()]))
    report = {
        "macro_pr_auc": macro,
        "n_val": len(val),
        "val_fingerprint": fingerprint,
        "split": "item-disjoint 70/30 seed42",
        "pair_max_length": PAIR_MAX_LENGTH,
        "token_cache": "Tokenize Items v4 human",
        "experts_dataset": "maxamsecret/frida-experts",
        "elapsed_seconds": time.time() - started,
        "per_category": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[DONE] macro PR-AUC={macro:.6f}; report={OUT}", flush=True)


if __name__ == "__main__":
    main()
