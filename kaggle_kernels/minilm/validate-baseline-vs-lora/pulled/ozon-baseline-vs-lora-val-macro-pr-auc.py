"""
Validate macro PR-AUC (per-category, matches competition metric) on the
val split of matches.parquet for two variants:

  1) baseline_no_lora — frozen cross-encoder/ms-marco-MiniLM-L-12-v2,
     [CLS] embeddings -> StandardScaler -> LogisticRegression
     (same shape as submits/baseline).
  2) lora_finetuned    — LoRA-tuned encoder (maxamsecret/ozon-lora-minilm-finetuned,
     best/ checkpoint) + trained Linear head, end-to-end forward pass.

Val split is reproduced with the exact same code (seed, stratify, id filtering,
text-building formula) as notebooks/cross-encoder-sft-pipline (3).ipynb so both
variants and the LoRA training run share one val set.
"""

import os
import sys
import gc
import json
import subprocess

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import AutoTokenizer, AutoModel

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

try:
    import peft.import_utils
    import peft.tuners.lora.torchao

    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    print("[INFO] torchao disabled for PEFT")
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel

SEED = 42
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

ITEMS_PATH = "/kaggle/input/ozon-ecup-2026-matching/items.parquet"
MATCHES_PATH = "/kaggle/input/ozon-ecup-2026-matching/matches.parquet"
LORA_DIR = "/kaggle/input/ozon-lora-minilm-finetuned/lora_mini_self_attention_memmap/best"

MAX_LENGTH = 128
MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64

BATCH_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device}")
print(f"[INFO] torch={torch.__version__}")

# =====================================================================
# Reproduce the exact train/val split used during LoRA training
# =====================================================================
print("[DATA] Loading matches.parquet ...")

matches_pl = pl.read_parquet(MATCHES_PATH)
matches_pl = (
    matches_pl
    .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    .with_columns([
        pl.col("target").fill_null(0).cast(pl.Float32),
        (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
    ])
    .select(["id1", "id2", "label"])
)
matches_pd = matches_pl.to_pandas()
del matches_pl
gc.collect()

labels_all = matches_pd["label"].to_numpy().astype(np.int64)
indices = np.arange(len(labels_all), dtype=np.int64)

train_idx, val_idx = train_test_split(
    indices, test_size=0.20, random_state=SEED, stratify=labels_all
)

train_pd = matches_pd.iloc[train_idx].reset_index(drop=True)
val_pd = matches_pd.iloc[val_idx].reset_index(drop=True)

print(f"[DATA] train rows: {len(train_pd):,} | val rows: {len(val_pd):,}")

needed_ids = pd.unique(
    pd.concat([matches_pd["id1"], matches_pd["id2"]])
).tolist()

del matches_pd, labels_all, indices, train_idx, val_idx
gc.collect()

# =====================================================================
# items -> text (identical truncation/concat formula to training)
# =====================================================================
print("[DATA] Loading items.parquet for needed ids ...")

items = (
    pl.scan_parquet(ITEMS_PATH)
    .filter(pl.col("id").is_in(needed_ids))
    .select(["id", "name", "category", "attributes"])
    .collect()
)
print(f"[DATA] items loaded: {items.height:,}")

items = items.with_columns([
    pl.col("name").fill_null("").cast(pl.Utf8).str.slice(0, MAX_NAME_CHARS),
    pl.col("category").fill_null("").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS),
    pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS),
])

id_to_category = dict(zip(
    items.get_column("id").to_list(),
    items.get_column("category").to_list()
))

items = items.with_columns(
    (
        pl.lit("Name: ") + pl.col("name") +
        pl.lit(" Category: ") + pl.col("category") +
        pl.lit(" Attributes: ") + pl.col("attributes")
    ).str.slice(0, MAX_ITEM_CHARS).alias("text")
)

id_to_text = dict(zip(
    items.get_column("id").to_list(),
    items.get_column("text").to_list()
))

del items, needed_ids
gc.collect()


def build_pairs(pairs_pd):
    text1 = pairs_pd["id1"].map(id_to_text)
    text2 = pairs_pd["id2"].map(id_to_text)
    keep = (text1.notna() & text2.notna()).to_numpy()
    dropped = int((~keep).sum())
    if dropped:
        print(f"[WARN] dropping {dropped} pairs with missing item text")
    return (
        text1[keep].tolist(),
        text2[keep].tolist(),
        pairs_pd.loc[keep, "label"].to_numpy().astype(np.int64),
        keep,
    )


train_t1, train_t2, train_labels, _ = build_pairs(train_pd)
val_t1, val_t2, val_labels, val_keep = build_pairs(val_pd)

val_categories = val_pd["id1"].map(id_to_category).fillna("UNKNOWN").to_numpy()
val_categories = val_categories[val_keep]

assert len(val_t1) == len(val_categories) == len(val_labels)
print(f"[DATA] val pairs usable: {len(val_t1):,} | unique categories: {len(set(val_categories))}")


def macro_pr_auc_by_category(y_true, y_prob, categories):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    categories = np.asarray(categories)

    overall_ap = float(average_precision_score(y_true, y_prob))

    per_category_aps = []
    for cat in np.unique(categories):
        mask = categories == cat
        y_t, y_p = y_true[mask], y_prob[mask]
        if len(np.unique(y_t)) < 2:
            continue
        per_category_aps.append(average_precision_score(y_t, y_p))

    if not per_category_aps:
        return float("nan"), overall_ap, 0

    return float(np.mean(per_category_aps)), overall_ap, len(per_category_aps)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_cls_embeddings(model, texts1, texts2, batch_size=BATCH_SIZE):
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts1), batch_size):
            b1 = texts1[i:i + batch_size]
            b2 = texts2[i:i + batch_size]
            enc = tokenizer(
                text=b1, text_pair=b2,
                padding="max_length", truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            out_chunks.append(cls)
    return np.concatenate(out_chunks, axis=0)


def run_forward_logits(model, texts1, texts2, batch_size=BATCH_SIZE):
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts1), batch_size):
            b1 = texts1[i:i + batch_size]
            b2 = texts2[i:i + batch_size]
            enc = tokenizer(
                text=b1, text_pair=b2,
                padding="max_length", truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(**enc)
            out_chunks.append(logits.float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


results = {}

# =====================================================================
# 1) baseline_no_lora: frozen encoder [CLS] -> StandardScaler -> LogReg
# =====================================================================
print("=" * 80)
print("[BASELINE] Loading frozen base encoder (no LoRA) ...")
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

print("[BASELINE] Extracting train embeddings ...")
train_emb = extract_cls_embeddings(base_model, train_t1, train_t2)
print("[BASELINE] Extracting val embeddings ...")
val_emb = extract_cls_embeddings(base_model, val_t1, val_t2)

print("[BASELINE] Fitting StandardScaler + LogisticRegression on train ...")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=1234)),
])
clf.fit(train_emb, train_labels)

val_probs_baseline = clf.predict_proba(val_emb)[:, 1]
macro_base, overall_base, n_cat_base = macro_pr_auc_by_category(val_labels, val_probs_baseline, val_categories)

print(
    f"[BASELINE] macro_pr_auc(per-category)={macro_base:.5f} | "
    f"overall_ap={overall_base:.5f} | n_categories={n_cat_base}"
)
results["baseline_no_lora"] = {
    "macro_pr_auc": macro_base,
    "overall_ap": overall_base,
    "n_categories_evaluated": n_cat_base,
}

del base_model, train_emb, val_emb
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =====================================================================
# 2) lora_finetuned: PEFT-adapted encoder + trained Linear head
# =====================================================================
print("=" * 80)
print("[LORA] Loading base encoder + PEFT adapter ...")

base_model2 = AutoModel.from_pretrained(MODEL_NAME)
peft_model = PeftModel.from_pretrained(base_model2, LORA_DIR).to(device)
hidden_dim = base_model2.config.hidden_size

classifier_head = nn.Linear(hidden_dim, 1)
state_dict = torch.load(os.path.join(LORA_DIR, "classifier.pt"), map_location=device)
classifier_head.load_state_dict(state_dict)
classifier_head.to(device)


class CrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).view(-1)


lora_model = CrossEncoderClassifier(peft_model, classifier_head).to(device)

print("[LORA] Running forward pass on val ...")
val_logits_lora = run_forward_logits(lora_model, val_t1, val_t2)
val_probs_lora = 1.0 / (1.0 + np.exp(-val_logits_lora))

macro_lora, overall_lora, n_cat_lora = macro_pr_auc_by_category(val_labels, val_probs_lora, val_categories)
print(
    f"[LORA] macro_pr_auc(per-category)={macro_lora:.5f} | "
    f"overall_ap={overall_lora:.5f} | n_categories={n_cat_lora}"
)

results["lora_finetuned"] = {
    "macro_pr_auc": macro_lora,
    "overall_ap": overall_lora,
    "n_categories_evaluated": n_cat_lora,
}

# =====================================================================
# Summary
# =====================================================================
delta = results["lora_finetuned"]["macro_pr_auc"] - results["baseline_no_lora"]["macro_pr_auc"]
results["delta_macro_pr_auc"] = float(delta)
results["val_pairs"] = len(val_t1)
results["val_unique_categories"] = int(len(set(val_categories)))

print("=" * 80)
print("[SUMMARY]")
print(json.dumps(results, indent=2))
print(f"LoRA vs baseline delta (macro_pr_auc): {delta:+.5f}")

with open("/kaggle/working/val_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("[DONE] results saved to /kaggle/working/val_comparison_results.json")
