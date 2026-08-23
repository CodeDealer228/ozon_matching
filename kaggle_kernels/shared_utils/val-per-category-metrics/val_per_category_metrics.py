"""
Per-category breakdown of the competition metric (macro-averaged PR-AUC) for the
CURRENT solution -- the merged LoRA cross-encoder (maxamsecret/ozon-lora-merged-minilm,
val macro_pr_auc 0.5282 / leaderboard 0.265) -- on the canonical seed-42 val split.

CPU-only and deliberately RAM-light:
  * items_human.parquet (214MB, covers every id in matches.parquet) instead of the
    4.1GB items.parquet;
  * semi-join down to the ~145K ids the val split actually references BEFORE any
    string/truncation work (the full-scale `.select().with_columns()` scan is what
    OOMs -- see CLAUDE.md);
  * plain hash joins + an explicit sort on a row-index column, never
    maintain_order="left" (that is what blew RSS up in the continue-train kernel);
  * inference streams in batches; only the 73K float32 probabilities are kept.
Peak RSS should stay well under ~4GB of the CPU box's 30GB.

Besides the per-category AP it records, for each category:
  * pos_rate -- the AP of a constant/random scorer, i.e. the floor the AP has to
    be read against (it ranges 0.07..0.58 across categories, so raw APs are NOT
    comparable between categories);
  * roc_auc -- prevalence-free ranking quality;
  * ap of a naive name-token-Jaccard scorer -- a "how hard is this category"
    yardstick that costs no forward pass;
  * bootstrap 95% CI on AP -- categories are only ~3.5K pairs each;
  * best F1 / precision at fixed recalls -- practical read of the PR curve;
  * token-budget stats -- how much of the pair text actually fits in the 128-token
    window, sampled per category (the encoder's vocab is English WordPiece, so
    Russian text fragments hard and truncation may bite unevenly per category).

Outputs (in /kaggle/working):
  val_scores.parquet        id1,id2,label,category,prob,naive_jaccard  (73,131 rows)
  per_category_metrics.csv  one row per category
  summary.json              macro/overall numbers + run metadata

The per-pair scores are saved so any further slicing can be done offline without
paying for another forward pass.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import gc
import re
import json
import time
import hashlib

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

# ---------------------------------------------------------------- config
SEED = 42
TEST_SIZE = 0.20

# sha256 (first 16 hex chars) of the sorted "id1-id2" pairs on the val side of
# data/splits/matches_split_seed42.parquet, computed locally from that file.
# The split is re-derived here rather than uploaded as a dataset; this pin turns
# any divergence (sklearn version drift, changed filter order) into a hard
# failure instead of a silently different val set.
EXPECTED_VAL_FINGERPRINT = "efddb3def997f87b"
EXPECTED_VAL_ROWS = 73131

# Paths are env-overridable purely so this exact file can be smoke-tested
# locally (on a few thousand pairs) before being pushed -- the Kaggle run uses
# the defaults and takes no other code path.
DATA_DIR = os.environ.get("OZON_DATA_DIR", "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching")
MERGED_DIR = os.environ.get("OZON_MERGED_DIR", "/kaggle/input/datasets/maxamsecret/ozon-lora-merged-minilm")
MATCHES_PATH = os.path.join(DATA_DIR, "matches.parquet")
ITEMS_PATH = os.path.join(DATA_DIR, "items_human.parquet")
CLASSIFIER_PATH = os.environ.get("OZON_CLASSIFIER_PATH", os.path.join(MERGED_DIR, "classifier.pt"))

WORKING_DIR = os.environ.get(
    "OZON_WORKING_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
)
# 0 = full val split (the real run). Anything else truncates it after the
# fingerprint check, for local smoke tests only.
LIMIT_PAIRS = int(os.environ.get("OZON_LIMIT_PAIRS", "0"))

# Must byte-match what the LoRA checkpoint was trained on (CLAUDE.md).
MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64
MAX_LENGTH = 128

BATCH_SIZE = 256
N_BOOTSTRAP = 500
RECALL_TARGETS = (0.5, 0.8, 0.9)
TOKEN_SAMPLE_PER_CAT = 200

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


n_threads = os.cpu_count() or 4
torch.set_num_threads(n_threads)
log(f"torch={torch.__version__} polars={pl.__version__} threads={n_threads} cuda={torch.cuda.is_available()}")

for _root, _dirs, _files in os.walk("/kaggle/input"):
    if _root[len("/kaggle/input"):].count(os.sep) <= 3:
        log(f"INPUT {_root} -> dirs={_dirs} files={_files}")

# ------------------------------------------------- 1. reproduce the val split
log("rebuilding canonical seed-42 split ...")
matches = (
    pl.read_parquet(MATCHES_PATH)
    .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    .with_columns([
        pl.col("target").fill_null(0).cast(pl.Float32),
        (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
    ])
    .select(["id1", "id2", "label"])
)

labels_all = matches.get_column("label").to_numpy().astype(np.int64)
indices = np.arange(len(labels_all), dtype=np.int64)
_, val_idx = train_test_split(
    indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels_all
)

# Row selection via an explicit index column + semi-join: version-agnostic
# across polars releases, unlike df[np.ndarray] fancy indexing.
matches = matches.with_columns(pl.Series("orig_idx", indices, dtype=pl.Int64))
val = (
    matches
    .join(pl.DataFrame({"orig_idx": np.sort(val_idx)}), on="orig_idx", how="semi")
    .sort("orig_idx")
    .select(["id1", "id2", "label"])
)
del matches, labels_all, indices, val_idx
gc.collect()

val_fingerprint = hashlib.sha256(
    ",".join(
        f"{a}-{b}" for a, b in sorted(zip(val.get_column("id1").to_list(),
                                          val.get_column("id2").to_list()))
    ).encode("utf-8")
).hexdigest()[:16]

log(f"val rows={val.height:,} positives={int(val.get_column('label').sum()):,} "
    f"fingerprint={val_fingerprint}")
assert val.height == EXPECTED_VAL_ROWS, f"val rows {val.height} != {EXPECTED_VAL_ROWS}"
assert val_fingerprint == EXPECTED_VAL_FINGERPRINT, (
    f"val fingerprint {val_fingerprint} != {EXPECTED_VAL_FINGERPRINT} -- the re-derived "
    f"split does NOT match data/splits/matches_split_seed42.parquet, so scores would not "
    f"be comparable with previous runs. Upload the split file instead of re-deriving it."
)
log("fingerprint matches data/splits/matches_split_seed42.parquet")

if LIMIT_PAIRS:
    # val rows sit in original matches order, i.e. categories are interleaved,
    # so a head() slice still touches all 20 of them.
    val = val.head(LIMIT_PAIRS)
    log(f"SMOKE TEST: truncated val to {val.height:,} pairs -- numbers are NOT the real metric")

n_expected = val.height

# ------------------------------------------------- 2. item texts + categories
needed_ids = pl.concat([
    val.select(pl.col("id1").alias("id")),
    val.select(pl.col("id2").alias("id")),
]).unique()
log(f"unique item ids referenced by val: {needed_ids.height:,}")

# Semi-join FIRST, truncate after -- the truncation must not force a decode of
# the full `attributes` column for every row in the file.
items = (
    pl.scan_parquet(ITEMS_PATH)
    .select(["id", "name", "category", "attributes"])
    .join(needed_ids.lazy(), on="id", how="semi")
    .collect()
)
log(f"items rows loaded: {items.height:,}")
assert items.height == needed_ids.height, "some val ids are missing from items_human.parquet"

items = items.with_columns([
    pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS).alias("name_t"),
    pl.col("category").fill_null("").cast(pl.Utf8, strict=False).alias("category_raw"),
    pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_CATEGORY_CHARS).alias("category_t"),
    pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS).alias("attr_t"),
]).with_columns(
    (
        pl.lit("Name: ") + pl.col("name_t")
        + pl.lit(" Category: ") + pl.col("category_t")
        + pl.lit(" Attributes: ") + pl.col("attr_t")
    ).str.slice(0, MAX_ITEM_CHARS).alias("text")
).select(["id", "text", "name_t", "category_raw"])

log(f"distinct raw categories among val items: {items.get_column('category_raw').n_unique()}")

val = val.with_columns(pl.Series("row_index", np.arange(val.height), dtype=pl.Int64))
pairs = (
    val
    .join(items.rename({"id": "id1", "text": "text1", "name_t": "name1", "category_raw": "cat1"}),
          on="id1", how="left")
    .join(items.rename({"id": "id2", "text": "text2", "name_t": "name2", "category_raw": "cat2"}),
          on="id2", how="left")
    .sort("row_index")
)
assert pairs.height == n_expected, f"join changed row count: {pairs.height} != {n_expected}"
assert pairs.get_column("text1").null_count() == 0
assert pairs.get_column("text2").null_count() == 0

cat_agree = int((pairs.get_column("cat1") == pairs.get_column("cat2")).sum())
log(f"cat1 == cat2 on {cat_agree:,}/{pairs.height:,} pairs ({100.0 * cat_agree / pairs.height:.3f}%)")

del items, needed_ids, val
gc.collect()

texts1 = pairs.get_column("text1").to_list()
texts2 = pairs.get_column("text2").to_list()
names1 = pairs.get_column("name1").to_list()
names2 = pairs.get_column("name2").to_list()
labels = pairs.get_column("label").to_numpy().astype(np.int64)
categories = np.asarray(pairs.get_column("cat1").to_list())
pair_ids = pairs.select(["id1", "id2"])
del pairs
gc.collect()

# ------------------------------------------------- 3. naive lexical reference
# Not a competitor model -- just a prevalence-independent yardstick for "how hard
# is this category", so a low AP can be attributed to either a hard category or a
# weak model. Token Jaccard over the same truncated `name` the encoder sees.
log("computing naive name-Jaccard reference ...")
_token_re = re.compile(r"\w+", flags=re.UNICODE)

naive = np.empty(len(names1), dtype=np.float32)
for i, (a, b) in enumerate(zip(names1, names2)):
    ta = set(_token_re.findall(a.lower()))
    tb = set(_token_re.findall(b.lower()))
    union = len(ta | tb)
    naive[i] = (len(ta & tb) / union) if union else 0.0
del names1, names2
gc.collect()

# ------------------------------------------------- 4. model + inference
log("loading merged LoRA encoder + classifier head ...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

_v = hf_logging.get_verbosity()
hf_logging.set_verbosity_error()
try:
    encoder = AutoModel.from_pretrained(MERGED_DIR, attn_implementation="sdpa")
except Exception as e:  # older transformers, or sdpa unsupported for this arch
    log(f"sdpa attn unavailable ({type(e).__name__}), falling back to default: {e}")
    encoder = AutoModel.from_pretrained(MERGED_DIR)
hf_logging.set_verbosity(_v)


class CrossEncoderClassifier(nn.Module):
    """Same shape as submits/lora/src/utils.py -- [CLS] -> Linear(hidden, 1)."""

    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(out.last_hidden_state[:, 0, :]).view(-1)


model = CrossEncoderClassifier(encoder, encoder.config.hidden_size)
model.classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
model.eval()

# ---- token-budget diagnostic (before inference, tiny sample, no truncation)
log("sampling untruncated token lengths per category ...")
rng = np.random.default_rng(SEED)
token_stats = {}
for cat in np.unique(categories):
    idx = np.flatnonzero(categories == cat)
    if len(idx) > TOKEN_SAMPLE_PER_CAT:
        idx = rng.choice(idx, TOKEN_SAMPLE_PER_CAT, replace=False)
    enc = tokenizer(
        text=[texts1[j] for j in idx],
        text_pair=[texts2[j] for j in idx],
        padding=False, truncation=False,
    )
    lens = np.array([len(x) for x in enc["input_ids"]], dtype=np.float64)
    token_stats[cat] = {
        "tokens_median": float(np.median(lens)),
        "tokens_p90": float(np.percentile(lens, 90)),
        "frac_pairs_truncated": float((lens > MAX_LENGTH).mean()),
        # what share of the pair's tokens actually reach the encoder
        "median_token_coverage": float(min(1.0, MAX_LENGTH / np.median(lens))),
    }

n_pairs = len(texts1)
log(f"running inference on {n_pairs:,} pairs (batch={BATCH_SIZE}) ...")
probs = np.empty(n_pairs, dtype=np.float32)
t_inf = time.time()
with torch.inference_mode():
    for i in range(0, n_pairs, BATCH_SIZE):
        b1 = texts1[i:i + BATCH_SIZE]
        b2 = texts2[i:i + BATCH_SIZE]
        enc = tokenizer(
            text=b1, text_pair=b2,
            padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        logits = model(**enc)
        probs[i:i + len(b1)] = torch.sigmoid(logits.float()).numpy()

        done = i + len(b1)
        if (i // BATCH_SIZE) % 25 == 0 or done == n_pairs:
            rate = done / max(time.time() - t_inf, 1e-9)
            log(f"inference {done:,}/{n_pairs:,} ({100.0 * done / n_pairs:5.1f}%) "
                f"{rate:.1f} pairs/s ETA {(n_pairs - done) / max(rate, 1e-9) / 60:.1f} min")

inf_sec = time.time() - t_inf
log(f"inference done in {inf_sec:.1f}s ({n_pairs / max(inf_sec, 1e-9):.1f} pairs/s)")
del texts1, texts2, model, encoder
gc.collect()


# ------------------------------------------------- 5. metrics
def _p_at_r(y_true, y_score, targets=RECALL_TARGETS):
    """Best precision reachable at recall >= target -- a practical read of the PR curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    out = {}
    for tgt in targets:
        mask = recall >= tgt
        out[f"precision_at_recall_{tgt}"] = float(precision[mask].max()) if mask.any() else float("nan")
    return out


def _best_f1(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    k = int(np.argmax(f1))
    return float(f1[k]), float(thresholds[k]) if k < len(thresholds) else 1.0


def _bootstrap_ap(y_true, y_score, n=N_BOOTSTRAP, seed=SEED):
    """Percentile CI for AP. Each category is only ~3.5K pairs, so a 0.01 gap
    between two categories is well inside the noise and shouldn't be over-read."""
    r = np.random.default_rng(seed)
    m = len(y_true)
    vals = []
    for _ in range(n):
        idx = r.integers(0, m, m)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        vals.append(average_precision_score(yt, y_score[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


log("computing per-category metrics ...")
rows = []
for cat in sorted(np.unique(categories)):
    mask = categories == cat
    y_t, y_p, y_n = labels[mask], probs[mask], naive[mask]

    n = int(mask.sum())
    n_pos = int(y_t.sum())

    # CLAUDE.md: single-class categories are skipped, not scored as 0.
    if len(np.unique(y_t)) < 2:
        log(f"SKIP category {cat!r}: single class present ({n_pos}/{n} positive)")
        continue

    pos_rate = n_pos / n
    ap = float(average_precision_score(y_t, y_p))
    ap_lo, ap_hi = _bootstrap_ap(y_t, y_p)

    row = {
        "category": cat,
        "n_pairs": n,
        "n_pos": n_pos,
        "pos_rate": pos_rate,
        "ap": ap,
        "ap_ci_lo": ap_lo,
        "ap_ci_hi": ap_hi,
        "ap_lift_over_prevalence": ap / pos_rate,
        "roc_auc": float(roc_auc_score(y_t, y_p)),
        "ap_naive_jaccard": float(average_precision_score(y_t, y_n)),
        "roc_auc_naive_jaccard": float(roc_auc_score(y_t, y_n)),
        "mean_prob_pos": float(y_p[y_t == 1].mean()),
        "mean_prob_neg": float(y_p[y_t == 0].mean()),
    }
    row.update(_p_at_r(y_t, y_p))
    row["best_f1"], row["best_f1_threshold"] = _best_f1(y_t, y_p)
    row.update(token_stats[cat])
    rows.append(row)

per_cat = pd.DataFrame(rows).sort_values("ap").reset_index(drop=True)

macro_pr_auc = float(per_cat["ap"].mean())
summary = {
    "model": "ozon-lora-merged-minilm (merged LoRA MiniLM-L12 + Linear head)",
    "split": "canonical seed-42 val (data/splits/matches_split_seed42.parquet)",
    "val_fingerprint": val_fingerprint,
    "val_pairs": int(n_pairs),
    "val_positives": int(labels.sum()),
    "n_categories_evaluated": int(len(per_cat)),
    "macro_pr_auc": macro_pr_auc,
    "macro_pr_auc_naive_jaccard": float(per_cat["ap_naive_jaccard"].mean()),
    "macro_pr_auc_prevalence_floor": float(per_cat["pos_rate"].mean()),
    "macro_roc_auc": float(per_cat["roc_auc"].mean()),
    "overall_ap_pooled": float(average_precision_score(labels, probs)),
    "overall_roc_auc_pooled": float(roc_auc_score(labels, probs)),
    "ap_min": float(per_cat["ap"].min()),
    "ap_max": float(per_cat["ap"].max()),
    "ap_spread": float(per_cat["ap"].max() - per_cat["ap"].min()),
    "ap_std": float(per_cat["ap"].std(ddof=0)),
    "worst_category": str(per_cat.iloc[0]["category"]),
    "best_category": str(per_cat.iloc[-1]["category"]),
    "corr_ap_vs_pos_rate": float(np.corrcoef(per_cat["ap"], per_cat["pos_rate"])[0, 1]),
    "corr_ap_vs_naive_ap": float(np.corrcoef(per_cat["ap"], per_cat["ap_naive_jaccard"])[0, 1]),
    "cat1_equals_cat2_pairs": cat_agree,
    "batch_size": BATCH_SIZE,
    "n_bootstrap": N_BOOTSTRAP,
    "inference_sec": round(inf_sec, 1),
    "inference_pairs_per_sec": round(n_pairs / max(inf_sec, 1e-9), 1),
    "runtime_sec": round(time.time() - t_start, 1),
}

# ------------------------------------------------- 6. save + print
scores_path = os.path.join(WORKING_DIR, "val_scores.parquet")
pl.DataFrame({
    "id1": pair_ids.get_column("id1"),
    "id2": pair_ids.get_column("id2"),
    "label": labels,
    "category": categories,
    "prob": probs,
    "naive_jaccard": naive,
}).write_parquet(scores_path)

per_cat.to_csv(os.path.join(WORKING_DIR, "per_category_metrics.csv"), index=False)
with open(os.path.join(WORKING_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

print("\n" + "=" * 120)
print("PER-CATEGORY PR-AUC (ascending) -- merged LoRA cross-encoder, seed-42 val split")
print("=" * 120)
show = per_cat[[
    "category", "n_pairs", "n_pos", "pos_rate", "ap", "ap_ci_lo", "ap_ci_hi",
    "ap_lift_over_prevalence", "roc_auc", "ap_naive_jaccard", "best_f1",
    "precision_at_recall_0.5", "precision_at_recall_0.8",
    "frac_pairs_truncated", "median_token_coverage",
]].copy()
for c in show.columns:
    if show[c].dtype.kind == "f":
        show[c] = show[c].round(4)
print(show.to_string(index=False))

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)
print(json.dumps(summary, indent=2, ensure_ascii=False))
log(f"wrote {scores_path}, per_category_metrics.csv, summary.json")
