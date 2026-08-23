"""
Feasibility/timing pilot: does a cheap, CPU-only lexical signal already
separate known positive pairs from known negative pairs within ONE
category, before spending any GPU time on hard-negative mining?

Category = "Аптека" (pharmacy), chosen as the pilot deliberately: it's the
HARDEST of the 20 categories by labeled positive rate (7.26%, lowest of
all 20 -- measured directly from matches.parquet + items_human.parquet,
this repo's own local check, not assumed) with 18,285 labeled pairs over
35,620 unique items. Restricting to one category at all is valid because
category1 == category2 for 100.0000% of the 365,654 labeled pairs
(measured, zero exceptions) -- the labeled data never has a cross-category
pair, so per-category analysis loses nothing.

===========================================================================
WHY THIS, INSTEAD OF ANCE-STYLE HARD-NEGATIVE MINING
===========================================================================
ANCE-style mining (score many candidates with the actual model, keep the
high-scoring non-matches) is the standard approach for BI-encoders, where
you precompute one vector per item once and do cheap nearest-neighbor
search across the whole catalog. It does NOT translate cheaply to a
CROSS-encoder like BERTA here: there is no per-item vector, every
candidate pair needs its own full joint forward pass, so "search for
confusable candidates" means O(items^2) cross-encoder calls -- for one
category alone (35,620 items) that's ~634M candidate pairs, wildly
infeasible on CPU and still expensive on GPU for a pilot.

This script instead asks a cheaper question first: does a signal that
costs near-zero compute (TF-IDF cosine, fastText trained from scratch on
just this category's item text, CPU-only, no GPU, no cross-encoder
inference at all) already separate the known positives from the known
negatives? If yes, that signal is itself usable two ways: (a) a cheap
hard-negative-mining tool (retrieve top-cosine neighbors per item, label
whatever's not in the known-positive set as a negative), and/or (b) a
cheap extra feature for the classifier head. If the signal barely
separates them at all, that's useful too -- it means the genuinely hard
cases need something more expensive (the classifier itself, or attribute-
level reasoning) and cheap lexical mining would mostly surface EASY
negatives, not hard ones.

Deliberately does NOT touch BERTA or any cross-encoder/GPU model. Pure
lexical/subword signal only, to isolate what a nearly-free method already
tells you before reaching for anything GPU-based. CPU-only kernel.
"""

import os
import sys
import time
import json
import subprocess

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():7.1f}s] {stage}: {msg}", flush=True)


subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fasttext-wheel"])
_log("SETUP", "fasttext installed")

import numpy as np
import pandas as pd
import polars as pl
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

CATEGORY = "Аптека"

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

MAX_NAME_CHARS, MAX_ATTR_CHARS = 128, 384
# Category deliberately excluded from the text used here -- within a
# single-category pilot it's constant by construction, zero discriminative
# value (same finding CLAUDE.md already documents for the pairwise task
# generally: category is 100% identical between both items of every pair).

FASTTEXT_DIM = 100
FASTTEXT_EPOCH = 5
FASTTEXT_MODEL = "skipgram"
FASTTEXT_MIN_COUNT = 1  # tiny per-category vocab -- don't drop rare words

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "fasttext_probe")


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def build_item_text(row):
    name = (row["name"] or "")[:MAX_NAME_CHARS]
    attrs = (row["attributes"] or "")[:MAX_ATTR_CHARS]
    return f"{name} {attrs}".replace("\n", " ").strip()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
    n_threads = os.cpu_count() or 4
    _log("CONFIG", f"category={CATEGORY!r} fasttext_dim={FASTTEXT_DIM} "
                    f"epoch={FASTTEXT_EPOCH} threads={n_threads}")

    # -------------------------------------------------------------
    # Load + filter to one category
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")

    matches = pl.read_parquet(matches_path)
    matches = matches.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    matches = matches.with_columns((pl.col("target") >= 0.5).cast(pl.Int8).alias("label"))

    items = pl.read_parquet(items_path, columns=["id", "name", "category", "attributes"])
    items_cat = items.filter(pl.col("category") == CATEGORY)
    cat_ids = set(items_cat.get_column("id").to_list())
    _log("DATA", f"loaded matches={matches.height:,} | {CATEGORY} items={len(cat_ids):,} "
                 f"({time.perf_counter() - t0:.1f}s)")

    pairs = matches.filter(pl.col("id1").is_in(cat_ids) & pl.col("id2").is_in(cat_ids))
    _log("DATA", f"{CATEGORY} labeled pairs: {pairs.height:,} "
                 f"(pos_rate={pairs['label'].mean():.4f})")

    # -------------------------------------------------------------
    # Build per-item text (Name + Attributes only, see module docstring)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    items_pd = items_cat.to_pandas()
    items_pd["text"] = items_pd.apply(build_item_text, axis=1)
    id_to_text = dict(zip(items_pd["id"], items_pd["text"]))
    _log("PREP", f"item texts built for {len(id_to_text):,} items ({time.perf_counter() - t0:.1f}s)")

    corpus_path = os.path.join(OUTPUT_DIR, f"corpus_{CATEGORY}.txt")
    t0 = time.perf_counter()
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in items_pd["text"]:
            f.write(text + "\n")
    _log("PREP", f"corpus file written ({time.perf_counter() - t0:.1f}s)")

    # -------------------------------------------------------------
    # Train fastText from scratch on JUST this category's item text
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    ft_model = fasttext.train_unsupervised(
        corpus_path, model=FASTTEXT_MODEL, dim=FASTTEXT_DIM,
        epoch=FASTTEXT_EPOCH, minCount=FASTTEXT_MIN_COUNT, thread=n_threads,
    )
    fasttext_train_s = time.perf_counter() - t0
    _log("FASTTEXT", f"trained ({fasttext_train_s:.1f}s), vocab size={len(ft_model.get_words()):,}")

    t0 = time.perf_counter()
    id_to_ftvec = {
        item_id: ft_model.get_sentence_vector(text.replace("\n", " "))
        for item_id, text in id_to_text.items()
    }
    _log("FASTTEXT", f"item vectors computed ({time.perf_counter() - t0:.1f}s)")

    # -------------------------------------------------------------
    # TF-IDF baseline (near-zero extra cost, sklearn only -- comparison point)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    ordered_ids = items_pd["id"].tolist()
    ordered_texts = items_pd["text"].tolist()
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(ordered_texts)
    id_to_row = {item_id: i for i, item_id in enumerate(ordered_ids)}
    _log("TFIDF", f"fit+transform done, vocab={len(tfidf.vocabulary_):,} "
                  f"({time.perf_counter() - t0:.1f}s)")

    # -------------------------------------------------------------
    # Cosine similarity for every labeled pair, both signals -- fully
    # vectorized (no per-pair Python loop / no per-pair sklearn call):
    # fastText via batched numpy row-wise dot/norm; TF-IDF exploits that
    # TfidfVectorizer's default norm='l2' already unit-normalizes every
    # row, so cosine between two rows is JUST their dot product -- an
    # element-wise sparse multiply + row-sum, no separate normalization
    # or per-pair function-call overhead needed.
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    pairs_pd = pairs.select(["id1", "id2", "label"]).to_pandas()

    vecs1 = np.stack([id_to_ftvec[i] for i in pairs_pd["id1"]]).astype(np.float32)
    vecs2 = np.stack([id_to_ftvec[i] for i in pairs_pd["id2"]]).astype(np.float32)
    dot = (vecs1 * vecs2).sum(axis=1)
    norms = np.linalg.norm(vecs1, axis=1) * np.linalg.norm(vecs2, axis=1)
    ft_cos = dot / np.clip(norms, 1e-9, None)

    rows1 = pairs_pd["id1"].map(id_to_row).to_numpy()
    rows2 = pairs_pd["id2"].map(id_to_row).to_numpy()
    tfidf_cos = np.asarray(
        tfidf_matrix[rows1].multiply(tfidf_matrix[rows2]).sum(axis=1)
    ).flatten()

    pairs_pd["ft_cosine"] = ft_cos.astype(np.float32)
    pairs_pd["tfidf_cosine"] = tfidf_cos.astype(np.float32)
    _log("SIMILARITY", f"cosine computed for {len(pairs_pd):,} pairs, both signals, "
                        f"fully vectorized ({time.perf_counter() - t0:.1f}s)")

    # -------------------------------------------------------------
    # Report: does either signal separate positive from negative?
    # -------------------------------------------------------------
    y_true = pairs_pd["label"].to_numpy()
    results = {"category": CATEGORY, "n_pairs": int(len(pairs_pd)),
               "pos_rate": float(y_true.mean())}

    for sig in ["ft_cosine", "tfidf_cosine"]:
        scores = pairs_pd[sig].to_numpy()
        ap = float(average_precision_score(y_true, scores))
        auc = float(roc_auc_score(y_true, scores))
        pos_mean, neg_mean = scores[y_true == 1].mean(), scores[y_true == 0].mean()
        pos_median, neg_median = np.median(scores[y_true == 1]), np.median(scores[y_true == 0])
        results[sig] = {
            "AP": ap, "ROC_AUC": auc,
            "positive_mean": float(pos_mean), "negative_mean": float(neg_mean),
            "positive_median": float(pos_median), "negative_median": float(neg_median),
        }
        _log("RESULT", f"{sig}: AP={ap:.4f} ROC_AUC={auc:.4f} | "
                       f"pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} "
                       f"pos_median={pos_median:.4f} neg_median={neg_median:.4f}")

    results["timing_s"] = {
        "fasttext_train": fasttext_train_s,
        "total_script": _elapsed(),
    }

    pairs_pd.to_csv(os.path.join(OUTPUT_DIR, f"pair_scores_{CATEGORY}.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    _log("DONE", f"total wall time {_elapsed():.1f}s -- results.json + pair_scores_{CATEGORY}.csv "
                 f"written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
