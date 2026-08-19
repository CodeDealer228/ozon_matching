"""
Fits one TfidfVectorizer PER CATEGORY (20 total) on item text drawn from the
FULL catalog (items.parquet, 13M+ rows) -- reusing the exact
TfidfVectorizer(max_features=20000, ngram_range=(1,2)) recipe already
proven in kaggle_kernels/fasttext-category-similarity-probe (there: TF-IDF
beat fastText, AP 0.166 vs 0.150, on the "Аптека" pilot category) -- just
extended from one pilot category to all 20, and fit on a much larger,
catalog-representative corpus instead of only the ~35K human-labeled items
of that pilot.

===========================================================================
MEMORY DESIGN -- how this stays safe for an 11M+-item run on the judging
stand (1x H100 / 20 CPU / 200GB RAM, tight time budget), the specific
concern that motivated writing this kernel this way
===========================================================================
The naive approach -- precompute and store a TF-IDF VECTOR for every item
in the full catalog -- does not fit the constraint. Even sparse, a
13M-row x 20K-col matrix is a real, large object, and more importantly
there is no reason to ever hold it: at inference time, run.py only ever
needs a score for the specific pairs in that run's input file (at most
~275K pairs, i.e. ~550K item references, not 13M).

So this kernel stores something much smaller instead: 20 FITTED
TfidfVectorizer objects (vocabulary_ dict + idf_ array each, capped at
max_features=20000) -- a few MB per category, ~100-300MB total for all 20,
trivial against the 200GB budget. At inference time (or here, when scoring
the val split below), a pair's tf-idf score is computed on the fly:
vectorizer.transform([text1, text2]) -- fast (proportional to the two
texts' own length, NOT to corpus size, since vocabulary/idf are already
fixed by fitting) -- then cosine similarity, which is just a dot product
because TfidfVectorizer's default norm='l2' already unit-normalizes every
row (same vectorized trick as fasttext-category-similarity-probe:
`vec1.multiply(vec2).sum()`, no separate normalization step, no per-pair
sklearn call in a Python loop for the DOT PRODUCT itself -- only the
.transform() calls are per-batch, and even those batch fine since
TfidfVectorizer.transform() accepts a list of documents at once).

Net effect: this kernel's own OUTPUT is ~20 small vectorizer files, and the
COST MODEL for run.py at inference time is O(pairs), never O(catalog size)
-- the "11M-item" catalog scale only ever matters HERE, once, offline,
during fitting, never at inference.

===========================================================================
FITTING CORPUS
===========================================================================
Each category's vectorizer is fit on up to FIT_SAMPLE_PER_CATEGORY=300,000
items from THAT category (all of them if the category has fewer), sampled
from the full items.parquet catalog (not just items_human.parquet's 711K
items) specifically so vocabulary/IDF statistics are representative of
items the judging stand may show that never appeared in matches.parquet at
all. Capped at 300K/category (not "use all N million") to keep fit() time
and peak RSS during fitting bounded and predictable -- TF-IDF's IDF
estimates stabilize well before that scale for a ~20K-term vocabulary.
"""
import os
import sys
import gc
import json
import pickle
import time

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

SEED = 42
MAX_NAME_CHARS, MAX_ATTR_CHARS = 128, 384   # matches fasttext-category-similarity-probe's own choice
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
FIT_SAMPLE_PER_CATEGORY = 300_000

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "tfidf_category_index")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():8.1f}s] {stage}: {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _safe_filename(index):
    """ASCII-only by construction (a plain zero-padded index, not a
    transliteration of the category name) -- a real run hit silent filename
    corruption ('tfidf_Автотовары.joblib' -> garbage bytes) going through a
    zip/unzip round-trip on Windows, because Cyrillic characters pass
    str.isalnum() unchanged (it's Unicode-aware), so the OLD version of this
    function produced Cyrillic filenames that are fragile across platforms/
    zip implementations. The real category name lives in manifest.json
    (UTF-8 JSON content has none of a filename's encoding fragility) --
    filenames never need to carry it."""
    return f"tfidf_cat{index:02d}.joblib"


def build_text_expr():
    return (
        (pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS) +
         pl.lit(" ") +
         pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS))
        .str.replace_all("\n", " ")
        .str.strip_chars()
        .alias("text")
    )


def fit_category_vectorizers(items_path):
    """Single streaming pass over the full catalog building (category, text),
    then per-category sampling + fit -- see module docstring for why this
    shape (not per-item vectors) is what stays memory-safe downstream."""
    _log("FIT", f"scanning {items_path} for category+text ...")
    t0 = time.perf_counter()
    frame = (
        pl.scan_parquet(items_path)
        .select(["category", "name", "attributes"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8))
        .with_columns(build_text_expr())
        .select(["category", "text"])
        .collect(engine="streaming")
    )
    _log("FIT", f"full catalog scanned: {frame.height:,} items ({time.perf_counter() - t0:.1f}s)")

    categories = sorted(frame.get_column("category").unique().to_list())
    _log("FIT", f"{len(categories)} categories found: {categories}")

    vectorizers = {}
    stats = {}
    for cat in categories:
        t0 = time.perf_counter()
        cat_texts = frame.filter(pl.col("category") == cat).get_column("text")
        n_available = cat_texts.len()
        if n_available > FIT_SAMPLE_PER_CATEGORY:
            cat_texts = cat_texts.sample(n=FIT_SAMPLE_PER_CATEGORY, seed=SEED)
        texts_list = cat_texts.to_list()

        vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        vec.fit(texts_list)
        vectorizers[cat] = vec
        stats[cat] = {"n_available": int(n_available), "n_fit_on": len(texts_list),
                      "vocab_size": len(vec.vocabulary_)}
        _log("FIT", f"{cat!r}: available={n_available:,} fit_on={len(texts_list):,} "
                    f"vocab={len(vec.vocabulary_):,} ({time.perf_counter() - t0:.1f}s)")
        del cat_texts, texts_list
        gc.collect()

    del frame
    gc.collect()
    return vectorizers, stats


def save_vectorizers(vectorizers, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    # Insertion order == the sorted `categories` order from
    # fit_category_vectorizers (Python dicts preserve insertion order),
    # so this enumeration is stable/deterministic run to run.
    for index, (cat, vec) in enumerate(vectorizers.items()):
        fname = _safe_filename(index)
        joblib.dump(vec, os.path.join(out_dir, fname), compress=3)
        manifest[cat] = fname
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"categories": manifest, "max_features": MAX_FEATURES, "ngram_range": list(NGRAM_RANGE),
                   "fit_sample_per_category": FIT_SAMPLE_PER_CATEGORY, "stats": stats}, f,
                  ensure_ascii=False, indent=2)
    total_bytes = sum(os.path.getsize(os.path.join(out_dir, fn)) for fn in manifest.values())
    _log("SAVE", f"{len(manifest)} vectorizers written to {out_dir}, "
                f"total size={total_bytes / 1e6:.1f}MB (manifest.json maps category -> filename)")


def score_pairs_tfidf(vectorizers, pairs_pd, id_to_text, id_to_category):
    """Vectorized within each category group: transform() once per category
    for all its items in this batch of pairs, then dot-product cosine (l2
    norm already applied by TfidfVectorizer) -- same trick as
    fasttext-category-similarity-probe, generalized across categories."""
    scores = np.zeros(len(pairs_pd), dtype=np.float32)
    pairs_pd = pairs_pd.reset_index(drop=True)
    for cat, group in pairs_pd.groupby("category1"):
        vec = vectorizers.get(cat)
        if vec is None:
            continue
        idx = group.index.to_numpy()
        texts1 = [id_to_text.get(i, "") for i in group["id1"]]
        texts2 = [id_to_text.get(i, "") for i in group["id2"]]
        m1 = vec.transform(texts1)
        m2 = vec.transform(texts2)
        cos = np.asarray(m1.multiply(m2).sum(axis=1)).flatten()
        scores[idx] = cos.astype(np.float32)
    return scores


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    _log("CONFIG", f"max_features={MAX_FEATURES} ngram_range={NGRAM_RANGE} "
                    f"fit_sample_per_category={FIT_SAMPLE_PER_CATEGORY} threads={os.cpu_count()}")

    items_path = _find_dataset_file(ITEMS_PATH_HINT, "items.parquet")
    vectorizers, stats = fit_category_vectorizers(items_path)
    save_vectorizers(vectorizers, stats, OUTPUT_DIR)

    # -------------------------------------------------------------
    # Score the SAME val-split pairs berta-top-embed-extract extracted
    # embeddings for -- this kernel's other job, feeding catboost training
    # directly (not just publishing the vectorizers for later/inference use).
    # -------------------------------------------------------------
    _log("SCORE", "scoring val-split pairs for the catboost training feature ...")
    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    matches_pl = pl.read_parquet(matches_path)
    matches_pl = (
        matches_pl.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([pl.col("target").fill_null(0).cast(pl.Float32),
                       (pl.col("target") >= 0.5).cast(pl.Int8).alias("label")])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()
    del matches_pl
    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)
    _train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels_all)
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    _log("SCORE", f"val pairs to score: {len(val_pd):,} (expect 73,131)")
    del matches_pd, labels_all, indices, _train_idx, val_idx
    gc.collect()

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_needed = (
        pl.scan_parquet(items_human_path)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"] + ["name", "attributes"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8))
        .with_columns(build_text_expr())
        .select(["id", "category", "text"])
        .collect(engine="streaming")
    )
    id_to_text = dict(zip(items_needed.get_column("id").to_list(), items_needed.get_column("text").to_list()))
    id_to_category = dict(zip(items_needed.get_column("id").to_list(), items_needed.get_column("category").to_list()))
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_needed, needed_ids, needed_ids_lazy
    gc.collect()

    t0 = time.perf_counter()
    tfidf_scores = score_pairs_tfidf(vectorizers, val_pd, id_to_text, id_to_category)
    _log("SCORE", f"tf-idf scored for {len(val_pd):,} pairs ({time.perf_counter() - t0:.1f}s) -- "
                f"mean={tfidf_scores.mean():.4f} std={tfidf_scores.std():.4f}")

    np.save(os.path.join(OUTPUT_DIR, "val_tfidf_score.npy"), tfidf_scores)
    val_pd[["id1", "id2", "label", "category1"]].to_parquet(
        os.path.join(OUTPUT_DIR, "val_tfidf_meta.parquet"), index=False
    )

    _log("DONE", f"vectorizers + val tf-idf scores written to {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
