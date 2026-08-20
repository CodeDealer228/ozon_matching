import os

# Hardware-aware env tuning for the test stand (1x H100 80GB / 20 CPU cores /
# 200GB RAM at inference time, per CLAUDE.md).
os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
_N_THREADS = str(os.cpu_count() or 20)
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ.setdefault("RAYON_NUM_THREADS", _N_THREADS)

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Must byte-match the formula this repo trains/evaluates against (see
# CLAUDE.md "Text preprocessing"): raw attribute string slice, no json.loads.
MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64

# Giga-Embeddings-instruct's own README: symmetric tasks (classification,
# similarity) need an instruction prefix on every input or "you will see a
# quality drop" (their FAQ, verbatim). Worded after their own example
# ("Классифицируй отзыв на товар как положительный/отрицательный/нейтральный").
INSTRUCTION = "Instruct: Определи, являются ли два товара дубликатами\n"


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_needed_item_texts(items_path, needed_ids):
    df = pd.read_parquet(
        items_path,
        engine="pyarrow",
        columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", list(set(needed_ids)))],
    )

    name = df["name"].fillna("").astype(str).str.slice(0, MAX_NAME_CHARS)
    category = df["category"].fillna("").astype(str).str.slice(0, MAX_CATEGORY_CHARS)
    attributes = df["attributes"].fillna("").astype(str).str.slice(0, MAX_ATTR_CHARS)

    text = (
        "Name: " + name + " Category: " + category + " Attributes: " + attributes
    ).str.slice(0, MAX_ITEM_CHARS)

    return dict(zip(df["id"].tolist(), text.tolist()))


def load_needed_item_categories(items_path, needed_ids):
    """category of id1 as a per-pair grouping key for macro_pr_auc_by_category
    -- val-only, not used by predict_pipeline (judging matches_path has no
    category info to leak from, this reads it straight from items_path)."""
    df = pd.read_parquet(
        items_path, engine="pyarrow", columns=["id", "category"],
        filters=[("id", "in", list(set(needed_ids)))],
    )
    return dict(zip(df["id"].tolist(), df["category"].fillna("UNKNOWN").astype(str).tolist()))


def macro_pr_auc_by_category(y_true, y_prob, categories):
    """Competition metric (CLAUDE.md): per-category average_precision_score,
    skip categories with only one class present, plain mean over the rest."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    categories = np.asarray(categories)

    overall_ap = float(average_precision_score(y_true, y_prob))

    per_category = []
    for cat in np.unique(categories):
        mask = categories == cat
        if len(np.unique(y_true[mask])) < 2:
            continue
        per_category.append(average_precision_score(y_true[mask], y_prob[mask]))

    macro = float(np.mean(per_category)) if per_category else float("nan")
    return macro, overall_ap, len(per_category)


def build_pair_texts(match_df, id_to_text):
    """One combined string per pair, not text/text_pair -- the Llama-style
    BPE tokenizer here has no working pair-join convention (verified
    locally: `tokenizer(text=, text_pair=)` glues the two strings together
    with zero separator, e.g. "hello worldsome other text"). Explicit
    "Товар 1:"/"Товар 2:" labels double as both a human-readable separator
    and instruction-consistent framing."""
    text1 = match_df["id1"].map(id_to_text).fillna("")
    text2 = match_df["id2"].map(id_to_text).fillna("")

    missing = int((text1 == "").sum() + (text2 == "").sum())
    if missing:
        print(f"[WARN] {missing} item-id references had no matching item text (empty text substituted)", flush=True)

    combined = (
        INSTRUCTION + "Товар 1: " + text1 + "\nТовар 2: " + text2
    )
    return combined.tolist()


def extract_pair_embeddings(encoder, texts, batch_size, show_progress=True):
    """Batched encode with adaptive-size CUDA OOM fallback: a single
    over-large batch halves and retries instead of hard-failing the run.

    If the encoder exposes tokenize_batch()/forward_batch() (GigaCrossEncoder
    does; a test double like DummyEncoder need not), overlaps CPU tokenization
    of the NEXT batch with the GPU forward pass of the CURRENT one on a
    background thread -- removes tokenizer wall-clock from the critical path
    instead of paying for it serially before every single forward call."""
    n = len(texts)
    bs = batch_size
    result = None
    i = 0
    t_start = t_last_log = time.time()

    has_split = hasattr(encoder, "tokenize_batch") and hasattr(encoder, "forward_batch")
    executor = None
    pending = None
    if has_split and i < n:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        pending = executor.submit(encoder.tokenize_batch, texts[i:i + bs])

    try:
        while i < n:
            cur_end = min(i + bs, n)
            batch = texts[i:cur_end]
            try:
                if executor is not None:
                    encoded = pending.result()
                    # Kick off tokenizing the NEXT window (at the CURRENT bs)
                    # before this batch's forward pass runs, so CPU work
                    # overlaps with GPU compute instead of happening serially.
                    nxt = texts[cur_end:cur_end + bs]
                    pending = executor.submit(encoder.tokenize_batch, nxt) if nxt else None
                    emb = encoder.forward_batch(encoded)
                else:
                    emb = encoder.encode_pairs(batch, batch_size=bs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or bs <= 4:
                    raise
                bs = max(4, bs // 2)
                print(f"[ENCODE] CUDA OOM -> retrying with batch_size={bs}"
                      + (", disabling prefetch" if executor is not None else ""), flush=True)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if executor is not None:
                    # Don't try to keep the prefetch pipeline consistent
                    # through a batch-size change mid-flight -- fall back to
                    # the simple serial path (encode_pairs re-tokenizes+
                    # forwards each retry at the new, smaller bs) rather than
                    # risk an index-bookkeeping bug for a rare recovery path.
                    executor.shutdown(wait=False)
                    executor = None
                    pending = None
                continue

            if result is None:
                result = np.empty((n, emb.shape[1]), dtype=np.float32)
            result[i:i + len(batch)] = emb
            i += len(batch)
            if show_progress and time.time() - t_last_log > 15:
                rate = i / max(time.time() - t_start, 1e-9)
                print(f"[ENCODE] {i:,}/{n:,} ({rate:.1f} pairs/sec, batch_size={bs})", flush=True)
                t_last_log = time.time()
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return result


def train_pipeline(items_path, matches_path, encoder, output_classifier_path,
                    batch_size=512, seed=42, val_fraction=0.2):
    """Fit StandardScaler+LogisticRegression on frozen pooled pair embeddings.
    Not used at judging time (matches_path there has no `target` column) --
    this is the local/offline step that produces the classifier joblib
    shipped in the submission zip.

    val_fraction: held out BEFORE fitting (stratified by label) and scored
    with the competition's own macro_pr_auc_by_category, so we know whether
    the classifier learned anything, not just that it ran. Set to 0/None to
    fit on 100% of the sample (old behavior) with no validation reported."""
    t0 = time.time()
    print("=== Starting Training Pipeline ===", flush=True)

    match_df = pd.read_parquet(matches_path)
    if "target" not in match_df.columns:
        raise ValueError("train_pipeline needs a matches file with a `target` column")
    labels_all = (match_df["target"].astype(float) >= 0.5).astype(int).to_numpy()
    print(f"[1/5] pairs={len(match_df):,} ({time.time() - t0:.1f}s)", flush=True)

    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    id_to_text = load_needed_item_texts(items_path, needed_ids)
    print(f"[2/5] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    texts_all = build_pair_texts(match_df, id_to_text)
    print(f"[3/5] pair texts built ({time.time() - t0:.1f}s)", flush=True)

    if val_fraction:
        idx = np.arange(len(match_df))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_fraction, random_state=seed, stratify=labels_all
        )
    else:
        train_idx, val_idx = np.arange(len(match_df)), np.array([], dtype=int)

    train_texts = [texts_all[i] for i in train_idx]
    train_labels = labels_all[train_idx]
    print(f"[4/5] train={len(train_idx):,} val={len(val_idx):,} -- extracting train embeddings ...", flush=True)
    train_embeddings = extract_pair_embeddings(encoder, train_texts, batch_size=batch_size)
    print(f"[4/5] train embeddings extracted ({time.time() - t0:.1f}s)", flush=True)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=1234)),
    ])
    clf.fit(train_embeddings, train_labels)

    if len(val_idx):
        val_texts = [texts_all[i] for i in val_idx]
        val_labels = labels_all[val_idx]
        id_to_cat = load_needed_item_categories(items_path, needed_ids)
        val_categories = [id_to_cat.get(match_df["id1"].iloc[i], "UNKNOWN") for i in val_idx]
        val_embeddings = extract_pair_embeddings(encoder, val_texts, batch_size=batch_size)
        val_probs = clf.predict_proba(val_embeddings)[:, 1]
        macro, overall, n_cat = macro_pr_auc_by_category(val_labels, val_probs, val_categories)
        print(f"[VAL] macro_pr_auc={macro:.5f} overall_ap={overall:.5f} "
              f"n_categories={n_cat} val_pairs={len(val_idx):,} ({time.time() - t0:.1f}s)", flush=True)

    joblib.dump(clf, output_classifier_path)
    print(f"[5/5] saved -> {output_classifier_path} ({time.time() - t0:.1f}s)", flush=True)

    print("=== Training Pipeline Complete ===", flush=True)
    return clf


def predict_pipeline(items_path, matches_path, encoder, classifier_path, output_csv_path, batch_size=512):
    t0 = time.time()
    print("=== Starting prediction pipeline ===", flush=True)

    print("[1/4] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/4] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/4] Loading needed item texts ...", flush=True)
    id_to_text = load_needed_item_texts(items_path, needed_ids)
    print(f"[2/4] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/4] Building pair texts ...", flush=True)
    texts = build_pair_texts(match_df, id_to_text)
    print(f"[3/4] done ({time.time() - t0:.1f}s)", flush=True)

    print("[4/4] Extracting embeddings + scoring ...", flush=True)
    embeddings = extract_pair_embeddings(encoder, texts, batch_size=batch_size)
    clf = joblib.load(classifier_path)
    predict = clf.predict_proba(embeddings)[:, 1]
    print(f"[4/4] done ({time.time() - t0:.1f}s)", flush=True)

    results_df = pd.DataFrame({
        "id1": match_df["id1"].to_numpy(),
        "id2": match_df["id2"].to_numpy(),
        "predict": predict,
    })

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - t0
    print(f"=== Done in {elapsed:.1f}s ({len(texts) / max(elapsed, 1e-9):.1f} pairs/sec) ===", flush=True)
    return results_df
