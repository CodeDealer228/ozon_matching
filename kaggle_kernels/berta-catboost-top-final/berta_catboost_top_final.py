"""
Trains the CatBoost head meant to actually ship, on top of the TOP BERTA+
LoRA checkpoint (maxamsecret/berta-tpu-lora-top-checkpoint -- test 0.47,
the best real test-stand score of any submission in this project so far).

Data: the canonical VAL split (73,131 pairs) that checkpoint never trained
on -- extracted by kaggle_kernels/berta-top-embed-extract (pooled
embedding + classifier score) and kaggle_kernels/tfidf-category-index-fit
(per-category tf-idf pairwise score), joined here by (id1, id2) rather than
assumed to share row order across two independently-run kernels. THIS is
what keeps catboost's own training disjoint from anything BERTA itself saw
-- same no-leak methodology as kaggle_kernels/berta-catboost-noleak, on the
better base checkpoint. Split 80/20 (stratified by label, seed=42) for
catboost's own train/val, per direct instruction.

Features (concatenated, per direct instruction): BERTA pooled embedding
(768) + classifier score (1, sigmoid-applied for a bounded [0,1] feature)
+ tf-idf pairwise score (1, already cosine-similarity-shaped, [0,1]-ish)
+ category (1, native CatBoost CATEGORICAL feature -- not a numeric
"category_id" treated as ordinal, which would impose a false ordering
between unrelated categories; CatBoost's own categorical handling is the
technically correct fit for what was asked).

Trained on 2 GPUs (task_type="GPU", devices="0:1"), TOTAL_ITERATIONS=5000
with real, deliberately raised regularization (l2_leaf_reg=15 vs CatBoost's
own default 3, moderate depth=8, bagging_temperature + random_strength for
extra stochastic regularization) -- "большое количество итераций" with the
regularization sized to match, not just cranking iterations alone.

===========================================================================
CHECKPOINTING -- 6 saved models, not 1
===========================================================================
Per direct instruction: save several checkpoints, not only the val-best
one. FIRST ATTEMPT trained in 5 sequential stages continued via CatBoost's
`init_model` -- that crashed for real: "catboost/cuda/train_lib/
train.cpp:323: Training continuation for GPU is not yet supported". GPU
task_type simply does not support resumed/continued training the way CPU
does (confirmed by the crash, not assumed). FIXED design: train ONCE, in a
single `.fit()` call, to the full TOTAL_ITERATIONS=5000 -- then produce the
5 checkpoints POST-HOC via `model.copy()` + `.shrink(ntree_end=N)`, which
truncates an already-fully-trained model to its first N trees. Five
independent truncated copies (at 1000/2000/3000/4000/5000 trees) stand in
for what would have been five sequential training stages -- same
observable outcome (5 real checkpoints, each with its own real val score
evaluated at that exact tree count), zero dependency on GPU-continuation
support. Each is evaluated with this project's actual scoring metric (real
macro PR-AUC, grouped by category -- not just CatBoost's own Logloss/AUC).
Whichever of those 5 has the best val macro PR-AUC is ALSO saved separately
as "best" (a 6th file, a copy -- not a distinct training run). Every
checkpoint's own train_state.json records exactly which stage/tree-count it
is and its own real val_macro_pr_auc -- explicit, never a stale or
inherited number (the same checkpoint-hygiene lesson this session's
berta_tpu_lora provenance trace surfaced, see
reports/berta_tpu_lora_provenance/ and
kaggle_kernels/berta-tpu-human-continue-384's own module docstring).
"""
import os
import sys
import gc
import json
import time

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

SEED = 42
TOTAL_ITERATIONS = 5000
STAGE_ITERATIONS = 1000   # 5 stages -> checkpoints at 20/40/60/80/100%
N_STAGES = TOTAL_ITERATIONS // STAGE_ITERATIONS

CATBOOST_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="GPU",
    devices="0:1",
    depth=8,
    l2_leaf_reg=15.0,          # raised well above CatBoost's own default (3) -- "хорошая регуляризация"
    bagging_temperature=1.0,   # Bayesian bagging, extra stochastic regularization
    random_strength=2.0,       # extra randomness in split-score selection
    random_seed=SEED,
    verbose=100,
)

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "catboost_top_final")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():8.1f}s] {stage}: {msg}", flush=True)


def _find_dir_with_file(filename, near=None):
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files and (near is None or near in root):
                return root
    return None


def compute_macro_pr_auc(y_true, y_prob, categories):
    """Same metric this whole project scores on -- plain mean of
    average_precision_score over categories with both classes present, NOT
    sklearn's interpolated `auc`. See CLAUDE.md -- Метрика."""
    per_cat = []
    for cat in np.unique(categories):
        mask = categories == cat
        y_t = y_true[mask]
        if len(np.unique(y_t)) < 2:
            continue
        per_cat.append(average_precision_score(y_t, y_prob[mask]))
    return float(np.mean(per_cat)) if per_cat else float("nan")


def load_and_join_features():
    embed_dir = _find_dir_with_file("val_embeddings.npy")
    if embed_dir is None:
        raise RuntimeError("berta-top-embed-extract output not found under /kaggle/input.")
    tfidf_dir = _find_dir_with_file("val_tfidf_score.npy")
    if tfidf_dir is None:
        raise RuntimeError("tfidf-category-index-fit output not found under /kaggle/input.")

    embeddings = np.load(os.path.join(embed_dir, "val_embeddings.npy"))
    classifier_score = np.load(os.path.join(embed_dir, "val_classifier_score.npy"))
    embed_meta = pd.read_parquet(os.path.join(embed_dir, "val_meta.parquet"))
    _log("LOAD", f"embeddings={embeddings.shape} classifier_score={classifier_score.shape} "
                f"meta={len(embed_meta):,} rows (from {embed_dir})")

    tfidf_score = np.load(os.path.join(tfidf_dir, "val_tfidf_score.npy"))
    tfidf_meta = pd.read_parquet(os.path.join(tfidf_dir, "val_tfidf_meta.parquet"))
    _log("LOAD", f"tfidf_score={tfidf_score.shape} meta={len(tfidf_meta):,} rows (from {tfidf_dir})")

    # Join by (id1, id2) -- do NOT assume the two kernels produced identical
    # row order (see module docstring). Build a lookup from the tfidf side,
    # apply it in embed_meta's own row order (which defines the embeddings'
    # row order too).
    tfidf_lookup = dict(zip(zip(tfidf_meta["id1"], tfidf_meta["id2"]), tfidf_score))
    aligned_tfidf = np.array(
        [tfidf_lookup.get((a, b), np.nan) for a, b in zip(embed_meta["id1"], embed_meta["id2"])],
        dtype=np.float32,
    )
    n_missing = int(np.isnan(aligned_tfidf).sum())
    if n_missing:
        _log("WARN", f"{n_missing} pairs missing a tf-idf score after join -- filling with 0.0")
        aligned_tfidf = np.nan_to_num(aligned_tfidf, nan=0.0)

    assert len(embed_meta) == len(embeddings) == len(classifier_score) == len(aligned_tfidf)
    _log("JOIN", f"{len(embed_meta):,} pairs aligned by (id1,id2) across both feature sources")

    return embeddings, classifier_score, aligned_tfidf, embed_meta


def build_feature_frame(embeddings, classifier_score, tfidf_score, meta):
    emb_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    frame = pd.DataFrame(emb_cols)
    frame["classifier_score"] = 1.0 / (1.0 + np.exp(-classifier_score))   # sigmoid -> bounded [0,1] feature
    frame["tfidf_score"] = tfidf_score
    frame["category"] = meta["category1"].astype(str).values
    return frame


def save_checkpoint(model, stage, iterations_done, val_macro_pr_auc, label, out_dir):
    save_dir = os.path.join(out_dir, label)
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(os.path.join(save_dir, "catboost_model.cbm"))
    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({
            "stage": stage, "iterations_done": iterations_done,
            "val_macro_pr_auc": val_macro_pr_auc,
        }, f, indent=2)
    print(f"[SAVE] {label}: stage={stage} iterations={iterations_done} "
          f"val_macro_pr_auc={val_macro_pr_auc}", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _log("CONFIG", f"total_iterations={TOTAL_ITERATIONS} stage_iterations={STAGE_ITERATIONS} "
                    f"n_stages={N_STAGES} params={CATBOOST_PARAMS}")

    embeddings, classifier_score, tfidf_score, meta = load_and_join_features()
    frame = build_feature_frame(embeddings, classifier_score, tfidf_score, meta)
    labels = meta["label"].to_numpy().astype(int)
    categories = meta["category1"].to_numpy()
    del embeddings
    gc.collect()

    indices = np.arange(len(frame))
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels)
    _log("SPLIT", f"catboost train={len(train_idx):,} val={len(val_idx):,} (80/20, stratified, seed={SEED})")

    X_train, y_train = frame.iloc[train_idx], labels[train_idx]
    X_val, y_val = frame.iloc[val_idx], labels[val_idx]
    cat_val = categories[val_idx]
    cat_features = ["category"]

    # Train ONCE, to the FULL iteration count, in a single .fit() call.
    # The original design trained in 5 sequential stages continued via
    # `init_model` -- CatBoost does not support that on GPU
    # ("catboost/cuda/train_lib/train.cpp:323: Training continuation for
    # GPU is not yet supported", a real crash hit mid-run). The 20%-interval
    # checkpoints are produced POST-HOC instead: CatBoost's own
    # `model.shrink(ntree_end=N)` truncates an already-fully-trained model
    # to its first N trees in place, so five independent truncated copies
    # of the ONE trained model stand in for what would have been five
    # sequential training stages -- same observable result (5 real
    # checkpoints, each with its own real val score at that tree count),
    # no GPU-continuation dependency at all.
    t0 = time.perf_counter()
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = TOTAL_ITERATIONS
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_features, verbose=100)
    _log("TRAIN", f"full {TOTAL_ITERATIONS}-iteration training done in {time.perf_counter() - t0:.1f}s")

    best_stage, best_val_macro_pr_auc, best_label = None, -1.0, None

    for stage in range(1, N_STAGES + 1):
        iterations_done = stage * STAGE_ITERATIONS
        stage_model = model.copy()
        stage_model.shrink(ntree_end=iterations_done)

        val_prob = stage_model.predict_proba(X_val)[:, 1]
        val_macro_pr_auc = compute_macro_pr_auc(y_val, val_prob, cat_val)
        _log("METRICS", f"stage {stage} ({iterations_done} trees): val_macro_pr_auc={val_macro_pr_auc:.4f}")

        label = f"checkpoint_stage{stage}_{iterations_done}it"
        save_checkpoint(stage_model, stage, iterations_done, val_macro_pr_auc, label, OUTPUT_DIR)

        if val_macro_pr_auc > best_val_macro_pr_auc:
            best_val_macro_pr_auc = val_macro_pr_auc
            best_stage = stage
            best_label = label
        del stage_model
        gc.collect()

    # "best" is a labeled COPY of whichever of the 5 real checkpoints had
    # the top val_macro_pr_auc -- not a 6th independent training run (see
    # module docstring).
    import shutil
    best_src = os.path.join(OUTPUT_DIR, best_label)
    best_dst = os.path.join(OUTPUT_DIR, "best")
    if os.path.isdir(best_dst):
        shutil.rmtree(best_dst)
    shutil.copytree(best_src, best_dst)
    _log("BEST", f"stage {best_stage} ({best_label}) had the top val_macro_pr_auc={best_val_macro_pr_auc:.4f} "
                f"-- copied to {best_dst}")

    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump({
            "total_iterations": TOTAL_ITERATIONS, "stage_iterations": STAGE_ITERATIONS,
            "n_stages": N_STAGES, "catboost_params": CATBOOST_PARAMS,
            "best_stage": best_stage, "best_val_macro_pr_auc": best_val_macro_pr_auc,
            "feature_columns": list(frame.columns), "n_features": frame.shape[1],
            "source_berta_checkpoint": "maxamsecret/berta-tpu-lora-top-checkpoint (test=0.47)",
        }, f, indent=2)

    _log("DONE", f"5 stage checkpoints + best -> {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
