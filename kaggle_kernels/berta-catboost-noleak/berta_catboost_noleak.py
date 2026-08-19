"""
CatBoost-on-BERTA-embeddings, NO-LEAK variant: trains and validates entirely
within the canonical val split (73,131 pairs) that BERTA itself never
trained on, split 80/20 here. This is the variant intended as the real
submission candidate -- see ../berta-catboost-leak for the explicit-leakage
comparison (BERTA train split as catboost train, canonical val as catboost
val), which exists only to quantify how much of any observed lift is
inflated by BERTA having already seen those specific pairs.

Input: ../berta-embed-extract's output -- val_embeddings.npy (73,131 x
hidden_dim, BERTA+LoRA mean-pooled pair vectors, extracted BEFORE the
classifier head) + val_meta.parquet (id1, id2, label, category1).

Hyperparameter search: Optuna, GPU CatBoostClassifier, objective = this
competition's own metric (macro-averaged PR-AUC per category, plain mean,
average_precision_score -- CLAUDE.md, never sklearn.metrics.auc) evaluated
on the catboost-val 20% split, not Logloss -- the thing being optimized
should be the thing that's actually reported.
"""
import os
import sys
import json
import subprocess
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna", "catboost"])

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
import optuna

SEED = 42
N_TRIALS = 30
OPTUNA_TIMEOUT_SECONDS = 1800   # 30min safety cap -- GPU catboost on <60K
                                  # rows x few-hundred features per trial
                                  # should be fast; this is a ceiling, not
                                  # an expectation.
EARLY_STOPPING_ROUNDS = 50

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "catboost_noleak")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _find_embed_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "val_embeddings.npy" in files and "val_meta.parquet" in files:
                return root
    raise RuntimeError("berta-embed-extract output not found under /kaggle/input "
                        "(expected val_embeddings.npy + val_meta.parquet, via kernel_sources).")


def compute_macro_pr_auc(y_true, y_prob, categories, category_names):
    per_cat = {}
    for cat_name in category_names:
        mask = categories == cat_name
        if mask.sum() < 2:
            continue
        yt = y_true[mask]
        if len(np.unique(yt)) < 2:
            continue
        per_cat[cat_name] = average_precision_score(yt, y_prob[mask])
    macro = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return macro, per_cat


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 90)
    print("CatBoost on BERTA+LoRA embeddings -- NO-LEAK (val 80/20)")
    print("=" * 90)

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    embed_dir = _find_embed_dir()
    print(f"[DATA] embeddings dir: {embed_dir}")
    with open(os.path.join(embed_dir, "meta.json")) as f:
        source_meta = json.load(f)
    print(f"[DATA] source meta: {source_meta}")

    val_emb = np.load(os.path.join(embed_dir, "val_embeddings.npy"))
    val_meta = pd.read_parquet(os.path.join(embed_dir, "val_meta.parquet"))
    print(f"[DATA] val embeddings: {val_emb.shape}, val meta: {val_meta.shape}")
    assert len(val_emb) == len(val_meta)

    y = val_meta["label"].to_numpy().astype(np.int32)
    cats = val_meta["category1"].to_numpy()
    category_names = sorted(val_meta["category1"].unique().tolist())
    print(f"[DATA] category vocab: {len(category_names)}")

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_test = val_emb[train_idx], val_emb[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    cats_train, cats_test = cats[train_idx], cats[test_idx]
    print(f"[DATA] catboost train={len(X_train):,} catboost test={len(X_test):,} "
          f"(both from BERTA's own held-out val split -- no leakage)")

    gpu_available = True
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    task_type = "GPU" if gpu_available else "CPU"
    print(f"[CATBOOST] task_type={task_type}")

    trial_log = []

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "task_type": task_type,
            "loss_function": "Logloss",
            "random_seed": SEED,
            "verbose": False,
        }
        if task_type == "GPU":
            params["devices"] = "0"
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        probs = model.predict_proba(X_test)[:, 1]
        macro, _ = compute_macro_pr_auc(y_test, probs, cats_test, category_names)
        trial_log.append({"trial": trial.number, "params": params, "macro_pr_auc": macro,
                           "best_iteration": model.get_best_iteration()})
        print(f"[OPTUNA] trial={trial.number} macro_pr_auc={macro:.4f} params={params}", flush=True)
        return macro

    print(f"[OPTUNA] searching {N_TRIALS} trials (timeout {OPTUNA_TIMEOUT_SECONDS}s) ...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT_SECONDS)

    print(f"[OPTUNA] best trial={study.best_trial.number} macro_pr_auc={study.best_value:.4f}")
    print(f"[OPTUNA] best params: {study.best_params}")

    best_params = dict(study.best_params)
    best_params.update({"task_type": task_type, "loss_function": "Logloss", "random_seed": SEED, "verbose": False})
    if task_type == "GPU":
        best_params["devices"] = "0"

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test),
                     early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)

    train_probs = final_model.predict_proba(X_train)[:, 1]
    test_probs = final_model.predict_proba(X_test)[:, 1]
    train_macro, train_per_cat = compute_macro_pr_auc(y_train, train_probs, cats_train, category_names)
    test_macro, test_per_cat = compute_macro_pr_auc(y_test, test_probs, cats_test, category_names)

    print(f"[FINAL] train_macro_pr_auc={train_macro:.4f} test_macro_pr_auc={test_macro:.4f}")
    print(f"[FINAL] per_category test: {json.dumps(test_per_cat, ensure_ascii=False, indent=2)}")

    final_model.save_model(os.path.join(OUTPUT_DIR, "catboost_noleak.cbm"))

    with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "variant": "no_leak_val_80_20", "source_checkpoint": source_meta.get("source_checkpoint"),
            "hidden_dim": source_meta.get("hidden_dim"), "n_trials_run": len(trial_log),
            "best_trial": study.best_trial.number, "best_params": best_params,
            "train_macro_pr_auc": train_macro, "test_macro_pr_auc": test_macro,
            "train_per_category": train_per_cat, "test_per_category": test_per_cat,
            "trial_log": trial_log, "n_catboost_train": int(len(X_train)),
            "n_catboost_test": int(len(X_test)), "total_wall_clock_s": _elapsed(),
        }, f, ensure_ascii=False, indent=2)

    print(f"[DONE] catboost no-leak variant complete. test_macro_pr_auc={test_macro:.4f} "
          f"total wall-clock: {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
