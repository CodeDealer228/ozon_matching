"""
CatBoost-on-BERTA-embeddings, LEAK-ACCEPTED variant: trains on the FULL
human train split (292,523 pairs, the exact pairs BERTA itself trained on)
and validates on the canonical val split (73,131 pairs). This is an
explicit, labeled leakage comparison against ../berta-catboost-noleak (which
trains and validates entirely within BERTA's own held-out val, no leakage)
-- NOT the real submission candidate. Its purpose is to quantify how much
of any observed lift is inflated by BERTA having already memorized these
specific train pairs' embeddings to some degree, rather than being a
legitimately better representation.

Input: ../berta-embed-extract's output -- train_embeddings.npy (292,523 x
hidden_dim) + train_meta.parquet for catboost-train, val_embeddings.npy
(73,131 x hidden_dim) + val_meta.parquet for catboost-val.

Hyperparameter search: identical setup to ../berta-catboost-noleak (Optuna,
GPU CatBoostClassifier, objective = macro-averaged PR-AUC per category) --
see that script's module docstring for the full rationale, not repeated
here.
"""
import os
import sys
import json
import subprocess
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna", "catboost"])

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
import optuna

SEED = 42
N_TRIALS = 30
OPTUNA_TIMEOUT_SECONDS = 2400   # 40min safety cap -- train split is ~4x
                                  # larger than the no-leak variant's, so
                                  # each trial costs more; still a ceiling,
                                  # not an expectation.
EARLY_STOPPING_ROUNDS = 50

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "catboost_leak")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _find_embed_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "train_embeddings.npy" in files and "val_embeddings.npy" in files:
                return root
    raise RuntimeError("berta-embed-extract output not found under /kaggle/input "
                        "(expected train_embeddings.npy + val_embeddings.npy, via kernel_sources).")


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
    print("CatBoost on BERTA+LoRA embeddings -- LEAK-ACCEPTED (train->val)")
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

    X_train = np.load(os.path.join(embed_dir, "train_embeddings.npy"))
    train_meta = pd.read_parquet(os.path.join(embed_dir, "train_meta.parquet"))
    X_val = np.load(os.path.join(embed_dir, "val_embeddings.npy"))
    val_meta = pd.read_parquet(os.path.join(embed_dir, "val_meta.parquet"))
    print(f"[DATA] train embeddings: {X_train.shape}, val embeddings: {X_val.shape}")
    assert len(X_train) == len(train_meta) and len(X_val) == len(val_meta)

    y_train = train_meta["label"].to_numpy().astype(np.int32)
    y_val = val_meta["label"].to_numpy().astype(np.int32)
    cats_train = train_meta["category1"].to_numpy()
    cats_val = val_meta["category1"].to_numpy()
    category_names = sorted(set(train_meta["category1"].unique().tolist()) | set(val_meta["category1"].unique().tolist()))
    print(f"[DATA] category vocab: {len(category_names)} | catboost train={len(X_train):,} "
          f"catboost val={len(X_val):,} (train = the SAME pairs BERTA trained on -- explicit leak)")

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
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        macro, _ = compute_macro_pr_auc(y_val, probs, cats_val, category_names)
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
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val),
                     early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)

    train_probs = final_model.predict_proba(X_train)[:, 1]
    val_probs = final_model.predict_proba(X_val)[:, 1]
    train_macro, train_per_cat = compute_macro_pr_auc(y_train, train_probs, cats_train, category_names)
    val_macro, val_per_cat = compute_macro_pr_auc(y_val, val_probs, cats_val, category_names)

    print(f"[FINAL] train_macro_pr_auc={train_macro:.4f} val_macro_pr_auc={val_macro:.4f}")
    print(f"[FINAL] per_category val: {json.dumps(val_per_cat, ensure_ascii=False, indent=2)}")

    final_model.save_model(os.path.join(OUTPUT_DIR, "catboost_leak.cbm"))

    with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "variant": "leak_accepted_train_to_val", "source_checkpoint": source_meta.get("source_checkpoint"),
            "hidden_dim": source_meta.get("hidden_dim"), "n_trials_run": len(trial_log),
            "best_trial": study.best_trial.number, "best_params": best_params,
            "train_macro_pr_auc": train_macro, "val_macro_pr_auc": val_macro,
            "train_per_category": train_per_cat, "val_per_category": val_per_cat,
            "trial_log": trial_log, "n_catboost_train": int(len(X_train)),
            "n_catboost_val": int(len(X_val)), "total_wall_clock_s": _elapsed(),
        }, f, ensure_ascii=False, indent=2)

    print(f"[DONE] catboost leak-accepted variant complete. val_macro_pr_auc={val_macro:.4f} "
          f"total wall-clock: {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
