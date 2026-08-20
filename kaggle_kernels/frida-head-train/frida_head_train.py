"""
Trains the multi-layer classification head that replaces FRIDA's single
nn.Linear, on the cached pair embeddings from
kaggle_kernels/frida-tpu-embed-cache-human.

WHY A SEPARATE KERNEL AT ALL
===========================================================================
The encoder forward pass is ~all of the cost of a training step; the head is
a rounding error on top of it. Once representations are cached, head
architecture becomes searchable in minutes on one T4 instead of hours of TPU
per variant. The head that wins here is then warm-started into
kaggle_kernels/frida-llm-tpu-sft so that run does not spend its first hour
teaching a randomly-initialised head what the old linear one already knew --
which is the "stabilises the next training stage" argument for doing this
first.

Two outputs, both deliberate:
  * head.pt          -- state_dict of MultiLayerHead, consumed by
                        frida-llm-tpu-sft (the class definition there must
                        stay byte-identical to the one below, or the
                        state_dict will not load).
  * val_features.npy -- the penultimate activation (HEAD_HIDDEN[-1] wide) on
                        the canonical val split, so a CatBoost/GBDT head can
                        be fitted on a compact learned feature space instead
                        of 1536 raw dims. That distinction matters: raw
                        embeddings are exactly what made berta_catboost_top_final
                        overfit (96.11% of importance spread flat across 768
                        near-noise dims, val 0.7467 -> test 0.457).

LEAK DISCIPLINE
===========================================================================
The encoder that produced these embeddings was fine-tuned on the canonical
seed-42 TRAIN split, so split==0 rows carry representations the encoder has
already fit. This kernel trains the head on split==0 anyway, and that is
correct here rather than sloppy: the head is a component of a model that
frida-llm-tpu-sft will itself train on split==0: no new contamination is
introduced. split==1 (73,131 pairs) is touched only for evaluation, so the
reported macro PR-AUC is directly comparable to every other val number in
this project and stays an honest holdout for the whole pipeline.

What that split==0 contamination DOES cause is unusually separable training
features, i.e. a head that will overfit if left alone. Hence dropout,
weight decay, and -- the part that actually matters -- model selection by the
clean val score with early stopping, never by train loss.

The alternative (train the head strictly inside split==1, the
berta-catboost-noleak methodology) is available via TRAIN_ON=="val_inner".
It is the right mode if the head is ever destined for a frozen-encoder
submission rather than a warm start, and it is the mode that answers "how
much of the split==0 score is contamination?" -- but it trains on 4x less
data, which is the exact shortage documented in README.md.
"""
import os
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-pair-embed-cache-human"

# "train_split": fit on split==0, evaluate on split==1 (default -- see the
#                leak-discipline note in the module docstring).
# "val_inner":   fit and evaluate strictly inside split==1, the
#                berta-catboost-noleak methodology.
TRAIN_ON = "train_split"

HEAD_HIDDEN = (512, 128)
HEAD_DROPOUT = 0.15

EPOCHS = 40
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 8

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_head_train")

DATASET_ID = "maxamsecret/frida-multilayer-head"
DATASET_TITLE = "FRIDA multi-layer classification head"
ENABLE_DATASET_PUSH = True

t0 = time.perf_counter()


def _log(stage, msg):
    print(f"[{time.perf_counter() - t0:8.1f}s] {stage}: {msg}", flush=True)


# =====================================================================
# HEAD -- this class definition is duplicated verbatim in
# kaggle_kernels/frida-llm-tpu-sft. Kernels are standalone scripts with no
# shared import, so the two copies must be kept in sync by hand; a mismatch
# surfaces as a state_dict load error there, not as silent wrong behaviour.
# =====================================================================

class MultiLayerHead(nn.Module):
    def __init__(self, hidden_dim, layer_sizes=HEAD_HIDDEN, dropout=HEAD_DROPOUT):
        super().__init__()
        blocks = []
        d = hidden_dim
        for h in layer_sizes:
            blocks.append(nn.Sequential(
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(d, 1)
        self.feature_dim = d

    def forward(self, x, return_features=False):
        for block in self.blocks:
            x = block(x)
        logit = self.out(x).view(-1)
        if return_features:
            return logit, x
        return logit


# =====================================================================
# DATA
# =====================================================================

def _find_cache_root(hint):
    if os.path.isdir(hint) and os.path.exists(os.path.join(hint, "cache_meta.json")):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "pair_embedding.bin" in files:
                return root
    raise RuntimeError(f"no pair-embedding cache found (hint was {hint})")


def load_cache():
    root = _find_cache_root(CACHE_HINT)
    with open(os.path.join(root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_rows, hidden_dim = meta["n_rows"], meta["hidden_dim"]
    _log("CACHE", f"{root}: n_rows={n_rows:,} hidden_dim={hidden_dim} "
                  f"complete={meta.get('complete')} val_fingerprint={meta.get('val_fingerprint')}")

    if not meta.get("complete", False):
        raise RuntimeError("cache is marked incomplete -- training a head on partially "
                           "written embeddings would silently learn from zero rows. "
                           "Check written.bin and re-run the cache kernel.")

    emb = np.memmap(os.path.join(root, "pair_embedding.bin"), dtype=np.float32, mode="r",
                    shape=(n_rows, hidden_dim))
    logit = np.memmap(os.path.join(root, "classifier_logit.bin"), dtype=np.float32, mode="r",
                      shape=(n_rows,))
    pairs = pd.read_parquet(os.path.join(root, "pairs.parquet"))
    assert len(pairs) == n_rows, f"pairs.parquet has {len(pairs)} rows, cache has {n_rows}"
    return np.asarray(emb), np.asarray(logit), pairs, meta


def compute_metrics(scores, labels, categories, category_names):
    per_cat = {}
    for cat_id, cat_name in enumerate(category_names):
        mask = categories == cat_id
        if mask.sum() < 2:
            continue
        y_true = labels[mask]
        if len(np.unique(y_true)) < 2:
            continue
        per_cat[cat_name] = float(average_precision_score(y_true, scores[mask]))
    macro = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return macro, per_cat


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    emb, base_logit, pairs, meta = load_cache()
    hidden_dim = meta["hidden_dim"]
    category_vocab = meta["category_vocab"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}

    labels = pairs["label"].to_numpy().astype(np.float32)
    split = pairs["split"].to_numpy().astype(np.int8)
    categories = pairs["category1"].map(category_to_id).fillna(len(category_vocab) - 1).to_numpy().astype(np.int64)

    if TRAIN_ON == "train_split":
        train_rows = np.where(split == 0)[0]
        val_rows = np.where(split == 1)[0]
    elif TRAIN_ON == "val_inner":
        val_pool = np.where(split == 1)[0]
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(val_pool))
        cut = int(0.8 * len(val_pool))
        train_rows, val_rows = val_pool[perm[:cut]], val_pool[perm[cut:]]
    else:
        raise ValueError(f"unknown TRAIN_ON={TRAIN_ON!r}")
    _log("SPLIT", f"mode={TRAIN_ON} train={len(train_rows):,} val={len(val_rows):,}")

    # The checkpoint's own linear head, scored on exactly this val slice: the
    # number the multi-layer head has to beat to be worth shipping.
    baseline_macro, _ = compute_metrics(base_logit[val_rows], labels[val_rows],
                                        categories[val_rows], category_vocab)
    _log("BASELINE", f"cached checkpoint's own linear head on this val slice: "
                     f"macro_pr_auc={baseline_macro:.5f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log("DEVICE", str(device))

    x_train = torch.from_numpy(np.ascontiguousarray(emb[train_rows]))
    y_train = torch.from_numpy(labels[train_rows])
    x_val = torch.from_numpy(np.ascontiguousarray(emb[val_rows])).to(device)
    y_val_np = labels[val_rows]
    cat_val = categories[val_rows]

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=False, num_workers=2)

    head = MultiLayerHead(hidden_dim).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    _log("HEAD", f"layers={HEAD_HIDDEN} dropout={HEAD_DROPOUT} feature_dim={head.feature_dim} "
                 f"params={n_params:,}")

    pos_rate = float(y_train.mean())
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-9), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_macro, best_epoch, best_state, since_improve = -1.0, -1, None, 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n_seen += len(yb)
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(x_val).cpu().numpy()
        macro, per_cat = compute_metrics(val_logits, y_val_np, cat_val, category_vocab)
        history.append({"epoch": epoch, "train_loss": total / max(n_seen, 1), "val_macro_pr_auc": macro})
        improved = macro > best_macro
        if improved:
            best_macro, best_epoch = macro, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_per_cat = per_cat
            since_improve = 0
        else:
            since_improve += 1
        _log("EPOCH", f"{epoch}/{EPOCHS} train_loss={total / max(n_seen, 1):.4f} "
                      f"val_macro_pr_auc={macro:.5f} best={best_macro:.5f}@{best_epoch}"
                      f"{' *' if improved else ''}")

        # Selection is on the clean val score, never on train loss -- the
        # split==0 features are contaminated by construction (see docstring).
        if since_improve >= EARLY_STOP_PATIENCE:
            _log("EARLY_STOP", f"no improvement for {EARLY_STOP_PATIENCE} epochs, stopping at {epoch}")
            break

    head.load_state_dict(best_state)
    head.eval()
    torch.save(best_state, os.path.join(OUTPUT_DIR, "head.pt"))

    with torch.no_grad():
        _, feats = head(x_val, return_features=True)
    np.save(os.path.join(OUTPUT_DIR, "val_features.npy"), feats.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "val_row_index.npy"), val_rows)

    result = {
        "train_on": TRAIN_ON,
        "hidden_dim": hidden_dim,
        "head_hidden": list(HEAD_HIDDEN),
        "head_dropout": HEAD_DROPOUT,
        "feature_dim": head.feature_dim,
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
        "baseline_linear_head_macro_pr_auc": baseline_macro,
        "best_macro_pr_auc": best_macro,
        "best_epoch": best_epoch,
        "delta_vs_baseline": best_macro - baseline_macro,
        "best_per_category": best_per_cat,
        "history": history,
        "val_fingerprint": meta.get("val_fingerprint"),
        "source_cache_adapter": meta.get("adapter_dir"),
    }
    with open(os.path.join(OUTPUT_DIR, "head_result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log("DONE", f"best macro_pr_auc={best_macro:.5f} @epoch {best_epoch} "
                 f"(linear baseline {baseline_macro:.5f}, delta {best_macro - baseline_macro:+.5f})")

    if ENABLE_DATASET_PUSH:
        _publish()


def _publish():
    import subprocess
    import sys
    try:
        with open(os.path.join(OUTPUT_DIR, "dataset-metadata.json"), "w") as f:
            json.dump({"id": DATASET_ID, "title": DATASET_TITLE,
                       "licenses": [{"name": "CC0-1.0"}]}, f, indent=2)
        subprocess.check_call([sys.executable, "-m", "kaggle", "datasets", "create",
                               "-p", OUTPUT_DIR, "-r", "skip"])
        _log("PUBLISH", "dataset created")
    except Exception as e:
        _log("PUBLISH", f"FAILED ({type(e).__name__}: {e}) -- output is still in /kaggle/working")


if __name__ == "__main__":
    main()
