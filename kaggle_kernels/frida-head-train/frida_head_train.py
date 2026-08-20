"""
Trains and compares 5 candidate multi-layer classification heads to replace
FRIDA's single nn.Linear, on the cached pair embeddings from
kaggle_kernels/frida-tpu-embed-cache-human. The winner is warm-started into
kaggle_kernels/frida-llm-tpu-sft so that run does not spend its first hour
teaching a randomly-initialised head what a good one already knows.

WHY 5 ARCHITECTURES, NOT 1
===========================================================================
The encoder forward pass is ~all of the cost of a training step; the head is
a rounding error on top of it. Once representations are cached, architecture
search costs minutes on one T4 instead of hours of TPU per variant -- so
there is no reason to commit to a single hand-picked shape (the earlier
version of this kernel used a fixed (512, 128) with no comparison point).

Five shapes spanning depth and width, dropout/weight-decay held FIXED across
all five so the comparison isolates architecture, not regularization:
  linear_plus  (256,)                   -- closest to the single nn.Linear
                                            this replaces; a capacity floor
  shallow_wide (512, 128)                -- this kernel's own previous
                                            default, kept as a continuity
                                            point against earlier runs
  wide         (1024, 256)               -- more capacity per layer, same
                                            depth as shallow_wide
  deep         (768, 384, 128)           -- one layer deeper
  deep_narrow  (512, 256, 128, 64)       -- deepest, narrows gradually

Each trains independently (fresh init, fresh optimizer, its own early
stopping) on the SAME cached embeddings and SAME split, so the only thing
that differs between runs is the architecture itself.

Two outputs per architecture, plus one combined comparison:
  * architectures/<name>/head.pt          -- state_dict of MultiLayerHead
  * architectures/<name>/val_features.npy -- penultimate activation on val,
                                              for a future CatBoost/GBDT head
                                              on a compact learned feature
                                              space instead of 1536 raw dims
                                              (raw embeddings are exactly
                                              what made berta_catboost_top_final
                                              overfit: 96.11% of importance on
                                              768 near-noise dims, val 0.7467
                                              -> test 0.457).
  * head.pt (dataset ROOT)                -- copy of the WINNING
                                              architecture's head.pt.
                                              frida-llm-tpu-sft's
                                              _find_head_file() checks this
                                              exact root path FIRST, before
                                              falling back to an os.walk that
                                              would otherwise find one of the
                                              5 head.pt files arbitrarily.
  * all_architectures_result.json         -- every architecture's full
                                              history + final metrics + which
                                              one won and why (highest clean
                                              val_macro_pr_auc).

Whoever pushes frida-llm-tpu-sft after this must set its HEAD_HIDDEN/
HEAD_DROPOUT to match the winner recorded here -- MultiLayerHead's shape is
baked into the state_dict; a mismatched HEAD_HIDDEN there fails to load, not
silently trains wrong.

LEAK DISCIPLINE
===========================================================================
The encoder that produced these embeddings was fine-tuned on the canonical
seed-42 TRAIN split, so split==0 rows carry representations the encoder has
already fit. Every architecture here trains on split==0 anyway, and that is
correct rather than sloppy: the head is a component of a model that
frida-llm-tpu-sft will itself train on split==0, so no new contamination is
introduced. split==1 (73,131 pairs) is touched only for evaluation, so the
reported macro PR-AUC is directly comparable to every other val number in
this project and stays an honest holdout for the whole pipeline -- and for
picking a winner among the 5, since all 5 share the exact same holdout.

What that split==0 contamination DOES cause is unusually separable training
features, i.e. heads that will overfit if left alone. Hence dropout, weight
decay, and -- the part that actually matters -- model selection by the clean
val score with early stopping per architecture, never by train loss.

The alternative (train strictly inside split==1, the berta-catboost-noleak
methodology) is available via TRAIN_ON=="val_inner". Right mode if a head is
ever destined for a frozen-encoder submission rather than a warm start, and
the mode that answers "how much of the split==0 score is contamination?" --
but it trains on 4x less data, the exact shortage documented in README.md.
"""
import os
import json
import shutil
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
# Actual published slug -- differs from the cache kernel's own DATASET_ID
# constant (frida-pair-embed-cache-human); doesn't matter for correctness
# since _find_cache_root() falls back to an os.walk search by CONTENT
# (cache_meta.json + pair_embedding.bin together), not by this exact path.
CACHE_HINT = "/kaggle/input/datasets/maximmarkovbb/frida-lora-embed-cache-human"

# "train_split": fit on split==0, evaluate on split==1 (default -- see the
#                leak-discipline note in the module docstring).
# "val_inner":   fit and evaluate strictly inside split==1, the
#                berta-catboost-noleak methodology.
TRAIN_ON = "train_split"

# name -> hidden layer sizes. Dropout is fixed across all five (see module
# docstring) so the comparison isolates architecture, not regularization.
HEAD_ARCHITECTURES = [
    ("linear_plus", (256,)),
    ("shallow_wide", (512, 128)),
    ("wide", (1024, 256)),
    ("deep", (768, 384, 128)),
    ("deep_narrow", (512, 256, 128, 64)),
]
HEAD_DROPOUT = 0.15

EPOCHS = 40
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 8

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_head_train")

DATASET_ID = "maximmarkovbb/frida-multilayer-head"
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
# layer_sizes/dropout have NO default here (unlike the single-architecture
# version this replaces) -- every caller in this file passes them explicitly,
# since there are now 5 real candidates rather than one.
# =====================================================================

class MultiLayerHead(nn.Module):
    def __init__(self, hidden_dim, layer_sizes, dropout):
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


# =====================================================================
# ONE ARCHITECTURE'S TRAIN+EVAL LOOP -- identical logic to the previous
# single-architecture version, just parameterized and given its own output
# subdirectory so nothing from one architecture's run can leak into another's
# (fresh model, fresh optimizer, fresh scheduler each call).
# =====================================================================

def train_one_architecture(name, layer_sizes, dropout, hidden_dim,
                           x_train, y_train, x_val, y_val_np, cat_val,
                           category_vocab, baseline_macro, device):
    arch_dir = os.path.join(OUTPUT_DIR, "architectures", name)
    os.makedirs(arch_dir, exist_ok=True)

    torch.manual_seed(SEED)  # same init seed across architectures -- the
                              # comparison should differ because of SHAPE,
                              # not because one got a luckier random draw
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=False, num_workers=2)

    head = MultiLayerHead(hidden_dim, layer_sizes, dropout).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    _log("HEAD", f"[{name}] layers={layer_sizes} dropout={dropout} "
                 f"feature_dim={head.feature_dim} params={n_params:,}")

    pos_rate = float(y_train.mean())
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-9), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_macro, best_epoch, best_state, best_per_cat, since_improve = -1.0, -1, None, {}, 0
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
        _log("EPOCH", f"[{name}] {epoch}/{EPOCHS} train_loss={total / max(n_seen, 1):.4f} "
                      f"val_macro_pr_auc={macro:.5f} best={best_macro:.5f}@{best_epoch}"
                      f"{' *' if improved else ''}")

        # Selection is on the clean val score, never on train loss -- the
        # split==0 features are contaminated by construction (see docstring).
        if since_improve >= EARLY_STOP_PATIENCE:
            _log("EARLY_STOP", f"[{name}] no improvement for {EARLY_STOP_PATIENCE} epochs, "
                               f"stopping at {epoch}")
            break

    head.load_state_dict(best_state)
    head.eval()
    torch.save(best_state, os.path.join(arch_dir, "head.pt"))

    with torch.no_grad():
        _, feats = head(x_val, return_features=True)
    np.save(os.path.join(arch_dir, "val_features.npy"), feats.cpu().numpy())

    result = {
        "name": name, "layer_sizes": list(layer_sizes), "dropout": dropout,
        "n_params": n_params, "feature_dim": head.feature_dim,
        "baseline_linear_head_macro_pr_auc": baseline_macro,
        "best_macro_pr_auc": best_macro, "best_epoch": best_epoch,
        "delta_vs_baseline": best_macro - baseline_macro,
        "best_per_category": best_per_cat, "history": history,
    }
    with open(os.path.join(arch_dir, "head_result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log("DONE", f"[{name}] best macro_pr_auc={best_macro:.5f} @epoch {best_epoch} "
                 f"(linear baseline {baseline_macro:.5f}, delta {best_macro - baseline_macro:+.5f})")
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    # number every one of the 5 architectures has to beat to be worth shipping.
    baseline_macro, _ = compute_metrics(base_logit[val_rows], labels[val_rows],
                                        categories[val_rows], category_vocab)
    _log("BASELINE", f"cached checkpoint's own linear head on this val slice: "
                     f"macro_pr_auc={baseline_macro:.5f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log("DEVICE", str(device))

    # Loaded ONCE, shared across all 5 architectures -- the embeddings and
    # split never change between them, only the head does.
    x_train = torch.from_numpy(np.ascontiguousarray(emb[train_rows]))
    y_train = torch.from_numpy(labels[train_rows])
    x_val = torch.from_numpy(np.ascontiguousarray(emb[val_rows])).to(device)
    y_val_np = labels[val_rows]
    cat_val = categories[val_rows]

    results = []
    for name, layer_sizes in HEAD_ARCHITECTURES:
        result = train_one_architecture(name, layer_sizes, HEAD_DROPOUT, hidden_dim,
                                        x_train, y_train, x_val, y_val_np, cat_val,
                                        category_vocab, baseline_macro, device)
        results.append(result)

    winner = max(results, key=lambda r: r["best_macro_pr_auc"])
    _log("WINNER", f"{winner['name']} layers={winner['layer_sizes']} "
                   f"macro_pr_auc={winner['best_macro_pr_auc']:.5f} "
                   f"(vs linear baseline {baseline_macro:.5f}, "
                   f"delta {winner['delta_vs_baseline']:+.5f})")

    # Winner's head.pt copied to the dataset ROOT -- frida-llm-tpu-sft's
    # _find_head_file() checks exactly this path first, so it picks up the
    # winner without needing to know its name. The other 4 stay under
    # architectures/<name>/ for inspection/comparison only.
    shutil.copy(os.path.join(OUTPUT_DIR, "architectures", winner["name"], "head.pt"),
               os.path.join(OUTPUT_DIR, "head.pt"))

    summary = {
        "train_on": TRAIN_ON, "hidden_dim": hidden_dim,
        "baseline_linear_head_macro_pr_auc": baseline_macro,
        "winner": {"name": winner["name"], "layer_sizes": winner["layer_sizes"],
                  "dropout": HEAD_DROPOUT, "macro_pr_auc": winner["best_macro_pr_auc"]},
        "all_architectures": [
            {"name": r["name"], "layer_sizes": r["layer_sizes"], "n_params": r["n_params"],
             "best_macro_pr_auc": r["best_macro_pr_auc"], "best_epoch": r["best_epoch"],
             "delta_vs_baseline": r["delta_vs_baseline"]}
            for r in results
        ],
        "val_fingerprint": meta.get("val_fingerprint"),
        "source_cache_adapter": meta.get("adapter_dir"),
    }
    with open(os.path.join(OUTPUT_DIR, "all_architectures_result.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _log("DONE", f"5 architectures compared, winner={winner['name']} "
                 f"({winner['best_macro_pr_auc']:.5f})")

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
