"""
Same recall@k / hard-negative-mining test as
../retrieval-recall-hard-negative-mining, but scoring real
sergeyzh/rubert-mini-frida item embeddings (dense, semantic) instead of
TF-IDF cosine (sparse, lexical-only). TF-IDF came back weak -- recall@10
ranged 0.015-0.35 across categories, nowhere near the ~0.9 target; pure
lexical overlap can't capture paraphrase/synonym/word-order variation.
This is the direct comparison point on the SAME 20 categories, SAME
recall@k methodology, SAME mining logic -- only the item vector source
changes.

Every labeled pair (id1, id2, target) in matches.parquet exists BECAUSE
organizers' retrieval surfaced id2 as a plausible candidate for id1 (or
vice versa) -- true regardless of target (0 or 1 both count as retrieval-
plausible, only the label differs). recall@k here checks: does OUR
retrieval (rubert-mini-frida cosine) reproduce that, across ALL 365,654
pairs, not just the 26% positives. If it does well, the OTHER top-k
neighbors (excluding anything already labeled) are hard-negative
candidates by construction -- see ../retrieval-recall-hard-negative-mining
for the full reasoning, unchanged here.

CPU only -- embeddings are already computed (../rubert-mini-frida-item-embeddings,
published as maxamsecret/rubert-mini-frida-item-embeddings), this kernel
just loads them and does nearest-neighbor search per category. No GPU, no
model, no tokenization.

Categories processed smallest-first, same caution as the TF-IDF version:
a full N-vs-N neighbor search per category (up to ~46K items) on DENSE
312-dim vectors has different cost characteristics than on sparse TF-IDF
(dense BLAS matmul, likely faster per comparison, but not measured at this
scale before either) -- smallest-first means most categories get a real
result even if the biggest one needs a different approach.
"""

import os
import time
import json

import numpy as np
import pandas as pd
import polars as pl
from sklearn.neighbors import NearestNeighbors

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
EMBEDDINGS_HINT = "/kaggle/input/datasets/maxamsecret/rubert-mini-frida-item-embeddings/rubert_mini_frida_item_embeddings"

K_VALIDATE = 50
RECALL_CUTOFFS = [1, 5, 10, 20, 50]
K_MINE = 10

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "rubert_mini_frida_recall_mining")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():7.1f}s] {stage}: {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _find_embeddings_dir():
    if os.path.isdir(EMBEDDINGS_HINT) and os.path.exists(os.path.join(EMBEDDINGS_HINT, "meta.json")):
        return EMBEDDINGS_HINT
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "meta.json" in files and "embeddings.npy" in files and "ids.npy" in files:
                return root
    raise RuntimeError(
        "No item-embeddings dir found under /kaggle/input -- expected "
        "maxamsecret/rubert-mini-frida-item-embeddings mounted (dataset_sources)."
    )


def process_category(category, items_all, matches_all, id_to_row_global, embeddings_global):
    t_cat = time.perf_counter()

    items_cat = items_all.filter(pl.col("category") == category)
    cat_ids = items_cat.get_column("id").to_numpy()
    n_items = len(cat_ids)

    cat_id_set = set(cat_ids.tolist())
    pairs = matches_all.filter(
        pl.col("id1").is_in(cat_id_set) & pl.col("id2").is_in(cat_id_set)
    ).select(["id1", "id2", "label"]).to_pandas()
    n_pairs = len(pairs)
    _log(category, f"items={n_items:,} labeled_pairs={n_pairs:,}")

    # Pull this category's rows out of the GLOBAL embedding matrix -- no
    # re-encoding, just an index gather.
    rows = np.fromiter((id_to_row_global[int(i)] for i in cat_ids), dtype=np.int64, count=n_items)
    cat_embeddings = np.ascontiguousarray(embeddings_global[rows]).astype(np.float32)
    id_to_row = {int(item_id): i for i, item_id in enumerate(cat_ids)}

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=os.cpu_count() or 4)
    nn.fit(cat_embeddings)
    n_query = min(K_VALIDATE + 1, n_items)
    distances, indices = nn.kneighbors(cat_embeddings, n_neighbors=n_query)
    _log(category, f"kneighbors done ({time.perf_counter() - t_cat:.1f}s so far)")

    neighbor_ids = np.empty((n_items, n_query - 1), dtype=np.int64)
    neighbor_cos = np.empty((n_items, n_query - 1), dtype=np.float32)
    for row in range(n_items):
        row_idx = indices[row]
        row_dist = distances[row]
        keep = row_idx != row
        kept_idx = row_idx[keep][: n_query - 1]
        kept_dist = row_dist[keep][: n_query - 1]
        neighbor_ids[row, : len(kept_idx)] = cat_ids[kept_idx]
        neighbor_cos[row, : len(kept_idx)] = 1.0 - kept_dist
        if len(kept_idx) < n_query - 1:
            neighbor_ids[row, len(kept_idx):] = -1
            neighbor_cos[row, len(kept_idx):] = np.nan

    ranks = []
    for id1, id2 in zip(pairs["id1"].to_numpy(), pairs["id2"].to_numpy()):
        r1, r2 = id_to_row.get(int(id1)), id_to_row.get(int(id2))
        if r1 is None or r2 is None:
            continue
        pos_of_id2_in_id1 = np.where(neighbor_ids[r1] == id2)[0]
        pos_of_id1_in_id2 = np.where(neighbor_ids[r2] == id1)[0]
        rank_1to2 = int(pos_of_id2_in_id1[0]) + 1 if len(pos_of_id2_in_id1) else None
        rank_2to1 = int(pos_of_id1_in_id2[0]) + 1 if len(pos_of_id1_in_id2) else None
        best_rank = min([r for r in (rank_1to2, rank_2to1) if r is not None], default=None)
        ranks.append((rank_1to2, rank_2to1, best_rank))

    ranks_df = pd.DataFrame(ranks, columns=["rank_1to2", "rank_2to1", "rank_best"])
    recall = {}
    for k in RECALL_CUTOFFS:
        recall[f"recall@{k}_best"] = float((ranks_df["rank_best"] <= k).mean())
        recall[f"recall@{k}_1to2"] = float((ranks_df["rank_1to2"] <= k).mean())
        recall[f"recall@{k}_2to1"] = float((ranks_df["rank_2to1"] <= k).mean())
    _log(category, "recall@k(best): " + " | ".join(
        f"@{k}={recall[f'recall@{k}_best']:.3f}" for k in RECALL_CUTOFFS))

    labeled_partners = {}
    for id1, id2 in zip(pairs["id1"].to_numpy(), pairs["id2"].to_numpy()):
        labeled_partners.setdefault(int(id1), set()).add(int(id2))
        labeled_partners.setdefault(int(id2), set()).add(int(id1))

    mined_rows = []
    for row in range(n_items):
        item_id = int(cat_ids[row])
        already = labeled_partners.get(item_id, set())
        kept = 0
        for cand_id, cand_cos in zip(neighbor_ids[row], neighbor_cos[row]):
            if cand_id == -1 or kept >= K_MINE:
                break
            cand_id = int(cand_id)
            if cand_id in already:
                continue
            mined_rows.append((item_id, cand_id, float(cand_cos), category))
            kept += 1

    mined_df = pd.DataFrame(mined_rows, columns=["id1", "id2", "embedding_cosine", "category"])
    _log(category, f"mined {len(mined_df):,} candidate hard-negative pairs "
                    f"(category total {time.perf_counter() - t_cat:.1f}s)")

    return {"category": category, "n_items": n_items, "n_pairs": n_pairs, "recall": recall}, mined_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    emb_dir = _find_embeddings_dir()
    with open(os.path.join(emb_dir, "meta.json")) as f:
        emb_meta = json.load(f)
    embeddings_global = np.load(os.path.join(emb_dir, "embeddings.npy"))
    ids_global = np.load(os.path.join(emb_dir, "ids.npy"))
    id_to_row_global = {int(item_id): i for i, item_id in enumerate(ids_global)}
    _log("SETUP", f"embeddings loaded: {embeddings_global.shape}, meta={emb_meta}")

    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")

    matches = pl.read_parquet(matches_path)
    matches = matches.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    matches = matches.with_columns((pl.col("target") >= 0.5).cast(pl.Int8).alias("label"))
    items = pl.read_parquet(items_path, columns=["id", "name", "category", "attributes"])

    cat_counts = items.group_by("category").agg(pl.len().alias("n")).sort("n")
    categories = cat_counts.get_column("category").to_list()
    _log("DATA", f"matches={matches.height:,} | categories={len(categories)}")
    for row in cat_counts.iter_rows(named=True):
        _log("DATA", f"  {row['category']}: {row['n']:,} items")

    all_summaries = []
    all_mined = []
    for category in categories:
        summary, mined_df = process_category(category, items, matches, id_to_row_global, embeddings_global)
        all_summaries.append(summary)
        all_mined.append(mined_df)

    mined_all = pd.concat(all_mined, ignore_index=True)
    mined_all.to_parquet(os.path.join(OUTPUT_DIR, "mined_hard_negatives.parquet"), index=False)
    with open(os.path.join(OUTPUT_DIR, "recall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    macro_recall_at_1 = float(np.mean([s["recall"]["recall@1_best"] for s in all_summaries]))
    macro_recall_at_10 = float(np.mean([s["recall"]["recall@10_best"] for s in all_summaries]))
    _log("DONE", f"macro recall@1={macro_recall_at_1:.4f} recall@10={macro_recall_at_10:.4f} "
                 f"(mean over {len(categories)} categories) | "
                 f"mined_hard_negatives.parquet: {len(mined_all):,} candidate pairs | "
                 f"total wall time {_elapsed():.1f}s")


if __name__ == "__main__":
    main()
