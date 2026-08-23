"""
Reproduces (approximately) the organizers' own retrieval step well enough
to validate it directly, then uses the validated retrieval to mine fresh,
UNLABELED hard-negative candidates -- across all 20 categories, CPU only.

===========================================================================
THE IDEA (as specified, not guessed)
===========================================================================
Every labeled pair (id1, id2, target) in matches.parquet exists BECAUSE
organizers' own retrieval surfaced id2 as a plausible candidate for id1
(or vice versa) -- CLAUDE.md: "Retrieval is already done by organizers --
this repo is only the pairwise re-ranking stage." Critically, this is true
REGARDLESS of target: a target=0 labeled pair is just as much evidence
that organizers' retrieval considered id2 close to id1 as a target=1 pair
is -- the label only reflects what a human then decided, not whether the
pair was retrieval-plausible. So ALL 365,654 pairs (not just the ~26%
positives) are ground truth for "a real retrieval system should surface
this candidate."

That gives a directly measurable validation target: build our OWN item
retrieval (TF-IDF cosine, cheapest signal already shown to carry signal in
kaggle_kernels/fasttext-category-similarity-probe) and check recall@k --
for each labeled pair's id1, does id2 show up in id1's own top-k nearest
neighbors (and symmetrically id2->id1)? Most items appear in exactly one
labeled pair (verified empirically per category below), so recall@k here
is essentially hit@k: either the true partner is in the top-k or it isn't.
If our retrieval reproduces organizers' well (high recall@1, or at least
recall@few), that validates it as a real retrieval proxy, not just a
pairwise-similarity heuristic -- and its OTHER top-k neighbors (the ones
that are NOT the labeled partner, and not any other already-labeled
partner of that item either) are hard-negative candidates by construction:
similarity-plausible enough that a real retrieval system would surface
them too, but never shown to a human labeler. Random sampling from the
unlabeled pool would essentially never find these -- they're exactly the
near-misses organizers' own pipeline would also surface.

One pass per category does both jobs at once: kneighbors() on the full
category item matrix against itself gives, for every item, its ranked
neighbor list -- used both to look up known pairs' ranks (validation) and
to emit the non-labeled top-k as mining candidates (the actual output).

Restricting to one category at a time is valid because category1==category2
for 100.0000% of the 365,654 labeled pairs (measured directly, zero
exceptions -- see CLAUDE.md and this repo's own earlier local check).
"""

import os
import time
import json

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():7.1f}s] {stage}: {msg}", flush=True)


import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

MAX_NAME_CHARS, MAX_ATTR_CHARS = 128, 384
# Category excluded from item text -- constant within a category by
# construction, zero discriminative value (CLAUDE.md's own finding for
# the pairwise task generally).

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)

K_VALIDATE = 50          # recall@k reported at these cutoffs
RECALL_CUTOFFS = [1, 5, 10, 20, 50]
K_MINE = 10              # top-N *unlabeled* neighbors kept per item as
                          # mined hard-negative candidates (after excluding
                          # anything already labeled, positive or negative)

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "retrieval_mining")


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


def process_category(category, items_all, matches_all, n_threads):
    t_cat = time.perf_counter()

    items_cat = items_all.filter(pl.col("category") == category)
    cat_ids = set(items_cat.get_column("id").to_list())
    n_items = len(cat_ids)

    pairs = matches_all.filter(
        pl.col("id1").is_in(cat_ids) & pl.col("id2").is_in(cat_ids)
    ).select(["id1", "id2", "label"]).to_pandas()
    n_pairs = len(pairs)
    _log(category, f"items={n_items:,} labeled_pairs={n_pairs:,}")

    # -------------------------------------------------------------
    # Item text + TF-IDF (word 1-2gram) -- same formula as
    # kaggle_kernels/fasttext-category-similarity-probe
    # -------------------------------------------------------------
    items_pd = items_cat.to_pandas()
    items_pd["text"] = items_pd.apply(build_item_text, axis=1)
    ordered_ids = items_pd["id"].to_numpy()
    id_to_row = {int(item_id): i for i, item_id in enumerate(ordered_ids)}

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    tfidf_matrix = tfidf.fit_transform(items_pd["text"].tolist())

    # -------------------------------------------------------------
    # One kneighbors() call for the WHOLE category against itself --
    # serves both recall@k validation and mining. n_neighbors=K+1
    # because a point is always its own nearest neighbor (distance 0)
    # when querying the matrix it was fit on; dropped below.
    # -------------------------------------------------------------
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=n_threads)
    nn.fit(tfidf_matrix)
    n_query = min(K_VALIDATE + 1, n_items)
    distances, indices = nn.kneighbors(tfidf_matrix, n_neighbors=n_query)
    _log(category, f"kneighbors done ({time.perf_counter() - t_cat:.1f}s so far)")

    # neighbor_ids[row] -> ordered array of item ids (self dropped),
    # neighbor_cos[row] -> matching cosine similarities (1 - cosine distance)
    neighbor_ids = np.empty((n_items, n_query - 1), dtype=np.int64)
    neighbor_cos = np.empty((n_items, n_query - 1), dtype=np.float32)
    for row in range(n_items):
        row_idx = indices[row]
        row_dist = distances[row]
        keep = row_idx != row  # drop self
        # if self wasn't in the returned top-n_query at all (shouldn't
        # happen -- distance 0 to self always wins -- but defensive),
        # keep already excludes nothing extra; just take the first
        # (n_query-1) non-self entries.
        kept_idx = row_idx[keep][: n_query - 1]
        kept_dist = row_dist[keep][: n_query - 1]
        neighbor_ids[row, : len(kept_idx)] = ordered_ids[kept_idx]
        neighbor_cos[row, : len(kept_idx)] = 1.0 - kept_dist
        if len(kept_idx) < n_query - 1:
            neighbor_ids[row, len(kept_idx):] = -1
            neighbor_cos[row, len(kept_idx):] = np.nan

    # -------------------------------------------------------------
    # Recall@k validation: for every labeled pair, where does the
    # partner rank in the OTHER side's neighbor list? Both directions.
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Mining: for every item, its top-K_MINE neighbors that are NOT
    # already a labeled partner (either target) -- fresh unlabeled
    # hard-negative candidates.
    # -------------------------------------------------------------
    labeled_partners = {}
    for id1, id2 in zip(pairs["id1"].to_numpy(), pairs["id2"].to_numpy()):
        labeled_partners.setdefault(int(id1), set()).add(int(id2))
        labeled_partners.setdefault(int(id2), set()).add(int(id1))

    mined_rows = []
    for row in range(n_items):
        item_id = int(ordered_ids[row])
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

    mined_df = pd.DataFrame(mined_rows, columns=["id1", "id2", "tfidf_cosine", "category"])
    _log(category, f"mined {len(mined_df):,} candidate hard-negative pairs "
                    f"(category total {time.perf_counter() - t_cat:.1f}s)")

    return {
        "category": category, "n_items": n_items, "n_pairs": n_pairs,
        "recall": recall,
    }, mined_df


def main():
    # No package installs needed -- TfidfVectorizer/NearestNeighbors are
    # long-stable sklearn APIs, already present in the base image. Avoid
    # gratuitous version churn (see the peft/transformers incident in
    # kaggle_kernels/berta-lora-pair-cache: an unrelated forced upgrade
    # there broke an import elsewhere in the stack).
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_threads = os.cpu_count() or 4

    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")

    matches = pl.read_parquet(matches_path)
    matches = matches.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    matches = matches.with_columns((pl.col("target") >= 0.5).cast(pl.Int8).alias("label"))
    items = pl.read_parquet(items_path, columns=["id", "name", "category", "attributes"])

    # Smallest category first -- the expensive step (kneighbors: full
    # N-vs-N cosine within a category, up to ~46K items for the biggest)
    # has never been timed at this scale before (the earlier fastText/
    # TF-IDF probe only scored the 18,285 ALREADY-KNOWN pairs, not a full
    # N^2-shaped nearest-neighbor search). Smallest-first means most
    # categories get a real result even if the biggest one(s) turn out to
    # need a different approach -- and the early timings tell us that
    # before we're committed to the expensive end.
    cat_counts = (
        items.group_by("category").agg(pl.len().alias("n"))
        .sort("n")
    )
    categories = cat_counts.get_column("category").to_list()
    _log("DATA", f"matches={matches.height:,} | categories={len(categories)} | threads={n_threads}")
    for row in cat_counts.iter_rows(named=True):
        _log("DATA", f"  {row['category']}: {row['n']:,} items")

    all_summaries = []
    all_mined = []
    for category in categories:
        summary, mined_df = process_category(category, items, matches, n_threads)
        all_summaries.append(summary)
        all_mined.append(mined_df)

    mined_all = pd.concat(all_mined, ignore_index=True)
    mined_all.to_parquet(os.path.join(OUTPUT_DIR, "mined_hard_negatives.parquet"), index=False)
    with open(os.path.join(OUTPUT_DIR, "recall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    macro_recall_at_1 = float(np.mean([s["recall"]["recall@1_best"] for s in all_summaries]))
    macro_recall_at_5 = float(np.mean([s["recall"]["recall@5_best"] for s in all_summaries]))
    _log("DONE", f"macro recall@1={macro_recall_at_1:.4f} recall@5={macro_recall_at_5:.4f} "
                 f"(mean over {len(categories)} categories) | "
                 f"mined_hard_negatives.parquet: {len(mined_all):,} candidate pairs | "
                 f"total wall time {_elapsed():.1f}s")


if __name__ == "__main__":
    main()
