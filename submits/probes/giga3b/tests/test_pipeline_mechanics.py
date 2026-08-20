"""
Local mechanics smoke test for the Giga cross-encoder submission pipeline.

Validates everything EXCEPT the real model (int8 needs bitsandbytes, which
needs an actual CUDA GPU even just to load -- untestable on a CPU-only dev
box): parquet IO, the pair-text formula (instruction + manual "Товар 1/2:"
concatenation, since this tokenizer has no working text_pair join), classifier
scoring, output-row order and count, predict range [0,1], and graceful
handling of a pair whose item id is missing from items.parquet. Uses a
deterministic dummy encoder + a classifier fit on synthetic embeddings, so it
runs with only pandas/numpy/scikit-learn installed -- no torch, no GPU.

    python submits/giga3b_probe/tests/test_pipeline_mechanics.py
"""

import os
import sys
import hashlib
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMIT_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SUBMIT_DIR))
sys.path.insert(0, SUBMIT_DIR)

from src.utils import predict_pipeline, build_pair_texts, INSTRUCTION  # noqa: E402

DIM = 64


class DummyEncoder:
    """Deterministic char-hash -> unit-ish vector. Mimics .encode_pairs() contract."""

    def encode_pairs(self, texts, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vals = np.frombuffer((h * ((DIM // 32) + 1))[:DIM], dtype=np.uint8)
            out[i] = (vals.astype(np.float32) / 127.5) - 1.0
        return out


def _make_dummy_classifier(path):
    """A LogisticRegression fit on synthetic separable data -- just enough
    to exercise predict_pipeline's classifier.predict_proba call path."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    x_pos = rng.normal(loc=1.0, scale=1.0, size=(50, DIM))
    x_neg = rng.normal(loc=-1.0, scale=1.0, size=(50, DIM))
    x = np.vstack([x_pos, x_neg]).astype(np.float32)
    y = np.array([1] * 50 + [0] * 50)

    clf = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    clf.fit(x, y)
    joblib.dump(clf, path)


def _make_fixtures(tmp):
    """Sample real pairs/items if available, else synthesize."""
    matches_src = os.path.join(REPO, "data", "matches.parquet")
    items_src = os.path.join(REPO, "data", "items_human.parquet")

    items_path = os.path.join(tmp, "items.parquet")
    matches_path = os.path.join(tmp, "matches.parquet")

    if os.path.exists(matches_src) and os.path.exists(items_src):
        pairs = pd.read_parquet(matches_src, columns=["id1", "id2"]).head(2000)
        needed = pd.unique(pd.concat([pairs["id1"], pairs["id2"]]))
        items = pd.read_parquet(
            items_src, columns=["id", "name", "category", "attributes"],
            engine="pyarrow", filters=[("id", "in", list(needed))],
        )
        missing_id = -999999999
        extra = pd.DataFrame({"id1": [needed[0]], "id2": [missing_id]})
        pairs = pd.concat([pairs, extra], ignore_index=True)
        items.to_parquet(items_path)
        pairs.to_parquet(matches_path)
        return items_path, matches_path, len(pairs), missing_id

    # Fallback synthetic fixtures.
    items = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Кофе молотый", "Кофе молотый 250г", "Чайник", None],
        "category": ["Продукты", "Продукты", "Техника", "Техника"],
        "attributes": ['{"вес":"250"}', '{"вес":"250"}', None, '{"мощность":"2000"}'],
    })
    pairs = pd.DataFrame({"id1": [1, 3, 1], "id2": [2, 4, 777]})  # 777 absent
    items.to_parquet(items_path)
    pairs.to_parquet(matches_path)
    return items_path, matches_path, len(pairs), 777


def main():
    tmp = tempfile.mkdtemp(prefix="giga_probe_test_")
    items_path, matches_path, n_pairs, missing_id = _make_fixtures(tmp)
    out_path = os.path.join(tmp, "out.csv")
    classifier_path = os.path.join(tmp, "classifier.joblib")
    _make_dummy_classifier(classifier_path)

    # pair-text formula check: instruction + labeled concatenation, no raw
    # text_pair (this tokenizer glues text_pair with zero separator). Item
    # truncation to MAX_ITEM_CHARS happens upstream in load_needed_item_texts
    # (same vectorized .str.slice() formula as the other cross-encoder
    # submissions), not in build_pair_texts itself.
    id_to_text = {0: "short item text"}
    sample = build_pair_texts(pd.DataFrame({"id1": [0], "id2": [0]}), id_to_text)
    text0 = sample[0]
    assert text0.startswith(INSTRUCTION), text0[:40]
    assert "Товар 1: short item text" in text0 and "\nТовар 2: short item text" in text0

    out = predict_pipeline(items_path, matches_path, encoder=DummyEncoder(),
                            classifier_path=classifier_path, output_csv_path=out_path,
                            batch_size=256)

    # --- assertions ---
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["id1", "id2", "predict"], list(df.columns)
    assert len(df) == n_pairs, f"row count {len(df)} != {n_pairs}"

    p = df["predict"].to_numpy()
    assert np.all((p >= 0.0) & (p <= 1.0)), f"predict out of [0,1]: {p.min()},{p.max()}"

    # order preserved: output id1/id2 must equal input pair order
    src_pairs = pd.read_parquet(matches_path)
    assert df["id1"].tolist() == src_pairs["id1"].tolist()
    assert df["id2"].tolist() == src_pairs["id2"].tolist()

    # missing-item pair still scored (not dropped)
    miss_row = df[df["id2"] == missing_id]
    assert len(miss_row) >= 1, "missing-id pair dropped"

    print("\n[TEST] all mechanics assertions passed "
          f"({len(df):,} rows, predict in [{p.min():.3f}, {p.max():.3f}])")


if __name__ == "__main__":
    main()
