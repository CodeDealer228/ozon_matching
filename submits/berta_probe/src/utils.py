import os

# Hardware-aware env tuning for the test stand (1x H100 80GB / 20 CPU cores /
# 200GB RAM at inference time, per CLAUDE.md). /dev/shm is RAM-backed, so
# pointing compile caches there avoids slow disk I/O. Thread count is read
# from the actual core count instead of a hardcoded guess (the baseline/lora
# scripts hardcode 32, which overshoots the real 20-core box).
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer, logging as hf_logging

# Must byte-match the formula this repo trains/evaluates against (see
# CLAUDE.md "Text preprocessing"): raw attribute string slice, no json.loads
# (parsing 13M+ JSON blobs OOMs at full items.parquet scale).
MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64
MAX_LENGTH = 128


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_needed_item_texts(items_path, needed_ids):
    # Push the id filter down to pyarrow (via pandas' `filters=`) so only the
    # referenced rows get materialized, instead of reading the full table and
    # filtering/apply()-ing after -- items.parquet is 13M+ rows / 4.1GB at
    # judging time. No polars: odsai/ecup26-matching-baseline:1.0 doesn't
    # have it installed (see CLAUDE.md); only pandas/pyarrow/torch/
    # transformers are guaranteed present.
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


def build_texts(match_df, id_to_text):
    # Vectorized pandas .map(), not a python-level df.apply(axis=1)/iterrows()
    # loop -- the organizer baseline's row-wise apply()+json.loads() and
    # iterrows() pair-builder are the real bottleneck at 365K+ pairs, not
    # batch size. Keep one row per input pair (no dropping) so output row
    # count always matches input row count.
    text1 = match_df["id1"].map(id_to_text).fillna("")
    text2 = match_df["id2"].map(id_to_text).fillna("")

    missing = int((text1 == "").sum() + (text2 == "").sum())
    if missing:
        print(f"[WARN] {missing} item-id references had no matching item text (empty text substituted)", flush=True)

    return text1.tolist(), text2.tolist()


def _resolve_model_class(model_path):
    """Instantiate the exact class named in config.json's `architectures`
    instead of relying on AutoModel's model_type->class default mapping.
    Needed for checkpoints published as an encoder-only export (e.g.
    T5EncoderModel) where config.model_type ("t5") would make AutoModel
    default to the full encoder-decoder T5Model and either error on missing
    decoder weights or waste memory loading a decoder we'll never call."""
    import transformers as _tf

    config = AutoConfig.from_pretrained(model_path)
    arch_name = (config.architectures or [None])[0]
    cls = getattr(_tf, arch_name, None) if arch_name else None
    return cls or AutoModel


def load_encoder(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_cls = _resolve_model_class(model_path)

    load_kwargs = {}
    if device.type == "cuda":
        # Load weights straight into bf16 (not just autocast at inference) --
        # halves VRAM/host-RAM footprint, which matters more as the encoder
        # grows from the baseline's 33M params. bf16, not fp16: native
        # full-speed support on H100 (Hopper) and no over/underflow games.
        load_kwargs["torch_dtype"] = torch.bfloat16

    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        encoder = model_cls.from_pretrained(model_path, attn_implementation="sdpa", **load_kwargs)
    except (TypeError, ValueError):
        # TypeError: this transformers version doesn't accept the kwarg at all.
        # ValueError: kwarg is accepted but this architecture has no SDPA path
        # (e.g. T5EncoderModel on some transformers versions -- observed for
        # real on the judging image even though it loaded fine in local dev
        # with a newer transformers). Fall back to the model's own default
        # attention implementation rather than hard-failing the submission.
        encoder = model_cls.from_pretrained(model_path, **load_kwargs)
    hf_logging.set_verbosity(default_verbosity)

    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer


def _encode_batch(encoder, tokenizer, texts1, texts2, device):
    enc = tokenizer(
        text=texts1, text_pair=texts2,
        padding="max_length", truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    # Only forward the keys the model actually accepts -- token_type_ids is
    # absent for RoBERTa/XLM-R-family and T5 tokenizers, present for BERT.
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()
           if k in ("input_ids", "attention_mask", "token_type_ids")}

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = encoder(**enc)
        else:
            out = encoder(**enc)

    return out.last_hidden_state[:, 0, :].float().cpu().numpy()


def extract_cls_embeddings(encoder, tokenizer, texts1, texts2, device, batch_size, show_progress=True):
    """CLS-token (position-0) pooling over the concatenated pair, batched
    with adaptive-size CUDA OOM fallback: a single over-large batch halves
    and retries instead of hard-failing the whole run."""
    n = len(texts1)
    hidden_dim = encoder.config.hidden_size
    result = np.empty((n, hidden_dim), dtype=np.float32)

    bs = batch_size
    i = 0
    t_start = t_last_log = time.time()
    while i < n:
        b1, b2 = texts1[i:i + bs], texts2[i:i + bs]
        try:
            emb = _encode_batch(encoder, tokenizer, b1, b2, device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            bs = max(8, bs // 2)
            print(f"[ENCODE] CUDA OOM -> retrying with batch_size={bs}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        result[i:i + len(b1)] = emb
        i += len(b1)
        if show_progress and time.time() - t_last_log > 15:
            rate = i / max(time.time() - t_start, 1e-9)
            print(f"[ENCODE] {i:,}/{n:,} ({rate:.1f} pairs/sec, batch_size={bs})", flush=True)
            t_last_log = time.time()

    return result


def train_pipeline(items_path, matches_path, model_path, output_classifier_path,
                    batch_size=4096, n_samples=None, seed=42):
    """Fit StandardScaler+LogisticRegression on frozen [CLS] embeddings, the
    same shape as the organizer baseline (submits/baseline). Not used at
    judging time (matches_path there has no `target` column) -- this is the
    local/offline step that produces the classifier joblib shipped in the
    submission zip."""
    t0 = time.time()
    print("=== Starting Training Pipeline ===", flush=True)

    print("[1/5] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    if "target" not in match_df.columns:
        raise ValueError("train_pipeline needs a matches file with a `target` column")
    if n_samples is not None and n_samples < len(match_df):
        match_df = match_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    labels = (match_df["target"].astype(float) >= 0.5).astype(int).to_numpy()
    print(f"[1/5] pairs={len(match_df):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/5] Loading needed item texts ...", flush=True)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    id_to_text = load_needed_item_texts(items_path, needed_ids)
    print(f"[2/5] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/5] Building text pairs ...", flush=True)
    texts1, texts2 = build_texts(match_df, id_to_text)
    print(f"[3/5] done ({time.time() - t0:.1f}s)", flush=True)

    print("[4/5] Loading encoder + extracting embeddings ...", flush=True)
    device = get_device()
    encoder, tokenizer = load_encoder(model_path, device)
    print(f"[4/5] device={device}", flush=True)
    embeddings = extract_cls_embeddings(encoder, tokenizer, texts1, texts2, device, batch_size=batch_size)
    print(f"[4/5] done ({time.time() - t0:.1f}s)", flush=True)

    print("[5/5] Fitting StandardScaler + LogisticRegression ...", flush=True)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=1234)),
    ])
    clf.fit(embeddings, labels)
    joblib.dump(clf, output_classifier_path)
    print(f"[5/5] saved -> {output_classifier_path} ({time.time() - t0:.1f}s)", flush=True)

    print("=== Training Pipeline Complete ===", flush=True)
    return clf


def predict_pipeline(items_path, matches_path, model_path, classifier_path, output_csv_path, batch_size=4096):
    t0 = time.time()
    print("=== Starting prediction pipeline ===", flush=True)

    print("[1/5] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/5] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/5] Loading needed item texts ...", flush=True)
    id_to_text = load_needed_item_texts(items_path, needed_ids)
    print(f"[2/5] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/5] Building text pairs ...", flush=True)
    texts1, texts2 = build_texts(match_df, id_to_text)
    print(f"[3/5] done ({time.time() - t0:.1f}s)", flush=True)

    print("[4/5] Loading encoder ...", flush=True)
    device = get_device()
    encoder, tokenizer = load_encoder(model_path, device)
    print(f"[4/5] device={device} ({time.time() - t0:.1f}s)", flush=True)

    print("[5/5] Extracting embeddings + scoring ...", flush=True)
    embeddings = extract_cls_embeddings(encoder, tokenizer, texts1, texts2, device, batch_size=batch_size)
    clf = joblib.load(classifier_path)
    predict = clf.predict_proba(embeddings)[:, 1]
    print(f"[5/5] done ({time.time() - t0:.1f}s)", flush=True)

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
    print(f"=== Done in {elapsed:.1f}s ({len(texts1) / max(elapsed, 1e-9):.1f} pairs/sec) ===", flush=True)
    return results_df
