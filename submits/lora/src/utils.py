import os

# Hardware-aware env tuning, mirrored from submits/baseline/src/utils.py --
# 20 CPU cores / 200GB RAM / 1x H100 80GB at inference time. /dev/shm is
# RAM-backed, so pointing compile caches there avoids slow disk I/O; thread
# env vars cover OpenMP (numpy/torch), MKL, and rayon (HF tokenizers' Rust
# backend all still uses rayon even without polars).
os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("RAYON_NUM_THREADS", "32")

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

# Must byte-match the formula the LoRA checkpoint was trained on
# (see notebooks/cross-encoder-sft-pipline (3).ipynb and CLAUDE.md).
MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64
MAX_LENGTH = 128


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_needed_item_texts(items_path, needed_ids):
    # items.parquet can be huge -- push the id filter down to pyarrow (via
    # pandas' `filters=`) so only the referenced rows get materialized,
    # instead of reading the full table and filtering/apply()-ing after.
    # No polars: odsai/ecup26-matching-baseline:1.0 doesn't have it installed
    # (confirmed the hard way against the real image -- see CLAUDE.md). Only
    # use packages submits/baseline/src/utils.py already relies on.
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


def _build_texts(match_df, id_to_text):
    # Keep one row per input pair (no dropping) so output row count always
    # matches input row count -- missing item text becomes an empty string
    # rather than silently disappearing from the submission.
    text1 = match_df["id1"].map(id_to_text).fillna("")
    text2 = match_df["id2"].map(id_to_text).fillna("")

    missing = int((text1 == "").sum() + (text2 == "").sum())
    if missing:
        print(f"[WARN] {missing} item-id references had no matching item text (empty text substituted)")

    return text1.tolist(), text2.tolist()


class CrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).view(-1)


def _load_model(model_path, classifier_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        encoder = AutoModel.from_pretrained(model_path, attn_implementation="sdpa")
    except TypeError:
        encoder = AutoModel.from_pretrained(model_path)
    hf_logging.set_verbosity(default_verbosity)

    model = CrossEncoderClassifier(encoder, encoder.config.hidden_size)

    state_dict = torch.load(classifier_path, map_location="cpu")
    model.classifier.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, tokenizer


def _run_inference(model, tokenizer, texts1, texts2, device, batch_size):
    use_amp = device.type == "cuda"
    probs = np.empty(len(texts1), dtype=np.float32)

    with torch.inference_mode():
        for i in range(0, len(texts1), batch_size):
            b1 = texts1[i:i + batch_size]
            b2 = texts2[i:i + batch_size]

            enc = tokenizer(
                text=b1, text_pair=b2,
                padding="max_length", truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

            if use_amp:
                # bf16, not fp16: native full-speed support on H100 (Hopper)
                # and wider exponent range means no GradScaler-style over/
                # underflow games needed -- matches baseline's own default.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**enc)
            else:
                logits = model(**enc)

            probs[i:i + len(b1)] = torch.sigmoid(logits.float()).cpu().numpy()

    return probs


def predict_pipeline(data_path, match_path, model_path, classifier_path, output_csv_path, batch_size=1024):
    t0 = time.time()
    print("=== Starting LoRA prediction pipeline ===")

    print("[1/5] Loading matches ...")
    match_df = pd.read_parquet(match_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/5] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)")

    print("[2/5] Loading needed item texts ...")
    id_to_text = _load_needed_item_texts(data_path, needed_ids)
    print(f"[2/5] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)")

    print("[3/5] Building text pairs ...")
    texts1, texts2 = _build_texts(match_df, id_to_text)
    print(f"[3/5] done ({time.time() - t0:.1f}s)")

    print("[4/5] Loading model ...")
    device = _get_device()
    model, tokenizer = _load_model(model_path, classifier_path, device)
    print(f"[4/5] device={device} ({time.time() - t0:.1f}s)")

    print("[5/5] Running inference ...")
    probs = _run_inference(model, tokenizer, texts1, texts2, device, batch_size=batch_size)
    print(f"[5/5] done ({time.time() - t0:.1f}s)")

    results_df = pd.DataFrame({
        "id1": match_df["id1"].to_numpy(),
        "id2": match_df["id2"].to_numpy(),
        "predict": probs,
    })

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - t0
    print(f"=== Done in {elapsed:.1f}s ({len(texts1) / max(elapsed, 1e-9):.1f} pairs/sec) ===")
    return results_df
