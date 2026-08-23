import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 8))

import time

import numpy as np
import pandas as pd
import polars as pl
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
    # items.parquet is large (13M+ rows) -- only scan/transform the rows this
    # match set actually references instead of processing the full table.
    items = (
        pl.scan_parquet(items_path)
        .filter(pl.col("id").is_in(needed_ids))
        .select(["id", "name", "category", "attributes"])
        .collect()
    )

    items = items.with_columns([
        pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS),
        pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_CATEGORY_CHARS),
        pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS),
    ])

    items = items.with_columns(
        (
            pl.lit("Name: ") + pl.col("name") +
            pl.lit(" Category: ") + pl.col("category") +
            pl.lit(" Attributes: ") + pl.col("attributes")
        ).str.slice(0, MAX_ITEM_CHARS).alias("text")
    )

    return dict(zip(items.get_column("id").to_list(), items.get_column("text").to_list()))


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
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**enc)
            else:
                logits = model(**enc)

            probs[i:i + len(b1)] = torch.sigmoid(logits.float()).cpu().numpy()

    return probs


def predict_pipeline(data_path, match_path, model_path, classifier_path, output_csv_path, batch_size=512):
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
