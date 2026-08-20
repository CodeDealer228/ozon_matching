"""
CPU-only check: for a 10,000-item sample of items_human.parquet, tokenize
name/category/attributes SEPARATELY (each field on its own, after this
project's standard char-truncation limits, no prefix, no concatenation) with
ai-forever/FRIDA's tokenizer (byte-level BPE, RobertaTokenizer under the
hood -- see CLAUDE.md), and report per-field token-length distributions plus
the combined-text char/token ratio.

v1 (1000 samples, combined text only) found chars/token ~3.12 -- this run
adds the per-field breakdown (name vs category vs attributes) at 10x the
sample size, since which FIELD eats the token budget matters more than the
combined number for deciding what to truncate.

Deliberately tiny and CPU-only (no GPU/TPU quota spent) and fails fast: the
tokenizer load (the one thing likely to break, e.g. the transformers/
tokenizers version-mismatch gotcha hit on the TPU image during the FRIDA
tokenization-cache work) happens before touching any data.
"""
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "polars"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==5.15.0", "tokenizers==0.22.2"])

import json
import statistics

import polars as pl
from transformers import AutoTokenizer

ITEMS_HUMAN_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

FRIDA_MODEL_NAME = "ai-forever/FRIDA"
FRIDA_PREFIX = "paraphrase: "

MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, byte-matches CLAUDE.md

N_SAMPLE = 10_000
SEED = 42

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS_PATH = os.path.join(OUT_DIR, "frida_char_token_ratio_results.json")


def _quantile(values, q):
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def token_length_stats(lengths):
    lengths = sorted(lengths)
    return {
        "n": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p50": _quantile(lengths, 50),
        "p70": _quantile(lengths, 70),
        "p80": _quantile(lengths, 80),
        "p90": _quantile(lengths, 90),
        "p99": _quantile(lengths, 99),
        "min": min(lengths),
        "max": max(lengths),
    }


def main():
    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}", flush=True)

    print(f"[TOKENIZER] Loading {FRIDA_MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(FRIDA_MODEL_NAME)
    print(f"[TOKENIZER] vocab_size={tokenizer.vocab_size} prefix={FRIDA_PREFIX!r}", flush=True)
    # Fail fast on a trivial canary before touching any real data.
    canary = tokenizer(FRIDA_PREFIX + "canary", add_special_tokens=True)
    print(f"[TOKENIZER] canary ok, {len(canary['input_ids'])} tokens", flush=True)

    print("[DATA] loading items_human.parquet, sampling ...", flush=True)
    items = (
        pl.scan_parquet(ITEMS_HUMAN_PATH)
        .select(["id", "name", "category", "attributes"])
        .with_columns([
            pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS),
            pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_CATEGORY_CHARS),
            pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS),
        ])
        .with_columns(
            (pl.lit("Name: ") + pl.col("name") + pl.lit(" Category: ") + pl.col("category") +
             pl.lit(" Attributes: ") + pl.col("attributes")).str.slice(0, MAX_ITEM_CHARS).alias("text")
        )
        .collect()
    )
    n_total = items.height
    sample = items.sample(n=min(N_SAMPLE, n_total), seed=SEED)
    print(f"[DATA] items_human total={n_total:,}, sampled={sample.height:,}", flush=True)

    results = {
        "model": FRIDA_MODEL_NAME,
        "prefix": FRIDA_PREFIX,
        "n_sample": sample.height,
        "n_items_total_in_items_human": n_total,
        "fields": {},
    }

    # --- per-field, standalone (no prefix, no concatenation) -------------
    for field, max_chars in (("name", MAX_NAME_CHARS), ("category", MAX_CATEGORY_CHARS),
                              ("attributes", MAX_ATTR_CHARS)):
        print(f"[TOKENIZE] field={field} (max_chars={max_chars}) ...", flush=True)
        field_texts = sample.get_column(field).to_list()
        enc = tokenizer(field_texts, add_special_tokens=True)
        n_tokens = [len(ids) for ids in enc["input_ids"]]
        n_chars = [len(t) for t in field_texts]
        total_chars, total_tokens = sum(n_chars), sum(n_tokens)

        field_result = {
            "max_chars": max_chars,
            "tokens": token_length_stats(n_tokens),
            "chars": token_length_stats(n_chars),
            "chars_per_token_pooled": total_chars / total_tokens if total_tokens else None,
            "pct_at_or_above_max_chars": sum(1 for c in n_chars if c >= max_chars) / len(n_chars) * 100,
        }
        results["fields"][field] = field_result
        t = field_result["tokens"]
        print(f"[RESULT] {field}: tokens mean={t['mean']:.1f} median={t['median']:.0f} "
              f"p70={t['p70']:.0f} p80={t['p80']:.0f} p90={t['p90']:.0f} p99={t['p99']:.0f} "
              f"max={t['max']} | chars/token={field_result['chars_per_token_pooled']:.3f}", flush=True)

    # --- combined text (Name:/Category:/Attributes: + prefix), for
    # comparability with the v1 (1000-sample) run ---------------------
    print("[TOKENIZE] field=combined (prefix + full formula) ...", flush=True)
    combined_texts = [FRIDA_PREFIX + t for t in sample.get_column("text").to_list()]
    enc = tokenizer(combined_texts, add_special_tokens=True)
    n_tokens = [len(ids) for ids in enc["input_ids"]]
    n_chars = [len(t) for t in combined_texts]
    total_chars, total_tokens = sum(n_chars), sum(n_tokens)
    combined_result = {
        "tokens": token_length_stats(n_tokens),
        "chars": token_length_stats(n_chars),
        "chars_per_token_pooled": total_chars / total_tokens,
    }
    results["combined"] = combined_result
    t = combined_result["tokens"]
    print(f"[RESULT] combined: tokens mean={t['mean']:.1f} median={t['median']:.0f} "
          f"p70={t['p70']:.0f} p80={t['p80']:.0f} p90={t['p90']:.0f} p99={t['p99']:.0f} "
          f"max={t['max']} | chars/token={combined_result['chars_per_token_pooled']:.3f}", flush=True)

    print("=" * 80, flush=True)
    print("[SUMMARY]", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] results written to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
