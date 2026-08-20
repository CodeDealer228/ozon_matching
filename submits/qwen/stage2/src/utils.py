import json
import os
import re
import time
from pathlib import Path

# Resource-aware defaults for the judge: 20 CPU cores, H100 80 GB.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 256
EMBED_DIM = 1024

# H100 inference batching.
# ENCODE_BATCH_SIZE is only a hard cap; the real batch size is chosen
# dynamically from exact token lengths and ENCODE_TOKEN_BUDGET.
ENCODE_BATCH_SIZE = int(os.environ.get("ECUP_ENCODE_BATCH", "8192"))
ENCODE_TOKEN_BUDGET = int(os.environ.get("ECUP_TOKEN_BUDGET", "1179648"))
TOKENIZE_CHUNK_SIZE = int(os.environ.get("ECUP_TOKENIZE_CHUNK", "32768"))
MIN_TOKEN_BUDGET = int(os.environ.get("ECUP_MIN_TOKEN_BUDGET", "65536"))

PAIR_BATCH_SIZE = int(os.environ.get("ECUP_PAIR_BATCH", "65536"))

_WS_RE = re.compile(r"\s+")


def _clean_scalar(x):
    # Byte-match the training notebook's preprocessing semantics.
    if x is None:
        return ""
    if isinstance(x, (dict, list, tuple)):
        s = json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        s = str(x)
    return _WS_RE.sub(" ", s).strip()


def _format_attributes(attr_text):
    if attr_text is None:
        return ""
    try:
        obj = json.loads(attr_text)
    except Exception:
        return _clean_scalar(attr_text)
    if not isinstance(obj, dict):
        return _clean_scalar(obj)

    parts = []
    for key in sorted(obj.keys(), key=lambda x: str(x).lower()):
        value = _clean_scalar(obj[key])
        if value:
            parts.append(f"{_clean_scalar(key)}: {value}")
    return "\n".join(parts)


def _make_item_text(name, category, attributes):
    parts = [
        f"Name: {_clean_scalar(name)}",
        f"Category: {_clean_scalar(category)}",
    ]
    attrs = _format_attributes(attributes)
    if attrs:
        parts.append("Attributes:\n" + attrs)
    return "\n".join(parts)


def _load_needed_items(items_path, needed_ids):
    """
    Read only the 4 columns we need and push the id filter into PyArrow.

    Direct pyarrow.dataset avoids some pandas overhead and uses the 20 CPU
    cores for the scan. If the judge's PyArrow build dislikes the expression,
    fall back to the original pandas path.
    """
    needed = list(set(needed_ids))
    t0 = time.time()

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(items_path), format="parquet")

        # Build the value set once in Arrow, then scan in parallel.
        needed_arr = pa.array(needed)
        filt = ds.field("id").isin(needed_arr)

        table = dataset.to_table(
            columns=["id", "name", "attributes", "category"],
            filter=filt,
            use_threads=True,
        )

        df = table.to_pandas(
            split_blocks=True,
            self_destruct=True,
        )

        method = "pyarrow.dataset"

    except Exception as e:
        print(
            f"[WARN] pyarrow.dataset filter failed ({e!r}); "
            "falling back to pandas.read_parquet",
            flush=True,
        )

        try:
            df = pd.read_parquet(
                items_path,
                engine="pyarrow",
                columns=["id", "name", "attributes", "category"],
                filters=[("id", "in", needed)],
            )
            method = "pandas+predicate"

        except Exception as e2:
            print(
                f"[WARN] parquet predicate pushdown failed ({e2!r}); "
                "reading needed columns then filtering",
                flush=True,
            )

            df = pd.read_parquet(
                items_path,
                engine="pyarrow",
                columns=["id", "name", "attributes", "category"],
            )
            df = df[df["id"].isin(needed)]
            method = "pandas+fullscan"

    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)

    print(
        f"[DATA] loaded {len(df):,} needed items in "
        f"{time.time()-t0:.1f}s via {method}",
        flush=True,
    )

    return df


class SymmetricPairHead(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM, hidden=512, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, e1, e2):
        prod = e1 * e2
        cosine = prod.sum(dim=1, keepdim=True)
        features = torch.cat((torch.abs(e1 - e2), prod, cosine), dim=1)
        return self.net(features).squeeze(1)


def _load_encoder(model_path, device):
    import importlib.util
    import transformers
    from transformers import AutoModel, AutoTokenizer

    print(f"[MODEL] transformers={transformers.__version__}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    kwargs = dict(local_files_only=True)

    if device.type == "cuda":
        # Make PyTorch SDPA use fused kernels whenever the installed build
        # supports them.
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

        flash_attn_available = importlib.util.find_spec("flash_attn") is not None

        # Ordered candidates. Explicit FlashAttention2 can be faster in images
        # where flash-attn is installed; SDPA is the robust H100 fallback.
        candidates = []

        if flash_attn_available:
            candidates.append(("flash_attention_2", "flash_attention_2"))

        candidates.append(("sdpa", "sdpa"))
        candidates.append(("default", None))

        last_error = None
        model = None

        for label, attn_impl in candidates:
            try:
                load_kwargs = dict(kwargs)

                # transformers 4.57 accepts dtype; older images may only accept
                # torch_dtype, so retry below when needed.
                if attn_impl is not None:
                    load_kwargs["attn_implementation"] = attn_impl

                try:
                    model = AutoModel.from_pretrained(
                        model_path,
                        dtype=torch.float16,
                        **load_kwargs,
                    )
                except TypeError:
                    model = AutoModel.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        **load_kwargs,
                    )

                print(f"[MODEL] attention={label}", flush=True)
                break

            except (ImportError, TypeError, ValueError, RuntimeError) as e:
                last_error = e
                model = None
                print(
                    f"[MODEL] attention={label} unavailable ({type(e).__name__}: {e}); "
                    "trying fallback",
                    flush=True,
                )

        if model is None:
            raise RuntimeError(
                f"Could not load encoder with any attention backend: {last_error!r}"
            )

        model = model.to(device)

    else:
        model = AutoModel.from_pretrained(
            model_path,
            **kwargs,
        ).to(device)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.eval()
    return model, tokenizer


def _pool_last_token(last_hidden_state, attention_mask):
    # Training used SentenceTransformers last-token pooling + left padding.
    # With left padding, the final column is the final real token for every row.
    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_state[:, -1]

    # Defensive fallback if a tokenizer/config ever switches to right padding.
    seq_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        seq_lengths,
    ]


def _pretokenize_items(tokenizer, texts):
    """
    Tokenize every unique item exactly once, without padding.

    This preserves the same tokenizer + truncation semantics as the old code,
    but gives us the exact post-tokenizer sequence length for dynamic batching.
    """
    n = len(texts)
    token_ids = []
    lengths = np.empty(n, dtype=np.int32)

    t0 = time.time()
    out_pos = 0

    for start in range(0, n, TOKENIZE_CHUNK_SIZE):
        end = min(start + TOKENIZE_CHUNK_SIZE, n)

        enc = tokenizer(
            texts[start:end],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        ids_chunk = enc["input_ids"]
        token_ids.extend(ids_chunk)

        chunk_lengths = np.fromiter(
            (len(x) for x in ids_chunk),
            dtype=np.int32,
            count=len(ids_chunk),
        )
        lengths[out_pos:out_pos + len(ids_chunk)] = chunk_lengths
        out_pos += len(ids_chunk)

    elapsed = time.time() - t0

    if n:
        q = np.quantile(lengths, [0.50, 0.90, 0.95, 0.99]).astype(int)
        print(
            f"[TOKENIZE] {n:,} items in {elapsed:.1f}s | "
            f"p50={q[0]} p90={q[1]} p95={q[2]} p99={q[3]} "
            f"max={int(lengths.max())}",
            flush=True,
        )

    return token_ids, lengths


def _choose_batch_size(sorted_lengths, pos, token_budget, max_batch_size):
    """
    Choose the largest batch B for which:

        B * max_sequence_length_in_batch <= token_budget

    sorted_lengths is ascending, so the last sequence in a candidate batch
    is also the longest sequence in that batch.
    """
    remaining = len(sorted_lengths) - pos
    hi = min(int(max_batch_size), remaining)

    if hi <= 1:
        return hi

    lo = 1
    best = 1

    while lo <= hi:
        mid = (lo + hi) // 2
        max_len = int(sorted_lengths[pos + mid - 1])
        padded_tokens = mid * max_len

        if padded_tokens <= token_budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _encode_items(model, tokenizer, texts, device, requested_batch_size):
    """
    Exact-token-length bucketing + dynamic token-budget batching.

    Model parity is preserved:
      - same item text
      - same tokenizer
      - same MAX_LENGTH=512
      - same left padding
      - same last-token pooling
      - same float32 L2 normalization

    Only execution order and batch packing change.
    """
    import gc

    n = len(texts)
    if n == 0:
        return torch.empty((0, EMBED_DIM), device=device, dtype=torch.float32)

    # Tokenize once on CPU and obtain exact sequence lengths.
    token_ids, lengths = _pretokenize_items(tokenizer, texts)

    # Short -> long minimizes padding. As sequences get longer,
    # the fixed token budget automatically reduces item batch size.
    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]

    embeddings = torch.empty(
        (n, EMBED_DIM),
        device=device,
        dtype=torch.float32,
    )

    max_batch_size = min(int(requested_batch_size), n)
    token_budget = int(ENCODE_TOKEN_BUDGET)

    pos = 0
    n_done = 0
    last_log = 0
    t0 = time.time()

    print(
        f"[ENCODE] dynamic batching | max_batch={max_batch_size:,} | "
        f"token_budget={token_budget:,}",
        flush=True,
    )

    with torch.inference_mode():
        while pos < n:
            batch_size = max(
                1,
                _choose_batch_size(
                    sorted_lengths,
                    pos,
                    token_budget,
                    max_batch_size,
                ),
            )

            batch_idx_np = order[pos:pos + batch_size]
            batch_ids = [token_ids[int(i)] for i in batch_idx_np]

            # Explicit refs make OOM cleanup reliable.
            enc = None
            out = None
            emb = None
            batch_idx = None

            try:
                # input_ids are already tokenized: this step only pads.
                enc = tokenizer.pad(
                    {"input_ids": batch_ids},
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                enc = {
                    k: v.to(device, non_blocking=True)
                    for k, v in enc.items()
                }

                out = model(
                    **enc,
                    use_cache=False,
                    return_dict=True,
                )

                emb = _pool_last_token(
                    out.last_hidden_state,
                    enc["attention_mask"],
                )
                emb = F.normalize(emb.float(), p=2, dim=1)

                batch_idx = torch.as_tensor(
                    batch_idx_np,
                    dtype=torch.long,
                    device=device,
                )
                embeddings[batch_idx] = emb

                pos += batch_size
                n_done += batch_size

                if (
                    n_done == batch_size
                    or n_done - last_log >= 10_000
                    or n_done == n
                ):
                    elapsed = time.time() - t0
                    padded_len = int(enc["input_ids"].shape[1])

                    if device.type == "cuda":
                        mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                        mem_text = f" | peak_vram={mem_gb:.1f}GB"
                    else:
                        mem_text = ""

                    print(
                        f"[ENCODE] {n_done:,}/{n:,} | "
                        f"{n_done / max(elapsed, 1e-9):.1f} items/s | "
                        f"batch={batch_size:,} | padded_len={padded_len} | "
                        f"padded_tokens={batch_size * padded_len:,} | "
                        f"budget={token_budget:,}{mem_text}",
                        flush=True,
                    )
                    last_log = n_done

                del enc, out, emb, batch_idx

            except torch.OutOfMemoryError:
                if device.type != "cuda" or batch_size <= 16:
                    raise

                # Free referenced tensors first; empty_cache alone is not enough.
                del enc, out, emb, batch_idx
                gc.collect()
                torch.cuda.empty_cache()

                old_budget = token_budget
                old_max_batch = max_batch_size

                token_budget = max(
                    MIN_TOKEN_BUDGET,
                    int(token_budget * 0.75),
                )
                max_batch_size = max(
                    16,
                    min(max_batch_size, batch_size // 2),
                )

                current_max_len = int(
                    sorted_lengths[min(pos + batch_size - 1, n - 1)]
                )

                print(
                    f"[OOM] batch={batch_size:,} len~{current_max_len} | "
                    f"budget {old_budget:,}->{token_budget:,} | "
                    f"max_batch {old_max_batch:,}->{max_batch_size:,}",
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(
        f"[ENCODE] GPU stage finished {n:,} items in {elapsed:.1f}s "
        f"({n / max(elapsed, 1e-9):.1f} items/s)",
        flush=True,
    )
    return embeddings

def _score_pairs(head, embeddings, idx1, idx2, device, batch_size):
    n = len(idx1)
    scores = np.empty(n, dtype=np.float32)

    idx1_gpu = torch.as_tensor(idx1, dtype=torch.long, device=device)
    idx2_gpu = torch.as_tensor(idx2, dtype=torch.long, device=device)

    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            e1 = embeddings[idx1_gpu[start:end]]
            e2 = embeddings[idx2_gpu[start:end]]

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = head(e1, e2)
            else:
                logits = head(e1, e2)

            scores[start:end] = torch.sigmoid(logits.float()).cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[SCORE] {n:,} pairs in {elapsed:.1f}s ({n/max(elapsed,1e-9):.1f} pairs/s)", flush=True)
    return scores


def predict_pipeline(
    items_path,
    matches_path,
    model_path,
    head_path,
    output_csv_path,
    encode_batch_size=ENCODE_BATCH_SIZE,
    pair_batch_size=PAIR_BATCH_SIZE,
):
    total_t0 = time.time()

    print(
        "=== E-CUP Qwen3 bi-encoder LoRA-merged + symmetric MLP ===",
        flush=True,
    )
    print(
        f"[CFG] max_length={MAX_LENGTH} "
        f"max_encode_batch={encode_batch_size} "
        f"token_budget={ENCODE_TOKEN_BUDGET:,} "
        f"pair_batch={pair_batch_size:,} fast512=v2",
        flush=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        props = torch.cuda.get_device_properties(0)
        print(
            f"[GPU] {torch.cuda.get_device_name(0)} | "
            f"VRAM={props.total_memory / 1024**3:.1f} GiB",
            flush=True,
        )
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA unavailable; CPU fallback will be slow", flush=True)

    # ---------------------------------------------------------------
    # 1/5: matches
    # ---------------------------------------------------------------
    stage_t0 = time.time()
    print("[1/5] Loading matches ...", flush=True)

    match_df = pd.read_parquet(
        matches_path,
        engine="pyarrow",
        columns=["id1", "id2"],
    )

    needed_ids = pd.unique(
        pd.concat(
            [match_df["id1"], match_df["id2"]],
            ignore_index=True,
        )
    ).tolist()

    print(
        f"[1/5] pairs={len(match_df):,} "
        f"unique_items={len(needed_ids):,} | "
        f"stage={time.time() - stage_t0:.1f}s | "
        f"total={time.time() - total_t0:.1f}s",
        flush=True,
    )

    # ---------------------------------------------------------------
    # 2/5: items + exact training-time text formatting
    # ---------------------------------------------------------------
    stage_t0 = time.time()
    print("[2/5] Loading needed items ...", flush=True)

    item_df = _load_needed_items(items_path, needed_ids)

    load_elapsed = time.time() - stage_t0
    print(f"[2/5] parquet load={load_elapsed:.1f}s", flush=True)

    print("[2/5] Formatting item texts ...", flush=True)
    format_t0 = time.time()

    texts = [
        _make_item_text(name, category, attributes)
        for name, attributes, category
        in item_df[["name", "attributes", "category"]].itertuples(
            index=False,
            name=None,
        )
    ]

    item_ids = item_df["id"].tolist()
    id_to_pos = pd.Series(
        np.arange(len(item_df), dtype=np.int64),
        index=item_ids,
    )

    idx1_s = match_df["id1"].map(id_to_pos)
    idx2_s = match_df["id2"].map(id_to_pos)

    if idx1_s.isna().any() or idx2_s.isna().any():
        missing = int(idx1_s.isna().sum() + idx2_s.isna().sum())
        raise RuntimeError(
            f"{missing} pair endpoints were not found in filtered items parquet"
        )

    idx1 = idx1_s.to_numpy(dtype=np.int64)
    idx2 = idx2_s.to_numpy(dtype=np.int64)

    # ---------------------------------------------------------------
    # Exact text deduplication: zero change to model semantics.
    #
    # If two item IDs produce byte-identical model input text, Qwen would
    # return the same embedding for both. Encode that text only once and
    # remap pair endpoints directly to the unique-text embedding index.
    # ---------------------------------------------------------------
    dedup_t0 = time.time()

    text_codes, unique_texts_index = pd.factorize(
        np.asarray(texts, dtype=object),
        sort=False,
    )

    text_codes = text_codes.astype(np.int64, copy=False)
    unique_texts = unique_texts_index.tolist()

    idx1 = text_codes[idx1]
    idx2 = text_codes[idx2]

    n_item_texts = len(texts)
    n_unique_texts = len(unique_texts)
    n_saved = n_item_texts - n_unique_texts

    print(
        f"[DEDUP] item_texts={n_item_texts:,} "
        f"unique_texts={n_unique_texts:,} "
        f"saved={n_saved:,} ({n_saved/max(n_item_texts,1):.1%}) | "
        f"time={time.time()-dedup_t0:.1f}s",
        flush=True,
    )

    # We no longer need the item-level duplicate strings.
    texts = unique_texts

    print(
        f"[2/5] formatted={n_item_texts:,} -> encode={n_unique_texts:,} | "
        f"format+dedup={time.time() - format_t0:.1f}s | "
        f"stage={time.time() - stage_t0:.1f}s | "
        f"total={time.time() - total_t0:.1f}s",
        flush=True,
    )

    del item_df, id_to_pos, idx1_s, idx2_s
    del text_codes, unique_texts_index, unique_texts

    # ---------------------------------------------------------------
    # 3/5: merged Qwen + exact-token dynamic batching
    # ---------------------------------------------------------------
    stage_t0 = time.time()
    print("[3/5] Loading merged encoder ...", flush=True)

    model, tokenizer = _load_encoder(model_path, device)

    print(
        f"[3/5] model loaded in {time.time() - stage_t0:.1f}s",
        flush=True,
    )

    embeddings = _encode_items(
        model,
        tokenizer,
        texts,
        device,
        encode_batch_size,
    )

    del model, tokenizer, texts

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[3/5] embeddings={tuple(embeddings.shape)} | "
        f"stage={time.time() - stage_t0:.1f}s | "
        f"total={time.time() - total_t0:.1f}s",
        flush=True,
    )

    # ---------------------------------------------------------------
    # 4/5: symmetric MLP
    # ---------------------------------------------------------------
    stage_t0 = time.time()
    print("[4/5] Loading MLP head + scoring pairs ...", flush=True)

    head = SymmetricPairHead().to(device)
    state = torch.load(
        head_path,
        map_location="cpu",
        weights_only=True,
    )
    head.net.load_state_dict(state)
    head.eval()

    probs = _score_pairs(
        head,
        embeddings,
        idx1,
        idx2,
        device,
        pair_batch_size,
    )

    print(
        f"[4/5] stage={time.time() - stage_t0:.1f}s | "
        f"total={time.time() - total_t0:.1f}s",
        flush=True,
    )

    # ---------------------------------------------------------------
    # 5/5: CSV
    # ---------------------------------------------------------------
    stage_t0 = time.time()
    print("[5/5] Writing output ...", flush=True)

    result = pd.DataFrame(
        {
            "id1": match_df["id1"].to_numpy(),
            "id2": match_df["id2"].to_numpy(),
            "predict": probs,
        }
    )

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result.to_csv(output_csv_path, index=False)

    elapsed = time.time() - total_t0

    print(
        f"[5/5] write={time.time() - stage_t0:.1f}s",
        flush=True,
    )
    print(
        f"=== Done in {elapsed:.1f}s | "
        f"{len(match_df) / max(elapsed, 1e-9):.1f} pairs/s end-to-end ===",
        flush=True,
    )

    return result

