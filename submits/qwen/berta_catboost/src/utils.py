import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time

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


QWEN_MAX_LENGTH = int(os.environ.get("ECUP_QWEN_MAX_LENGTH", "256"))
QWEN_ENCODE_BATCH_SIZE = int(os.environ.get("ECUP_QWEN_ENCODE_BATCH", "8192"))
QWEN_TOKEN_BUDGET = int(os.environ.get("ECUP_QWEN_TOKEN_BUDGET", "1179648"))
QWEN_TOKENIZE_CHUNK = int(os.environ.get("ECUP_QWEN_TOKENIZE_CHUNK", "32768"))
QWEN_MIN_TOKEN_BUDGET = int(os.environ.get("ECUP_QWEN_MIN_TOKEN_BUDGET", "65536"))

BERTA_PREFIX = "paraphrase: "
BERTA_ITEM_MAX_LENGTH = int(os.environ.get("ECUP_BERTA_ITEM_MAX_LENGTH", "128"))
BERTA_PAIR_MAX_LENGTH = int(os.environ.get("ECUP_BERTA_PAIR_MAX_LENGTH", "128"))
BERTA_BATCH_SIZE = int(os.environ.get("ECUP_BERTA_BATCH", "2048"))
BERTA_TOKENIZE_CHUNK = int(os.environ.get("ECUP_BERTA_TOKENIZE_CHUNK", "32768"))

CATBOOST_BATCH_SIZE = int(os.environ.get("ECUP_CATBOOST_BATCH", "32768"))

MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

_WS_RE = re.compile(r"\s+")


def _elapsed(t0):
    return f"{time.time() - t0:.1f}s"


def _device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda:0")
    return torch.device("cpu")


def _maybe_offline_install(wheels_dir, package_names, target_name):
    missing = []
    for pkg, import_name in package_names:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return

    if not os.path.isdir(wheels_dir):
        print(f"[SETUP] wheels dir missing: {wheels_dir}", flush=True)
        return

    target_dir = os.path.join(tempfile.gettempdir(), target_name)
    os.makedirs(target_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "install",
        "-q", "--no-index", "--no-cache-dir", "--no-deps",
        "--find-links", wheels_dir, "--target", target_dir,
        *missing,
    ]
    print(f"[SETUP] offline installing {missing} into {target_dir}", flush=True)
    subprocess.check_call(cmd)
    sys.path.append(target_dir)


def _install_peft_if_needed(wheels_dir):
    packages = [
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("huggingface_hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
        ("packaging", "packaging"),
        ("tqdm", "tqdm"),
        ("filelock", "filelock"),
        ("fsspec", "fsspec"),
        ("typing_extensions", "typing_extensions"),
        ("psutil", "psutil"),
        ("pyyaml", "yaml"),
    ]
    _maybe_offline_install(wheels_dir, packages, "qwen_berta_peft_pkgs")


def _install_catboost_if_needed(wheels_dir):
    packages = [
        ("catboost", "catboost"),
        ("scipy", "scipy"),
        ("graphviz", "graphviz"),
        ("six", "six"),
        ("plotly", "plotly"),
    ]
    _maybe_offline_install(wheels_dir, packages, "qwen_berta_catboost_pkgs")


def _clean_scalar(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, (dict, list, tuple)):
        s = json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        s = str(x)
    return _WS_RE.sub(" ", s).strip()


def _format_qwen_attributes(attr_text):
    if attr_text is None:
        return ""
    if isinstance(attr_text, float) and pd.isna(attr_text):
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


def _make_qwen_text(name, category, attributes):
    parts = [
        f"Name: {_clean_scalar(name)}",
        f"Category: {_clean_scalar(category)}",
    ]
    attrs = _format_qwen_attributes(attributes)
    if attrs:
        parts.append("Attributes:\n" + attrs)
    return "\n".join(parts)


def _make_berta_text(name, category, attributes):
    name = _clean_scalar(name)[:MAX_NAME_CHARS]
    category = _clean_scalar(category)[:MAX_CATEGORY_CHARS]
    attributes = _clean_scalar(attributes)[:MAX_ATTR_CHARS]
    text = f"Name: {name} Category: {category} Attributes: {attributes}"
    return text[:MAX_ITEM_CHARS]


def _load_needed_items(items_path, needed_ids):
    needed = list(pd.unique(pd.Series(needed_ids)))
    t0 = time.time()
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(items_path), format="parquet")
        table = dataset.to_table(
            columns=["id", "name", "attributes", "category"],
            filter=ds.field("id").isin(pa.array(needed)),
            use_threads=True,
        )
        df = table.to_pandas(split_blocks=True, self_destruct=True)
        method = "pyarrow.dataset"
    except Exception as e:
        print(f"[WARN] pyarrow.dataset filter failed ({e!r}); using pandas", flush=True)
        df = pd.read_parquet(
            items_path,
            engine="pyarrow",
            columns=["id", "name", "attributes", "category"],
            filters=[("id", "in", needed)],
        )
        method = "pandas+filters"

    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)
    print(f"[DATA] loaded {len(df):,} items via {method} in {_elapsed(t0)}", flush=True)
    return df


def _load_qwen_encoder(model_path, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"

    kwargs = {"local_files_only": True}
    if device.type == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        try:
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )
        except (TypeError, ValueError):
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, **kwargs)
    else:
        model = AutoModel.from_pretrained(model_path, **kwargs)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()
    return model, tokenizer


def _pool_qwen_last_token(last_hidden_state, attention_mask):
    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_state[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        seq_lengths,
    ]


def _pretokenize_qwen(tokenizer, texts):
    n = len(texts)
    token_ids = []
    lengths = np.empty(n, dtype=np.int32)
    out_pos = 0
    t0 = time.time()

    for start in range(0, n, QWEN_TOKENIZE_CHUNK):
        end = min(start + QWEN_TOKENIZE_CHUNK, n)
        enc = tokenizer(
            texts[start:end],
            padding=False,
            truncation=True,
            max_length=QWEN_MAX_LENGTH,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids_chunk = enc["input_ids"]
        token_ids.extend(ids_chunk)
        chunk_lengths = np.fromiter((len(x) for x in ids_chunk), dtype=np.int32, count=len(ids_chunk))
        lengths[out_pos:out_pos + len(ids_chunk)] = chunk_lengths
        out_pos += len(ids_chunk)

    if n:
        q = np.quantile(lengths, [0.50, 0.90, 0.99]).astype(int)
        print(
            f"[QWEN] tokenized {n:,} unique texts in {_elapsed(t0)} | "
            f"p50={q[0]} p90={q[1]} p99={q[2]} max={int(lengths.max())}",
            flush=True,
        )
    return token_ids, lengths


def _choose_qwen_batch(sorted_lengths, pos, token_budget, max_batch_size):
    remaining = len(sorted_lengths) - pos
    hi = min(int(max_batch_size), remaining)
    if hi <= 1:
        return hi

    lo = 1
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        padded_tokens = mid * int(sorted_lengths[pos + mid - 1])
        if padded_tokens <= token_budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _encode_qwen_texts(model, tokenizer, texts, device):
    n = len(texts)
    hidden_size = int(getattr(model.config, "hidden_size", 1024))
    if n == 0:
        return np.empty((0, hidden_size), dtype=np.float16)

    token_ids, lengths = _pretokenize_qwen(tokenizer, texts)
    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]
    embeddings = np.empty((n, hidden_size), dtype=np.float16)

    pos = 0
    done = 0
    last_log = 0
    token_budget = QWEN_TOKEN_BUDGET
    max_batch = min(QWEN_ENCODE_BATCH_SIZE, n)
    t0 = time.time()

    print(
        f"[QWEN] encoding | max_batch={max_batch:,} token_budget={token_budget:,} "
        f"max_length={QWEN_MAX_LENGTH}",
        flush=True,
    )

    with torch.inference_mode():
        while pos < n:
            batch_size = max(1, _choose_qwen_batch(sorted_lengths, pos, token_budget, max_batch))
            batch_idx_np = order[pos:pos + batch_size]
            batch_ids = [token_ids[int(i)] for i in batch_idx_np]
            try:
                enc = tokenizer.pad(
                    {"input_ids": batch_ids},
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                out = model(**enc, use_cache=False, return_dict=True)
                emb = _pool_qwen_last_token(out.last_hidden_state, enc["attention_mask"])
                emb = F.normalize(emb.float(), p=2, dim=1)
                embeddings[batch_idx_np] = emb.cpu().numpy().astype(np.float16)

                pos += batch_size
                done += batch_size
                if done == batch_size or done - last_log >= 10_000 or done == n:
                    print(
                        f"[QWEN] {done:,}/{n:,} | "
                        f"{done / max(time.time() - t0, 1e-9):.1f} texts/s | "
                        f"batch={batch_size:,} padded_len={int(enc['input_ids'].shape[1])}",
                        flush=True,
                    )
                    last_log = done
                del enc, out, emb
            except torch.OutOfMemoryError:
                if device.type != "cuda" or batch_size <= 16:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                token_budget = max(QWEN_MIN_TOKEN_BUDGET, int(token_budget * 0.75))
                max_batch = max(16, min(max_batch, batch_size // 2))
                print(f"[QWEN][OOM] new token_budget={token_budget:,} max_batch={max_batch:,}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[QWEN] finished in {_elapsed(t0)}", flush=True)
    return embeddings


def _build_qwen_features(item_df, match_df, qwen_model_path, device):
    t0 = time.time()
    qwen_texts = [
        _make_qwen_text(name, category, attributes)
        for name, attributes, category in item_df[["name", "attributes", "category"]].itertuples(index=False, name=None)
    ]

    unique_texts, item_to_text_idx = np.unique(np.asarray(qwen_texts, dtype=object), return_inverse=True)
    print(
        f"[QWEN] item texts={len(qwen_texts):,} unique_texts={len(unique_texts):,}",
        flush=True,
    )

    id_to_item_pos = pd.Series(np.arange(len(item_df), dtype=np.int64), index=item_df["id"])
    item_pos1 = match_df["id1"].map(id_to_item_pos)
    item_pos2 = match_df["id2"].map(id_to_item_pos)
    if item_pos1.isna().any() or item_pos2.isna().any():
        raise RuntimeError("Some pair ids were not found in items parquet")

    qwen_idx1 = item_to_text_idx[item_pos1.to_numpy(dtype=np.int64)]
    qwen_idx2 = item_to_text_idx[item_pos2.to_numpy(dtype=np.int64)]

    model, tokenizer = _load_qwen_encoder(qwen_model_path, device)
    qwen_embeddings = _encode_qwen_texts(model, tokenizer, unique_texts.tolist(), device)

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[QWEN] stage done in {_elapsed(t0)}", flush=True)
    return qwen_embeddings, qwen_idx1, qwen_idx2


def _find_token_pattern_starts(arr, pattern):
    pattern = np.asarray(pattern, dtype=arr.dtype)
    plen = len(pattern)
    n_positions = arr.shape[1] - plen + 1
    if n_positions <= 0:
        return np.full(arr.shape[0], -1, dtype=np.int32)
    match = np.ones((arr.shape[0], n_positions), dtype=bool)
    for offset, tok in enumerate(pattern):
        match &= arr[:, offset:offset + n_positions] == tok
    rows, cols = np.where(match)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def _compact_berta_ids(input_ids, item_length, category_start, attr_start):
    real = input_ids[:item_length]
    if category_start < 0 or attr_start < 0 or attr_start <= category_start:
        return real.astype(np.int32, copy=True)
    return np.concatenate((input_ids[:category_start], input_ids[attr_start:item_length])).astype(np.int32, copy=False)


def _tokenize_berta_items(tokenizer, item_df):
    t0 = time.time()
    id_to_compact = {}
    ids = item_df["id"].tolist()
    texts = [
        BERTA_PREFIX + _make_berta_text(name, category, attributes)
        for name, attributes, category in item_df[["name", "attributes", "category"]].itertuples(index=False, name=None)
    ]

    for start in range(0, len(texts), BERTA_TOKENIZE_CHUNK):
        end = min(start + BERTA_TOKENIZE_CHUNK, len(texts))
        enc = tokenizer(
            texts[start:end],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=BERTA_ITEM_MAX_LENGTH,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int32, copy=False)
        lengths = enc["attention_mask"].sum(axis=1).astype(np.int32)
        cat_starts = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_PATTERN)
        attr_starts = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_PATTERN)
        for row, item_id in enumerate(ids[start:end]):
            id_to_compact[item_id] = _compact_berta_ids(
                input_ids[row],
                int(lengths[row]),
                int(cat_starts[row]),
                int(attr_starts[row]),
            )

    print(f"[BERTA] tokenized {len(id_to_compact):,} items in {_elapsed(t0)}", flush=True)
    return id_to_compact


def _allocate_pair_budget(len1, len2, budget):
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def _build_berta_pair_arrays(batch_df, id_to_compact, cls_id, sep_id, pad_id):
    n = len(batch_df)
    input_ids = np.full((n, BERTA_PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    attention_mask = np.zeros((n, BERTA_PAIR_MAX_LENGTH), dtype=np.int64)
    token_type_ids = np.zeros((n, BERTA_PAIR_MAX_LENGTH), dtype=np.int64)
    content_budget = BERTA_PAIR_MAX_LENGTH - 3

    id1_values = batch_df["id1"].to_numpy()
    id2_values = batch_df["id2"].to_numpy()
    missing = 0

    for i in range(n):
        c1 = id_to_compact.get(id1_values[i])
        c2 = id_to_compact.get(id2_values[i])
        if c1 is None or c2 is None:
            missing += 1
            continue
        len1, len2 = len(c1), len(c2)
        if len1 + len2 <= content_budget:
            keep1, keep2 = len1, len2
        else:
            keep1, keep2 = _allocate_pair_budget(len1, len2, content_budget)

        n1 = 1 + keep1 + 1
        n_total = n1 + keep2 + 1
        input_ids[i, 0] = cls_id
        input_ids[i, 1:1 + keep1] = c1[:keep1]
        input_ids[i, 1 + keep1] = sep_id
        input_ids[i, n1:n1 + keep2] = c2[:keep2]
        input_ids[i, n1 + keep2] = sep_id
        token_type_ids[i, n1:n_total] = 1
        attention_mask[i, :n_total] = 1

    if missing:
        print(f"[BERTA][WARN] {missing} pair endpoints missing in token cache", flush=True)
    return input_ids, attention_mask, token_type_ids


class BertaPairEmbedder(nn.Module):
    def __init__(self, encoder, classifier_head):
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            logits = self.classifier_head(pooled).view(-1)
            return pooled.float(), logits.float()


def _load_berta_model(base_path, adapter_path, classifier_path, wheels_dir, device):
    _install_peft_if_needed(wheels_dir)
    try:
        import peft.import_utils
        import peft.tuners.lora.torchao
        peft.import_utils.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(base_path, local_files_only=True)
    hidden_size = int(base_model.config.hidden_size)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    classifier_head = nn.Linear(hidden_size, 1)
    classifier_head.load_state_dict(torch.load(classifier_path, map_location="cpu"))

    model = BertaPairEmbedder(peft_model, classifier_head)
    model.to(device)
    model.eval()
    return model, tokenizer, hidden_size


def _encode_berta_pairs(item_df, match_df, base_path, adapter_path, classifier_path, wheels_dir, device):
    t0 = time.time()
    model, tokenizer, hidden_size = _load_berta_model(
        base_path, adapter_path, classifier_path, wheels_dir, device
    )
    id_to_compact = _tokenize_berta_items(tokenizer, item_df)

    n = len(match_df)
    embeddings = np.empty((n, hidden_size), dtype=np.float16)
    fallback_scores = np.empty(n, dtype=np.float32)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    print(
        f"[BERTA] encoding {n:,} pairs | batch={BERTA_BATCH_SIZE:,} "
        f"pair_max_length={BERTA_PAIR_MAX_LENGTH}",
        flush=True,
    )

    done = 0
    last_log = 0
    with torch.inference_mode():
        for start in range(0, n, BERTA_BATCH_SIZE):
            end = min(start + BERTA_BATCH_SIZE, n)
            input_ids_np, attention_mask_np, token_type_ids_np = _build_berta_pair_arrays(
                match_df.iloc[start:end],
                id_to_compact,
                cls_id,
                sep_id,
                pad_id,
            )
            input_ids = torch.from_numpy(input_ids_np).to(device, non_blocking=True)
            attention_mask = torch.from_numpy(attention_mask_np).to(device, non_blocking=True)
            token_type_ids = torch.from_numpy(token_type_ids_np).to(device, non_blocking=True)

            pooled, logits = model(input_ids, attention_mask, token_type_ids)
            embeddings[start:end] = pooled.cpu().numpy().astype(np.float16)
            fallback_scores[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

            done += end - start
            if done - last_log >= 10_000 or done == n:
                print(
                    f"[BERTA] {done:,}/{n:,} | "
                    f"{done / max(time.time() - t0, 1e-9):.1f} pairs/s",
                    flush=True,
                )
                last_log = done

            del input_ids_np, attention_mask_np, token_type_ids_np
            del input_ids, attention_mask, token_type_ids, pooled, logits

    if device.type == "cuda":
        torch.cuda.synchronize()
    del model, tokenizer, id_to_compact
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[BERTA] stage done in {_elapsed(t0)}", flush=True)
    return embeddings, fallback_scores


def _predict_catboost(catboost_model_path, wheels_dir, qwen_embeddings, qwen_idx1, qwen_idx2, berta_embeddings):
    _install_catboost_if_needed(wheels_dir)
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(catboost_model_path)

    n = len(qwen_idx1)
    out = np.empty(n, dtype=np.float32)
    q_dim = qwen_embeddings.shape[1]
    b_dim = berta_embeddings.shape[1]
    feature_dim = q_dim * 2 + b_dim

    print(
        f"[CATBOOST] predicting {n:,} rows | feature_dim={feature_dim} "
        f"batch={CATBOOST_BATCH_SIZE:,}",
        flush=True,
    )
    t0 = time.time()

    for start in range(0, n, CATBOOST_BATCH_SIZE):
        end = min(start + CATBOOST_BATCH_SIZE, n)
        m = end - start
        features = np.empty((m, feature_dim), dtype=np.float32)
        features[:, :q_dim] = qwen_embeddings[qwen_idx1[start:end]].astype(np.float32, copy=False)
        features[:, q_dim:q_dim * 2] = qwen_embeddings[qwen_idx2[start:end]].astype(np.float32, copy=False)
        features[:, q_dim * 2:] = berta_embeddings[start:end].astype(np.float32, copy=False)

        pred = model.predict_proba(features)
        if np.ndim(pred) == 2 and pred.shape[1] > 1:
            out[start:end] = pred[:, 1].astype(np.float32)
        else:
            out[start:end] = np.asarray(pred).reshape(-1).astype(np.float32)

    print(f"[CATBOOST] done in {_elapsed(t0)}", flush=True)
    return np.clip(out, 0.0, 1.0)


def predict_pipeline(
    items_path,
    matches_path,
    output_csv_path,
    qwen_model_path,
    berta_base_path,
    berta_adapter_path,
    berta_classifier_path,
    catboost_model_path,
    wheels_dir,
):
    total_t0 = time.time()
    print("=== E-CUP Qwen embeddings + BERTA LoRA pair embeddings + CatBoost ===", flush=True)
    print(
        f"[CFG] qwen_max_length={QWEN_MAX_LENGTH} "
        f"berta_item_max_length={BERTA_ITEM_MAX_LENGTH} "
        f"berta_pair_max_length={BERTA_PAIR_MAX_LENGTH}",
        flush=True,
    )

    device = _device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM={props.total_memory / 1024**3:.1f} GiB", flush=True)
    else:
        print("[WARN] CUDA unavailable; this will be slow", flush=True)

    print("[1/5] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path, engine="pyarrow", columns=["id1", "id2"])
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]], ignore_index=True)).tolist()
    print(f"[1/5] pairs={len(match_df):,} unique_items={len(needed_ids):,}", flush=True)

    print("[2/5] Loading needed items ...", flush=True)
    item_df = _load_needed_items(items_path, needed_ids)

    has_catboost = os.path.exists(catboost_model_path)

    if has_catboost:
        print("[3/5] Qwen item embeddings ...", flush=True)
        qwen_embeddings, qwen_idx1, qwen_idx2 = _build_qwen_features(
            item_df, match_df, qwen_model_path, device
        )
    else:
        print(
            f"[3/5] Skipping Qwen: CatBoost model not found at {catboost_model_path}",
            flush=True,
        )
        qwen_embeddings = None
        qwen_idx1 = None
        qwen_idx2 = None

    print("[4/5] BERTA pair embeddings ...", flush=True)
    berta_embeddings, berta_fallback_scores = _encode_berta_pairs(
        item_df,
        match_df,
        berta_base_path,
        berta_adapter_path,
        berta_classifier_path,
        wheels_dir,
        device,
    )

    print("[5/5] Final scoring ...", flush=True)
    if has_catboost:
        predictions = _predict_catboost(
            catboost_model_path,
            wheels_dir,
            qwen_embeddings,
            qwen_idx1,
            qwen_idx2,
            berta_embeddings,
        )
    else:
        print(
            f"[WARN] CatBoost model not found at {catboost_model_path}; "
            "using BERTA classifier fallback scores.",
            flush=True,
        )
        predictions = berta_fallback_scores

    results_df = pd.DataFrame({
        "id1": match_df["id1"].to_numpy(),
        "id2": match_df["id2"].to_numpy(),
        "predict": predictions.astype(np.float32),
    })

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    print(f"=== Done in {_elapsed(total_t0)} ===", flush=True)
    return results_df
