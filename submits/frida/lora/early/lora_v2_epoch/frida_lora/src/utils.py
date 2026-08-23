import os

# Hardware-aware env tuning, mirrored from submits/lora/src/utils.py -- 20 CPU
# cores / 200GB RAM / 1x H100 80GB at inference time.
os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("RAYON_NUM_THREADS", "32")

import sys
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, logging as hf_logging


def maybe_offline_install(wheels_dir):
    """peft (+ its own dependency closure) is absent from
    odsai/ecup26-matching-baseline:1.0 (confirmed for real on the actual
    H100 test stand, not just inferred from CLAUDE.md's guaranteed-package
    list: torch/transformers/numpy/pandas+pyarrow are there, peft is not).
    This submission deliberately does NOT merge the LoRA adapter into the
    base weights (explicit instruction), so peft is a real runtime
    dependency, unlike submits/lora which merges offline specifically to
    avoid this.

    Only the TOP-LEVEL packages need listing below -- pip's own dependency
    resolver walks the rest from whatever wheels are present in wheels_dir
    (that's what --find-links is for). But every wheel actually needed does
    have to be THERE: a real test-stand run caught huggingface_hub 1.27.0's
    full requires list (click, filelock, fsspec, hf-xet, httpx, packaging,
    pyyaml, tqdm, typing-extensions -- confirmed via `pip show
    huggingface_hub`), which pulls in httpx's own stack (httpcore, h11,
    anyio, idna, certifi) and, in this specific version, a typer/rich CLI
    cluster (annotated-doc, pygments, markdown-it-py, mdurl, shellingham) --
    none of which submits/lora or any earlier probe in this repo needed, so
    there was no precedent to anticipate it from. Deliberately NOT bundled:
    torch/numpy/transformers/tokenizers (already guaranteed present, and a
    bundled torch could conflict with the CUDA-matched build already in the
    image) and their own torch-only transitive deps (sympy, mpmath,
    networkx, jinja2, MarkupSafe, setuptools) -- not needed since this
    submission doesn't install its own torch.

    --no-index --find-links: fully offline, no PyPI reachability assumed at
    judging time.

    --target /tmp/... + manual sys.path insert, NOT a normal (site-packages)
    install -- a second real test-stand run showed --break-system-packages
    alone isn't enough: pip still tried to write to /root/.local and hit
    "OSError: [Errno 30] Read-only file system". This NGC/CUDA-12.8 judging
    container's filesystem is read-only outside specific writable mounts,
    apparently including BOTH the system site-packages (why
    --break-system-packages was needed at all) AND the default per-user
    site (/root/.local, pip's automatic fallback when it can't write
    system-wide). --target sidesteps this by installing into an arbitrary
    directory instead of either of those -- the OS temp dir (tempfile.
    gettempdir(): /tmp on Linux, about the only location virtually
    guaranteed writable in ANY Docker/OCI container regardless of how
    locked-down the rest of the filesystem is; unlike this submission's own
    extracted directory, which is plausibly read-only too, being treated as
    immutable input). tempfile.gettempdir() rather than a hardcoded "/tmp"
    -- portable, AND makes this function actually testable on a Windows dev
    box instead of silently no-op-ing on a path Windows can't use (which is
    exactly how the first version of this fix produced a false PASS
    locally: peft was already globally installed from earlier unrelated
    testing, masking a real install failure)."""
    if not os.path.isdir(wheels_dir):
        print(f"[SETUP] no wheels dir at {wheels_dir}, skipping offline install", flush=True)
        return
    # Every package explicitly, --no-deps -- NOT just the top-level names
    # relying on pip to resolve the rest. Discovered the hard way: --target
    # installs are resolved in ISOLATION from the ambient environment (unlike
    # a normal `pip install`, which checks site-packages for already-
    # satisfied requirements first) -- asking for just "peft" made pip try
    # to resolve peft's full requires list (numpy>=1.17, torch, transformers,
    # ...) from --find-links alone and fail on numpy, even though numpy is
    # very much already installed and importable. --no-deps sidesteps
    # dependency resolution entirely: install exactly these names into
    # target_dir, nothing else. numpy/torch/transformers are then found at
    # IMPORT time (not install time) via the ambient environment's own
    # sys.path, same as any other already-installed package -- target_dir
    # only needs to supply what's genuinely missing there.
    # Package names below are the CANONICAL PyPI names, verified directly
    # from each wheel's own METADATA "Name:" field (unzip -p *.whl
    # "*.dist-info/METADATA") rather than assumed from the wheel filename --
    # hf_xet, annotated_doc, and markdown_it_py's wheel FILENAMES use
    # underscores (wheel filename convention) but their actual package names
    # are hyphenated (hf-xet, annotated-doc, markdown-it-py); pip resolves
    # requirement names by PEP 503 normalization so this does matter for
    # --no-deps + explicit-name installs like this one, even though it's a
    # no-op for case (Pygments/pygments, PyYAML/pyyaml are equivalent).
    pkgs = ["peft", "accelerate", "huggingface_hub", "click", "hf-xet", "httpx", "httpcore",
            "h11", "anyio", "idna", "certifi", "colorama", "typer", "annotated-doc", "rich",
            "pygments", "markdown-it-py", "mdurl", "shellingham", "packaging", "tqdm", "filelock",
            "fsspec", "typing_extensions", "psutil", "pyyaml", "safetensors"]
    target_dir = os.path.join(tempfile.gettempdir(), "frida_lora_adapter_offline_pkgs")
    print(f"[SETUP] offline-installing {pkgs} (--no-deps, exact list, from {wheels_dir}) "
          f"into {target_dir} ...", flush=True)
    try:
        os.makedirs(target_dir, exist_ok=True)
        subprocess.check_call(
            # --no-cache-dir: belt-and-suspenders against the same class of
            # surprise as /root/.local -- pip's default wheel cache
            # (~/.cache/pip) is another location this could plausibly hit a
            # read-only filesystem on, even though a pure-wheel install
            # (no sdist build) doesn't strictly need the cache.
            [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-cache-dir", "--no-deps",
             "--find-links", wheels_dir, "--target", target_dir, *pkgs]
        )
        # append, not insert(0, ...): if a package this installs (e.g.
        # huggingface_hub, which transformers itself also depends on and
        # may already have a validated version of in the image) turns out
        # to already be present, the already-installed one should win --
        # appending only fills in genuinely ABSENT names (peft, accelerate)
        # instead of risking a version clash with whatever transformers was
        # actually tested against.
        sys.path.append(target_dir)
        print(f"[SETUP] offline install ok, {target_dir} appended to sys.path", flush=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[WARN] offline install failed ({e}) -- continuing, import will fail loudly "
              f"if peft is truly absent", flush=True)


# Must byte-match kaggle_kernels/frida-lora-sft/frida_lora_sft.py's actual
# training pipeline -- NOT the older joint-tokenize formula submits/lora
# uses (that one predates this pipeline and was never the one FRIDA's LoRA
# weights were trained against). Three things distinguish it from a plain
# cross-encoder tokenize(text, text_pair=...) call:
#   1. Item-level tokenization (each item's own "paraphrase: "+text, WITH
#      special tokens -- bos/eos baked in per item, matching the pre-built
#      frida-item-tokens-human cache training read from), not pair-level.
#   2. Category-span excision: `category` is 100% identical between both
#      items of every pair (CLAUDE.md, verified) -- zero discriminative
#      signal, pure budget waste at PAIR_MAX_LENGTH=128 (94% of real pairs
#      truncate at that budget). Located via a stable BPE token-pattern
#      search, not string manipulation (avoids re-tokenizing after editing
#      text).
#   3. allocate_pair_budget fair-share truncation instead of the tokenizer's
#      own truncation="longest_first" pair logic.
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640
PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256        # matches frida-item-tokens-human's own cap
PAIR_MAX_LENGTH = 128        # matches frida_lora_sft.py's active config

CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes" (+ colon=30 immediately after)
LABEL_COLON_TOKEN = 30


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_needed_item_texts(items_path, needed_ids):
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


def _find_token_pattern_starts(arr, tok_a, tok_b):
    """First index i per row where arr[row,i]==tok_a and arr[row,i+1]==tok_b,
    or -1 if not found. Vectorized (no Python loop over rows) -- identical
    to frida_lora_sft.py's own version (2-token pattern is all FRIDA's
    byte-BPE tokenizer needs; unlike BERTA's WordPiece, "Category"/
    "Attributes" are each single vocab tokens here)."""
    mask = (arr[:, :-1] == tok_a) & (arr[:, 1:] == tok_b)
    rows, cols = np.where(mask)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def _tokenize_items(tokenizer, id_to_text):
    """Tokenizes every unique item ONCE (paraphrase: prefix, WITH special
    tokens -- default add_special_tokens=True gives bos...eos, matching the
    training cache), then locates the Category:/Attributes: label span per
    item for excision. Returns id -> (input_ids[:real_len] int32 array,
    category_start, attr_start) so build_compact_ids/allocate_pair_budget
    can run per-pair without ever re-tokenizing."""
    ids = list(id_to_text.keys())
    texts = [PREFIX + id_to_text[i] for i in ids]

    enc = tokenizer(texts, padding="max_length", truncation=True,
                     max_length=ITEM_MAX_LENGTH, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int32)
    lengths = enc["attention_mask"].sum(axis=1).astype(np.int32)

    category_start = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)

    id_to_item = {}
    for row, item_id in enumerate(ids):
        id_to_item[item_id] = (input_ids[row], int(lengths[row]), int(category_start[row]), int(attr_start[row]))
    return id_to_item


def allocate_pair_budget(len1, len2, budget):
    """Fair-share truncation over COMPACT (category-excised) lengths --
    identical logic to frida_lora_sft.py's own (unit-tested there): a side
    under half-budget is never cut below its real length; slack from a
    short side goes to the other; if both exceed half, split evenly."""
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def _build_compact_ids(input_ids, item_length, category_start, attr_start):
    if category_start < 0 or attr_start < 0:
        return input_ids[:item_length]
    head = input_ids[:category_start]
    tail = input_ids[attr_start:item_length]
    return np.concatenate((head, tail))


def _build_pairs(match_df, id_to_item, pad_id):
    """Collate-time concatenation, mirroring frida_lora_sft.py's
    make_pair_collate_fn exactly: excise Category from each side, fair-share
    truncate to PAIR_MAX_LENGTH, concatenate (no CLS/SEP insertion -- T5 has
    no segment concept, each side's own bos/eos from tokenization already
    marks its boundary)."""
    n = len(match_df)
    input_ids = np.full((n, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
    attention_mask = np.zeros((n, PAIR_MAX_LENGTH), dtype=np.int64)

    id1_list = match_df["id1"].tolist()
    id2_list = match_df["id2"].tolist()
    n_missing = 0

    for i in range(n):
        item1 = id_to_item.get(id1_list[i])
        item2 = id_to_item.get(id2_list[i])
        if item1 is None or item2 is None:
            n_missing += 1
            continue

        compact1 = _build_compact_ids(*item1)
        compact2 = _build_compact_ids(*item2)
        cl1, cl2 = len(compact1), len(compact2)

        if cl1 + cl2 <= PAIR_MAX_LENGTH:
            keep1, keep2 = cl1, cl2
        else:
            keep1, keep2 = allocate_pair_budget(cl1, cl2, PAIR_MAX_LENGTH)

        combined = np.concatenate((compact1[:keep1], compact2[:keep2]))
        m = len(combined)
        input_ids[i, :m] = combined
        attention_mask[i, :m] = 1

    if n_missing:
        print(f"[WARN] {n_missing} pairs referenced an item id missing from items.parquet "
              f"(left as all-pad -- attention_mask all-zero for those rows)", flush=True)

    return input_ids, attention_mask


class FridaCrossEncoderClassifier(nn.Module):
    """CLS (position-0) pooling -- matches frida_lora_sft.py's own
    FridaCrossEncoderClassifier exactly (FRIDA's model card recommendation;
    NOT mean pooling, which is BERTA's own separate choice, not this one's)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).view(-1)


def _load_model(base_model_path, adapter_path, classifier_path, device):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    # torch_dtype=, not dtype=: this repo already hit this exact
    # incompatibility once (giga3b probe phase, CLAUDE.md/memory) -- an
    # older pinned transformers didn't recognize the newer dtype= kwarg
    # name. The judging image's transformers version is unverified (Docker
    # Desktop wasn't reachable locally to check), so use the
    # longer-supported name defensively rather than assume a recent version.
    try:
        # eager, not sdpa/default -- matches what the adapter was actually
        # trained under (see frida_lora_sft.py's build_fresh_model docstring
        # for the root-cause history); "must byte-match training/inference"
        # applies to the computational path too, not just the text formula.
        base_model = T5EncoderModel.from_pretrained(base_model_path, attn_implementation="eager",
                                                       torch_dtype=torch.bfloat16)
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    hf_logging.set_verbosity(default_verbosity)

    # is_trainable defaults to False here, which is what we want at inference
    # (no gradients needed) -- unlike a resumed TRAINING run, which needs
    # is_trainable=True explicitly (see this repo's own documented PEFT
    # is_trainable trap in CLAUDE.md / kaggle_kernels/berta-lora-sft).
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    hidden_dim = base_model.config.d_model

    classifier_head = nn.Linear(hidden_dim, 1)
    classifier_head.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    # Match the encoder's dtype (bf16, loaded via torch_dtype= above -- see
    # that comment for why weights are stored bf16, not left fp32-with-
    # autocast: halves the bundled base model's on-disk size, which matters
    # for the 5GB submission zip budget). Without this the classifier stays
    # fp32 and errors on CPU (no autocast there to paper over the mismatch)
    # -- caught by a real local test run, not assumed safe on CUDA either.
    classifier_head = classifier_head.to(dtype=torch.bfloat16)

    model = FridaCrossEncoderClassifier(peft_model, classifier_head)
    model.to(device)
    model.eval()
    return model, tokenizer


def _run_inference(model, input_ids_all, attention_mask_all, device, batch_size):
    use_amp = device.type == "cuda"
    n = input_ids_all.shape[0]
    probs = np.empty(n, dtype=np.float32)

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            input_ids = torch.from_numpy(input_ids_all[i:i + batch_size]).to(device, non_blocking=True)
            attention_mask = torch.from_numpy(attention_mask_all[i:i + batch_size]).to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            probs[i:i + input_ids.shape[0]] = torch.sigmoid(logits.float()).cpu().numpy()

    return probs


def predict_pipeline(items_path, matches_path, base_model_path, adapter_path, classifier_path,
                      output_csv_path, wheels_dir, batch_size=1024):
    t0 = time.time()
    print("=== Starting FRIDA LoRA (adapter, no merge) prediction pipeline ===", flush=True)

    maybe_offline_install(wheels_dir)

    print("[1/6] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[1/6] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[2/6] Loading needed item texts ...", flush=True)
    id_to_text = _load_needed_item_texts(items_path, needed_ids)
    print(f"[2/6] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/6] Loading model + tokenizer ...", flush=True)
    device = _get_device()
    model, tokenizer = _load_model(base_model_path, adapter_path, classifier_path, device)
    print(f"[3/6] device={device} ({time.time() - t0:.1f}s)", flush=True)

    print("[4/6] Tokenizing items (once each) + locating Category:/Attributes: labels ...", flush=True)
    id_to_item = _tokenize_items(tokenizer, id_to_text)
    print(f"[4/6] done ({time.time() - t0:.1f}s)", flush=True)

    print("[5/6] Building pairs (excise Category, fair-share truncate to "
          f"PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}) ...", flush=True)
    input_ids_all, attention_mask_all = _build_pairs(match_df, id_to_item, tokenizer.pad_token_id)
    print(f"[5/6] done ({time.time() - t0:.1f}s)", flush=True)

    print("[6/6] Running inference ...", flush=True)
    probs = _run_inference(model, input_ids_all, attention_mask_all, device, batch_size=batch_size)
    print(f"[6/6] done ({time.time() - t0:.1f}s)", flush=True)

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
    print(f"=== Done in {elapsed:.1f}s ({len(match_df) / max(elapsed, 1e-9):.1f} pairs/sec) ===", flush=True)
    return results_df
