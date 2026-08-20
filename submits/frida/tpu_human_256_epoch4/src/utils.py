import os

# Hardware-aware env tuning, mirrored from submits/lora/src/utils.py -- 20 CPU
# cores / 200GB RAM / 1x H100 80GB at inference time.
#
# Thread-pool sizes set to 20, matching the actual judging core count (was
# 32 in the copied-from original -- oversubscribing 12 more OS threads than
# there are cores buys nothing for CPU-bound work here and just adds
# scheduling/context-switch overhead; confirmed the same dynamic locally,
# where this dev box's 12 logical cores were oversubscribed by the same
# stale "32" for RAYON_NUM_THREADS). If the judging box's "20 CPU cores" in
# CLAUDE.md turns out to mean 20 PHYSICAL cores with hyperthreading (i.e.
# ~40 logical), 20 is still safe -- just leaves some headroom rather than
# oversubscribing -- so this is a one-directional fix, not a guess that can
# make things worse.
os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")

import gc
import json
import sys
import subprocess
import tempfile
import threading
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
ITEM_MAX_LENGTH = 256        # matches the v2 item cache's own cap
# 256, NOT the 128 the earlier FRIDA submissions used: this checkpoint comes
# from kaggle_kernels/frida-tpu-human-sft, whose one deliberate change was
# raising the pair budget 128 -> 256 (128 truncates ~94% of real pairs).
# Serving it at 128 would feed it pairs half the length it was trained on.
PAIR_MAX_LENGTH = 256

# ---------------------------------------------------------------------
# BATCH SIZE -- an ANALYTIC ceiling estimate, not a measurement and not a
# live probe (a probe was tried and removed -- see predict_pipeline's own
# docstring: it added several real forward passes as pure startup overhead,
# and a real judging run failed the 1-minute Check stage after it was
# added). 1024 -- the "Frida Lora V2 Epoch" default -- is almost certainly
# well under what an H100 80GB can actually hold; the arithmetic below
# derives a real ceiling instead of guessing a round number in either
# direction.
#
# FRIDA (d_model=1536, 24 layers, d_ff=4096, 24 heads, d_kv=64 -- read
# straight from config.json, not assumed) forward-only under
# torch.inference_mode(): no backward graph is retained, so peak activation
# memory is bounded by roughly how many layers' worth of intermediate
# tensors the allocator keeps alive at once, not by all 24 simultaneously
# (that would be the training-memory story, not inference's). Per layer,
# per batch unit, counting every major intermediate tensor as if it were
# simultaneously live (deliberately conservative -- an overcount, not a
# best-case fit): 9 buffers of shape (S, d_model) [pre-attn norm, Q, K, V,
# attn-out, out-proj, residual, pre-ffn norm, ffn-down, residual -- note
# that's actually more than 9 named items above; the constant below is
# computed directly from the formula, not hand-counted here] + 2 buffers of
# shape (num_heads, S, S) [attention scores + softmax probs] + 3 buffers of
# shape (S, d_ff) [T5's gated FFN: wi_0, wi_1, and their product] --
#   per_batch_bytes = (9*S*d_model + 3*S*d_ff + 2*num_heads*S*S) * 2 bytes(bf16)
#   = (9*256*1536 + 3*256*4096 + 2*24*256*256) * 2 = 19,660,800 bytes/layer
# Multiplied by an assumed 2 layers' worth of simultaneous residency (a
# real, PyTorch-internals-dependent number this project has no direct
# measurement of -- 2 is a middle assumption, not a lower or upper bound):
# ~37.5MB per unit of batch. Target 70% of the H100's 80GB, minus ~1.7GB for
# the frozen bf16 weights and ~1GB for CUDA context overhead, gives a safe
# batch on the order of ~1300-1400; BATCH_SIZE_ESTIMATE below sits at 2048,
# deliberately above the conservative end of that range rather than at it
# -- OOM is not fatal here (a crashed submission does not cost a daily slot
# any more than a slow one does), so probing the estimate's real headroom by
# testing slightly past it is cheap, and batch size is not this run's
# current bottleneck anyway (see the Check-stage investigation above) --
# but it is NOT free of OOM risk if the "2 layers resident" assumption
# turns out to be wrong. Lower toward ~1280-1536 first if this OOMs.
BATCH_SIZE_ESTIMATE = 2048

CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes" (+ colon=30 immediately after)
LABEL_COLON_TOKEN = 30

# ---------------------------------------------------------------------
# v2 ATTRIBUTE-ORDERING POLICY -- replicated verbatim from
# kaggle_kernels/frida-tokenize-items-v2, which built the item cache
# (maximmarkovbb/frida-tokenize-items-v2-cache) this checkpoint trained on.
#
# This is not an optional refinement. The earlier FRIDA submission served
# raw, unparsed attribute text to a model trained on reordered text, so the
# characters surviving the 384-char slice were not the ones training saw --
# a real train/serve mismatch, and a candidate cause of that submission
# scoring 0.386. Getting it right here is the whole reason this file is not
# just the old one with PAIR_MAX_LENGTH bumped.
#
# The rules demote attributes to the end of the JSON before the slice:
#   1. value longer than LONG_VALUE_CHARS
#   2. value containing "https"
#   3. "oem" in the key or the value
# Applied as a single rank because the original applies them sequentially,
# so a later rule overwrites an earlier one, and a STABLE sort by that rank
# is exactly what repeated "move to the end" passes produce.
#
# variant_base only: the cache's 5%-probability shuffled variant is a
# training-time augmentation and must not run at inference.
# ---------------------------------------------------------------------
LONG_VALUE_CHARS = 200
LINK_MARKER = "https"
OEM_MARKER = "oem"


def _demotion_rank(key, value):
    v = value if isinstance(value, str) else str(value)
    rank = 0
    if len(v) > LONG_VALUE_CHARS:
        rank = 1
    vl = v.lower()
    if LINK_MARKER in vl:
        rank = 2
    if OEM_MARKER in vl or OEM_MARKER in str(key).lower():
        rank = 3
    return rank


def reorder_attributes(attr_str):
    """Re-serialized attributes JSON with the policy applied. Falls back to
    the untouched input on anything not cleanly parseable as a non-empty
    dict -- the same defensive convention the tokenize kernel uses, so
    malformed blobs are handled identically on both sides."""
    if not attr_str:
        return attr_str
    try:
        d = json.loads(attr_str)
    except Exception:
        return attr_str
    if not isinstance(d, dict) or not d:
        return attr_str

    ranked = [(_demotion_rank(k, v), k, v) for k, v in d.items()]
    ranked.sort(key=lambda t: t[0])
    return json.dumps({k: v for _r, k, v in ranked}, ensure_ascii=False)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_bf16_capability(device):
    """Diagnostic only -- does NOT change dtype. This submission runs bf16
    unconditionally (matching how the checkpoint was trained: frozen T5
    base in bf16 throughout, per kaggle_kernels/frida-tpu-human-sft), which
    is correct and fast on Ampere+/Hopper (native bf16 tensor cores,
    including the real H100 judging stand) but on an older GPU without them
    (Turing/T4, sm_75) bf16 falls back to an EMULATED, slow path -- exactly
    the trap CLAUDE.md documents for this project's own Giga3B probe
    (torch.cuda.is_bf16_supported() returns True even on a T4; the real gate
    is compute capability >= 8). Switching precision here is NOT done
    automatically: T5 models are well known to overflow in fp16 (narrower
    exponent range than bf16, and T5's activations run large), so a silent
    fallback risks a correctness regression worse than the slowdown it would
    fix. This just makes the risk visible in the run's own logs, so a dry
    run on a non-H100 GPU (e.g. Kaggle's own T4, per this project's
    established kaggle_kernels/lora-timing-dryrun practice) shows up as a
    labeled, expected slowdown rather than an unexplained one."""
    if device.type != "cuda":
        return
    major, _minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if major >= 8:
        print(f"[PRECISION] {name}: compute capability {major}.x -- native bf16 tensor cores, "
              f"bf16 path is the fast path here", flush=True)
    else:
        print(f"[PRECISION] WARNING: {name}: compute capability {major}.x has NO native bf16 "
              f"tensor cores -- bf16 here runs an EMULATED, slow path (real project precedent: "
              f"CLAUDE.md's Giga3B probe measured ~5x slower emulated bf16 vs int8 on a T4). "
              f"This submission does not fall back to fp16 automatically (T5 is known to "
              f"overflow in fp16's narrower exponent range) -- if this run is a T4 dry-run "
              f"rather than the real H100 stand, expect this to be the slow, non-representative "
              f"case, not a bug.", flush=True)


# multiprocessing.Pool(os.cpu_count()) was tried here and REMOVED after a
# real judging-stand crash: os.cpu_count() reported 240 on that container
# (see README.md's "os.cpu_count() ... returned 240, not the 20 CLAUDE.md
# documents"), and Pool(240) forks 240 new OS processes essentially at once
# in multiprocessing.pool._repopulate_pool_static -- which failed with
# `BlockingIOError: [Errno 11] Resource temporarily unavailable` from
# os.fork() itself, killing the whole run on Public (114,235 pairs) within
# the first minute. CPU CORE COUNT and PROCESS COUNT are governed by
# different, unrelated limits in a container (cpu.max/cpuset vs.
# pids.max/RLIMIT_NPROC) -- os.cpu_count() says nothing safe about the
# latter, and this run proved a real container here caps it well below 240.
#
# No replacement worker-count constant was picked, on purpose: any number
# is a guess against an unmeasured real ceiling, and finding that ceiling
# by resubmitting until one doesn't crash would cost more real submission
# attempts than this optimization was ever going to save. Measured cost of
# the single-threaded fallback (what this now always uses) at REAL run
# scale: reorder_attributes runs ~52K items/sec locally (10.6s for the
# largest 550K-item scale this project has measured) -- at 223,536 items
# (this project's own Public-stage scale), that is ~4.3s. A single-digit
# number of seconds, not a reason to reintroduce a proven crash risk.
def _reorder_and_slice(attr):
    return reorder_attributes(attr)[:MAX_ATTR_CHARS]


def _load_needed_item_texts(items_path, needed_ids):
    """Byte-matches kaggle_kernels/frida-tokenize-items-v2's build_texts.

    The ordering of operations matters and is not the obvious one: the
    attribute policy runs on the FULL attributes string and the
    MAX_ATTR_CHARS slice is applied AFTER. Slicing first would hand
    json.loads a truncated, invalid blob -- and deciding what survives that
    slice is the entire point of the policy.

    A per-item json.loads is the price, bounded by the number of UNIQUE
    items the pairs reference (~2 per pair, so ~230K-550K at Public/Private
    scale), not by the 13M-row catalog -- the case CLAUDE.md warns OOMs.
    Parse failures fall through to the raw string rather than raising, so a
    malformed blob costs that one item's ordering, never the run.

    Measured cost of this step alone at 550K real items:
    reorder_attributes ~10.6s single-threaded (48x a raw-string-slice
    baseline of 0.22s -- the json.loads + re-serialize is real but small in
    absolute terms, not the dominant stage of the pipeline). Parallelized
    below across the judging box's cores (20 per CLAUDE.md) since each
    item's reorder is independent and the function is pure -- pickling one
    string in, one string out, per item, is cheap relative to the json work
    itself."""
    df = pd.read_parquet(
        items_path,
        engine="pyarrow",
        columns=["id", "name", "category", "attributes"],
        filters=[("id", "in", list(set(needed_ids)))],
    )
    names = df["name"].fillna("").astype(str).tolist()
    categories = df["category"].fillna("").astype(str).tolist()
    attributes = df["attributes"].fillna("").astype(str).tolist()

    attrs_ordered = [_reorder_and_slice(a) for a in attributes]

    texts = []
    for name, category, attr_ordered in zip(names, categories, attrs_ordered):
        name = name[:MAX_NAME_CHARS]
        category = category[:MAX_CATEGORY_CHARS]
        texts.append(
            f"Name: {name} Category: {category} Attributes: {attr_ordered}"[:MAX_ITEM_CHARS]
        )
    return dict(zip(df["id"].tolist(), texts))


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


# A single tokenizer() call over the full item set (a real deployment sees
# 230K-550K texts at once) hit a real, reproducible MemoryError locally --
# not a CUDA OOM, a plain Python one, inside transformers' own tensor-
# conversion path (tokenization_utils_base.py's `flatten()`/`as_tensor()`),
# on transformers 5.15.0. The judging image's transformers version is
# explicitly UNVERIFIED (see maybe_offline_install's own docstring -- Docker
# Desktop wasn't reachable to check it directly), so rather than hope the
# judging image's version doesn't share this behavior, tokenize in bounded
# chunks. Chunking changes nothing about the output: each text is tokenized
# independently (no cross-item padding/truncation interaction), so
# concatenating chunk results is byte-identical to one giant call that
# happened to succeed.
#
# 150_000, not the 20_000 borrowed from kaggle_kernels/frida-tokenize-items-v2
# (a training-time CPU kernel with its own, unrelated memory budget) -- that
# number was never actually derived from THIS constraint. The real
# MemoryError above was hit on a dev box with only 16.5GB RAM; the judging
# box has 200GB, ~12x more headroom. 150_000 (3.7x smaller than the 550K
# that broke on 12x less RAM -- a real, comfortable margin, not "no
# chunking") cuts the chunk count at Private scale from ~28 down to ~4,
# removing most of the per-chunk Python-list-append + final concatenate
# overhead, while keeping the chunk boundary itself as a real safety net if
# the judging image's transformers/pyarrow behavior turns out not to be
# purely RAM-bound (version-specific inefficiency, not just size).
TOKENIZE_CHUNK_SIZE = 150_000


def _tokenize_items(tokenizer, id_to_text):
    """Tokenizes every unique item ONCE (paraphrase: prefix, WITH special
    tokens -- default add_special_tokens=True gives bos...eos, matching the
    training cache), then locates the Category:/Attributes: label span per
    item for excision. Returns id -> (input_ids[:real_len] int32 array,
    category_start, attr_start) so build_compact_ids/allocate_pair_budget
    can run per-pair without ever re-tokenizing."""
    ids = list(id_to_text.keys())
    texts = [PREFIX + id_to_text[i] for i in ids]

    input_ids_chunks = []
    lengths_chunks = []
    for start in range(0, len(texts), TOKENIZE_CHUNK_SIZE):
        chunk = texts[start:start + TOKENIZE_CHUNK_SIZE]
        enc = tokenizer(chunk, padding="max_length", truncation=True,
                         max_length=ITEM_MAX_LENGTH, return_tensors="np")
        input_ids_chunks.append(enc["input_ids"].astype(np.int32))
        lengths_chunks.append(enc["attention_mask"].sum(axis=1).astype(np.int32))
    input_ids = np.concatenate(input_ids_chunks, axis=0)
    lengths = np.concatenate(lengths_chunks, axis=0)
    del input_ids_chunks, lengths_chunks
    gc.collect()

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


# Split from one _load_model() into two functions -- _load_tokenizer (small
# files, fast: vocab/merges/config) and _load_model_weights (the base T5
# encoder's real weights, the LoRA adapter, the classifier head -- the part
# that actually reads multiple GB off disk and does the peft/device-move
# work). The split exists so predict_pipeline can run the SLOW half in a
# background thread while the main thread gets on with item-text loading
# and tokenization, instead of the two waiting on each other for no reason
# (tokenization only ever needed the tokenizer, never the model weights).
# See predict_pipeline's own comment for how the threads are joined.
def _load_tokenizer(base_model_path):
    return AutoTokenizer.from_pretrained(base_model_path)


def _load_model_weights(base_model_path, adapter_path, classifier_path, device):
    from peft import PeftModel

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
    return model


# GPU batch-size auto-probing was tried here and removed, per direct
# instruction, after a real judging run ("Container did not finish in time")
# failed at the Check stage (1 min / 1000 pairs) in ~3 minutes. The specific
# zip that failed used a manually fixed batch_size=16384, not this probe --
# so that data point alone doesn't prove the probe itself was the cause. But
# the probe adds several REAL forward passes (doubling from 128 up to a
# possible 8192, then a binary search) purely to measure memory, before any
# of the run's actual 1000 Check-stage pairs are processed -- pure fixed
# overhead that a 1000-pair run gains nothing from (it would never need a
# batch anywhere near what the probe searches up to). Removing it is correct
# regardless of whether it was THE cause: it cannot help at Check-stage
# scale and can only add startup latency, which is exactly what Check
# exists to catch.


# FIXED-SHAPE batching -- every GPU-facing forward call gets EXACTLY
# batch_size rows, never a ragged final batch. Added after a real, still not
# fully explained anomaly: seq=192/batch=6912 OOM'd needing MORE total
# memory (67.57 GiB already in use before the failed allocation) than the
# LARGER seq=192/batch=8192 config had needed (52.65 GiB) -- backwards from
# what pure per-batch linear scaling predicts. The attn-probs term itself
# matched the derived formula exactly both times (22.78 GiB predicted and
# logged at 6912, 27.00 GiB at 8192 -- see BATCH_SIZE_FIXED's own comment),
# so the discrepancy is specifically in the OTHER, "already in use" memory,
# not in this file's core VRAM model. The batch=6912 run needed MORE
# iterations (17 vs 14 at batch=8192) AND its ragged final batch (3643 rows,
# 52.7% of a full batch) was a DIFFERENT shape than every batch before it --
# a real, previously-hit pattern in this project (see .claude/skills/
# kaggle-experiment-safety: "a training loop that lets shapes drift... pays
# a real compile on every new shape" -- the TPU/XLA version of this same
# problem) where PyTorch's caching allocator can end up holding reserved
# blocks for MULTIPLE distinct shapes simultaneously instead of reusing one,
# inflating peak reserved memory well past what a single steady-state shape
# would need. This is the best-supported hypothesis given the data, not a
# proven root cause -- confirming it would need either real GPU profiling
# or the deliberate-OOM-logging technique noted in README.md's "Как дёшево
# профилировать реальный H100-стенд" section.
#
# The fix costs nothing in accuracy: the ragged tail is padded up to
# batch_size by repeating its own last row (same technique already proven
# in kaggle_kernels/frida-tpu-embed-cache-human's own ragged-tail handling),
# only the valid prefix of that batch's output is kept, and the repeated
# rows' predictions are computed and then discarded, never written to
# `probs`. Every real pair still gets exactly the same forward pass and the
# same prediction as before -- this only removes the ONE shape-varying call
# from the loop.
# Progress logging every ~20% of rows, not per-batch tqdm -- a full tqdm bar
# either doesn't render in a non-interactive judging log at all, or if its
# carriage-return-based updates get flattened to one plain line per batch by
# a simple log capture, hundreds of lines could push the actually useful
# early lines out of a length-capped log. If a run dies on TIME (not OOM),
# this is the only way to know afterward how far inference actually got --
# same motivation as README.md's "Как дёшево профилировать реальный
# H100-стенд": every log line here survives even a run that never finishes.
PROGRESS_LOG_FRACTION = 0.20


def _run_inference(model, input_ids_all, attention_mask_all, device, batch_size):
    use_amp = device.type == "cuda"
    n = input_ids_all.shape[0]
    probs = np.empty(n, dtype=np.float32)
    t0 = time.time()
    next_log_at = max(1, int(n * PROGRESS_LOG_FRACTION))

    def run_one_batch(ids_np, mask_np, n_valid):
        input_ids = torch.from_numpy(ids_np).to(device, non_blocking=True)
        attention_mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(logits.float()).cpu().numpy()[:n_valid]

    def maybe_log_progress(done):
        nonlocal next_log_at
        if done < next_log_at:
            return
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        print(f"[7/7]   progress: {done:,}/{n:,} ({100 * done / n:.0f}%) "
              f"in {elapsed:.1f}s ({rate:.1f} pairs/s, ETA {eta:.0f}s)", flush=True)
        while next_log_at <= done:
            next_log_at += max(1, int(n * PROGRESS_LOG_FRACTION))

    with torch.inference_mode():
        n_full = (n // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            probs[i:i + batch_size] = run_one_batch(
                input_ids_all[i:i + batch_size], attention_mask_all[i:i + batch_size], batch_size)
            maybe_log_progress(i + batch_size)

        if n_full < n:
            tail = n - n_full
            pad_ids = np.tile(input_ids_all[n - 1:n], (batch_size, 1))
            pad_mask = np.tile(attention_mask_all[n - 1:n], (batch_size, 1))
            pad_ids[:tail] = input_ids_all[n_full:n]
            pad_mask[:tail] = attention_mask_all[n_full:n]
            probs[n_full:n] = run_one_batch(pad_ids, pad_mask, tail)
            maybe_log_progress(n)

    print(f"[7/7]   inference done: {n:,}/{n:,} (100%) in {time.time() - t0:.1f}s", flush=True)
    return probs


def predict_pipeline(items_path, matches_path, base_model_path, adapter_path, classifier_path,
                      output_csv_path, wheels_dir, batch_size=BATCH_SIZE_ESTIMATE):
    """batch_size defaults to an ANALYTIC estimate (see BATCH_SIZE_ESTIMATE's
    own comment for the arithmetic) -- not measured, not probed. A live
    probe was tried and removed: it adds several real forward passes as pure
    startup overhead before any of the run's actual pairs are touched, which
    a real judging run showed exceeding the Check-stage budget ("Container
    did not finish in time"). Pass an explicit int to override.

    Model weights load in a BACKGROUND THREAD, overlapping with item-text
    loading + tokenization + pair-building on the main thread -- see the
    inline comments below for why this is safe (no shared GPU state touched
    by the main thread during the overlap window) and what it targets (the
    model-loading stage was structurally forced to finish before ANY other
    work could start, even though nothing before inference actually needs
    the model weights -- only the tokenizer, which loads separately and
    fast). This is a scheduling change only: same model, same tokenizer,
    same texts, same tensors going into inference -- it cannot change what
    gets predicted, only when each step happens relative to the others."""
    t0 = time.time()
    print("=== Starting FRIDA LoRA (adapter, no merge) prediction pipeline ===", flush=True)

    maybe_offline_install(wheels_dir)

    print("[1/7] Loading tokenizer (fast) + starting model-weight load in the background ...",
          flush=True)
    device = _get_device()
    _log_bf16_capability(device)
    tokenizer = _load_tokenizer(base_model_path)
    model_holder = {}
    model_error = {}

    def _load_model_thread():
        try:
            model_holder["model"] = _load_model_weights(base_model_path, adapter_path,
                                                         classifier_path, device)
        except Exception as e:  # noqa: BLE001 -- re-raised on the main thread below, not swallowed
            model_error["exc"] = e

    model_thread = threading.Thread(target=_load_model_thread, name="model-weights-loader")
    model_thread.start()
    print(f"[1/7] tokenizer ready, model loading in background ({time.time() - t0:.1f}s)", flush=True)

    print("[2/7] Loading matches ...", flush=True)
    match_df = pd.read_parquet(matches_path)
    needed_ids = pd.unique(pd.concat([match_df["id1"], match_df["id2"]])).tolist()
    print(f"[2/7] pairs={len(match_df):,} unique ids needed={len(needed_ids):,} ({time.time() - t0:.1f}s)", flush=True)

    print("[3/7] Loading needed item texts (single-threaded reorder -- "
          "see _reorder_and_slice's comment for why) ...", flush=True)
    id_to_text = _load_needed_item_texts(items_path, needed_ids)
    print(f"[3/7] items loaded={len(id_to_text):,} ({time.time() - t0:.1f}s)", flush=True)

    print(f"[4/7] Tokenizing items (once each, in chunks of {TOKENIZE_CHUNK_SIZE:,}) "
          "+ locating Category:/Attributes: labels ...", flush=True)
    id_to_item = _tokenize_items(tokenizer, id_to_text)
    print(f"[4/7] done ({time.time() - t0:.1f}s)", flush=True)

    print("[5/7] Building pairs (excise Category, fair-share truncate to "
          f"PAIR_MAX_LENGTH={PAIR_MAX_LENGTH}) ...", flush=True)
    input_ids_all, attention_mask_all = _build_pairs(match_df, id_to_item, tokenizer.pad_token_id)
    print(f"[5/7] done ({time.time() - t0:.1f}s)", flush=True)

    print("[6/7] Waiting for background model-weight load (if not already done) ...", flush=True)
    model_thread.join()
    if "exc" in model_error:
        raise model_error["exc"]
    model = model_holder["model"]
    print(f"[6/7] model ready ({time.time() - t0:.1f}s)", flush=True)

    print(f"[7/7] Running inference (batch_size={batch_size}) ...", flush=True)
    probs = _run_inference(model, input_ids_all, attention_mask_all, device, batch_size=batch_size)
    print(f"[7/7] done ({time.time() - t0:.1f}s)", flush=True)

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
