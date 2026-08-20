"""
Cross-encoder-style loader for a pre-quantized 4-bit NF4 build of
`ai-sage/Giga-Embeddings-instruct` (`iMiW/Giga-Embeddings-instruct-4bit-nf4`
on HuggingFace), isolated from utils.py so the data pipeline stays importable
without torch on a laptop.

Swapped from this project's own int8 quantize-and-save pipeline
(kaggle_kernels/giga3b-quantize-benchmark) to this pre-quantized checkpoint:
smaller on disk (2.17GB vs 3.5GB -- more zip headroom), same base model and
near-byte-identical modeling code (diffed against the old checkpoint's copy:
the ONLY difference across 44KB is `configuration.text_config.name_or_path`
vs `configuration.name_or_path` on one unused-by-us init path), and its own
baked-in `quantization_config` (bnb NF4 + double-quant, compute dtype
bfloat16) -- so no `BitsAndBytesConfig` needs constructing here at all; the
checkpoint's own saved config takes precedence over anything passed at load
time regardless (observed on the old int8 checkpoint: "You passed
quantization_config ... but the model already has a quantization_config
attribute. The quantization_config from the model will be used"). This also
removes our own quantization step as a possible contributor to the
~13.8GB-vs-~3.5GB baseline VRAM mystery the old int8 checkpoint showed on a
T4 (see reports/experiment_log.md) -- still needs its own fresh empirical
verification on a real GPU, not assumed fixed by the swap alone.

Corrections from the earlier bi-encoder version of this probe (still true,
unchanged by the checkpoint swap):
  - This repo's whole "cross-encoder ceiling" batch (BERTA/matryoshka/FRIDA)
    encodes each PAIR once (id1+id2 concatenated -> one forward pass), not
    each item separately + cosine similarity. Giga was originally wired as a
    bi-encoder; on this dataset (~1.94 items/pair, almost no item reuse -- see
    CLAUDE.md) that pays ~2x the forward passes at ~2x the tokens each for no
    benefit, so it's switched to the same cross-encoder pattern here.
  - The tokenizer here is Llama-style BPE, not BERT/RoBERTa/T5 -- verified
    locally that `tokenizer(text=, text_pair=)` on this checkpoint just glues
    the two strings together with NO separator at all ("hello worldsome other
    text"), unlike the WordPiece/SentencePiece tokenizers of the other three
    models. So the pair is built as ONE manually-formatted string instead
    (see utils.py's build_pair_texts).
  - The model card's own README says symmetric tasks (classification,
    similarity) need an instruction prefix on every input or "you'll see a
    quality drop" -- the earlier bi-encoder version used none. Product
    matching is exactly that kind of symmetric classification task, so
    utils.py now prepends one, worded after the README's own example
    ("Классифицируй отзыв на товар...").
  - `return_embeddings=True` is passed to the forward call. Confirmed
    directly from the checkpoint's own modeling_gigarembed.py this time (not
    just inferred): `forward()` wraps the base LM + the model's own trained
    latent-attention pooling head in `torch.autocast('cuda', dtype=
    torch.bfloat16)`, then -- when `return_embeddings=True` -- masked-mean-
    pools the pooling head's output over the sequence dimension and
    L2-normalizes it, returning a plain tensor (not a ModelOutput). That
    `autocast('cuda', bfloat16)` lives INSIDE forward() regardless of the
    weight dtype/quantization, so actual compute always happens in bf16 --
    on the judge stand's H100 (native bf16 tensor cores) this should be
    meaningfully faster than the T4 measurements this project has so far
    (T4 has no bf16 tensor cores; bf16 there runs emulated, see the Giga-3B
    bi-encoder probe's own T4 gotcha in CLAUDE.md).
"""

import logging
import os
import warnings

# bitsandbytes logs a "MatMul8bitLt: inputs will be cast from ... to float16
# during quantization" WARNING on every single quantized matmul call --
# thousands of lines per run, which on Kaggle evicts the actually-useful
# [GIGA]/[LOAD]/[DATA] progress lines from the tail-truncated log (see
# CLAUDE.md's giga3b-quantize-benchmark gotcha). Silence it at import time,
# before any bitsandbytes op runs.
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*will be cast from.*during quantization.*")


class GigaCrossEncoder:
    """encode_pairs(list[str] combined-pair-texts, batch_size) -> np.ndarray
    for simple/direct use; tokenize_batch()/forward_batch() are split out
    separately so utils.py's extract_pair_embeddings can overlap CPU
    tokenization of the next batch with the GPU forward pass of the current
    one."""

    # Measured on real pair texts (instruction + item1 + item2, 3000-pair
    # sample): mean 269.8 tokens, p50=276, p90=350 -- at the old 192 cap,
    # 85.7% of pairs were already being TRUNCATED, not padded (padding waste
    # at 192 was only ~2.4%). So this isn't a padding-waste fix, it's a
    # direct compute cut: 128/192 =~0.67x the linear-term FLOPs/pair =~1.5x
    # throughput, at the cost of keeping even less of an already-mostly-cut
    # pair (consistent with the other 3 cross-encoder probes' 128-token
    # budget -- this was the odd one out at 192). Token distribution is a
    # property of the base model's tokenizer + our own text formula, both
    # unchanged by the 4-bit swap, so this number still applies as-is.
    MAX_SEQ_LENGTH = 128

    def __init__(self, weights_dir, attn="auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir, trust_remote_code=True)

        # "torch_dtype", not "dtype": this custom model needs a transformers
        # version whose from_pretrained still accepts this kwarg name (see
        # maybe_offline_install) -- "dtype" is a newer rename that an older
        # pinned transformers doesn't recognize as a loading kwarg, so it
        # falls through to GigarEmbedModel.__init__() as an arbitrary kwarg
        # and TypeErrors (confirmed on a real Kaggle T4 run with the old
        # int8 checkpoint). Sets the dtype for whatever submodules the bnb
        # 4-bit conversion leaves unquantized (embeddings, norms, etc.) --
        # the quantized layers' own compute dtype is separately pinned to
        # bfloat16 by the checkpoint's own quantization_config.
        base_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            # Explicit single-GPU device_map instead of relying on the
            # implicit default: without it, bnb-quantized loading can leave
            # non-quantized submodules (embeddings, the latent-attention
            # pooling head) placed by whatever transformers' auto-dispatch
            # decides rather than pinned to the one GPU we actually have.
            base_kwargs["device_map"] = {"": 0}
        # No quantization_config passed here: this checkpoint's own
        # config.json already carries one (bnb NF4 + double-quant, compute
        # dtype bfloat16) and a saved config takes precedence over anything
        # passed at load time regardless (see module docstring).

        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        backends = []
        if attn == "auto":
            if cap[0] >= 8:
                try:
                    import flash_attn  # noqa: F401
                    backends.append("flash_attention_2")
                except Exception:
                    pass
            backends += ["sdpa", "eager"]
        else:
            backends = [attn]

        last_err = None
        model = None
        for candidate in backends:
            model_kwargs = dict(base_kwargs, attn_implementation=candidate)
            try:
                print(f"[GIGA] loading with attn_implementation={candidate} ...", flush=True)
                model = AutoModel.from_pretrained(weights_dir, **model_kwargs)
                print(f"[GIGA] loaded ok via attn_implementation={candidate}", flush=True)
                break
            except Exception as e:
                last_err = e
                print(f"[GIGA] attn_implementation={candidate} failed: {type(e).__name__}: {e}", flush=True)
        if model is None:
            raise RuntimeError(f"all attention backends failed to load the encoder; last error: {last_err}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        self.model = model
        self.device = device
        self._torch = torch
        self._compiled = False
        # Loads eager, full stop -- compiling is a separate, explicit,
        # evidence-based step (see benchmark_and_maybe_compile) run AFTER
        # the real batch size is known, not here. Compiling at load time
        # used to happen automatically here; a real T4 validation run
        # showed why that's wrong: probe_batch_size() then runs against an
        # already-compiled model, and since it deliberately tries many
        # DIFFERENT batch sizes (that's the whole point of the probe), every
        # new shape triggered its own fresh, expensive recompilation --
        # polluting the probe's peak-VRAM measurements (compile-time
        # autotuning buffers count toward "peak") and crushing the
        # recommended batch size (64 instead of something much larger).

    def benchmark_and_maybe_compile(self, batch_size, n_measure=3):
        """Compile ONCE, at the single final batch size the real run will
        actually use (called after probe_batch_size(), never before/during
        it) -- and only KEEP the compiled model if timing a few real forward
        passes shows it's actually faster than eager at that exact shape.
        `torch.compile()` not raising an exception is not the same as it
        helping: a real T4 validation run showed a "successful" compile
        (with `use_torch_compile=True` at load time, the old design) that
        was silently degraded the whole time ("Tesla T4 does not support
        bfloat16 compilation natively, skipping" / "Not enough SMs to use
        max_autotune_gemm mode") and made throughput *worse*, not better --
        no exception was ever raised, so a canary-only guard (catch errors,
        nothing else) can't detect that failure mode at all. This measures
        instead of assuming, on whatever GPU the run actually lands on."""
        import time
        torch = self._torch
        if not torch.cuda.is_available():
            print("[GIGA] no GPU -- skipping torch.compile", flush=True)
            return

        probe_text = "товар " * (self.MAX_SEQ_LENGTH * 2)
        enc = self.tokenizer(
            [probe_text] * batch_size, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.MAX_SEQ_LENGTH,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        def _timed_forward(model, n):
            with torch.no_grad():
                model(**enc, return_embeddings=True)  # warmup / triggers compilation
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(n):
                    model(**enc, return_embeddings=True)
            torch.cuda.synchronize()
            return (time.time() - t0) / n

        try:
            eager_secs = _timed_forward(self.model, n_measure)
            print(f"[GIGA] eager: {eager_secs * 1000:.0f}ms/batch @ batch_size={batch_size}", flush=True)
        except Exception as e:
            print(f"[GIGA] eager benchmark failed ({type(e).__name__}: {e}) -- "
                  f"skipping compile decision, staying eager", flush=True)
            return

        try:
            compiled = torch.compile(self.model)
            compiled_secs = _timed_forward(compiled, n_measure)
            print(f"[GIGA] compiled: {compiled_secs * 1000:.0f}ms/batch @ batch_size={batch_size}", flush=True)
            if compiled_secs < eager_secs * 0.95:
                self.model = compiled
                self._compiled = True
                print(f"[GIGA] compile wins ({compiled_secs * 1000:.0f}ms < {eager_secs * 1000:.0f}ms) "
                      f"-- using compiled model", flush=True)
            else:
                print(f"[GIGA] compile does not win ({compiled_secs * 1000:.0f}ms vs "
                      f"{eager_secs * 1000:.0f}ms eager) -- staying eager", flush=True)
        except Exception as e:
            print(f"[GIGA] torch.compile failed ({type(e).__name__}: {e}) -- staying eager", flush=True)

        del enc
        torch.cuda.empty_cache()

    def probe_batch_size(self, start=64, cap=8192, round_to=32, safety_margin_gb=2.0):
        """Grow-to-real-OOM then binary-search the exact boundary (same
        algorithm as kaggle_kernels/frida-batch-probe's run_vram_probe,
        adapted to a plain inference forward pass instead of forward+
        backward) -- so the batch size actually used is measured on
        whatever GPU this run lands on (the judge stand's H100), not a
        hand-derived guess extrapolated from a T4 that this project already
        flagged as unverified. Every probe step tokenizes at
        padding="max_length", so its (batch, MAX_SEQ_LENGTH) shape is
        byte-identical to a real batch's -- a faithful memory measurement,
        not an approximation. Returns the recommended batch size; if no GPU
        is present this probe can't mean anything and just returns `start`."""
        torch = self._torch
        if not torch.cuda.is_available():
            print("[PROBE] no GPU available -- skipping, using start as a conservative default", flush=True)
            return start

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # Repeated filler guaranteed to truncate to exactly MAX_SEQ_LENGTH
        # tokens, so every probe step's real shape matches a real batch's
        # worst case (no padding, full sequence length).
        probe_text = "товар " * (self.MAX_SEQ_LENGTH * 2)

        def _is_oom(exc):
            return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

        def _probe_step(bs):
            enc = self.tokenizer(
                [probe_text] * bs, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.MAX_SEQ_LENGTH,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                self.model(**enc, return_embeddings=True)
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            del enc
            torch.cuda.empty_cache()
            return peak_gb

        print(f"[PROBE] GPU total VRAM: {total_vram_gb:.1f}GB -- doubling from batch={start} "
              f"until a real OOM ...", flush=True)
        candidate = start
        last_good, last_good_gb, oom_at = None, None, None
        while candidate <= cap:
            try:
                peak_gb = _probe_step(candidate)
                print(f"[PROBE] batch={candidate:5d} -> peak={peak_gb:.2f}GB "
                      f"({100 * peak_gb / total_vram_gb:.0f}% of {total_vram_gb:.1f}GB)", flush=True)
                last_good, last_good_gb = candidate, peak_gb
                candidate *= 2
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                print(f"[PROBE] batch={candidate:5d} -> real OOM (ceiling doubling found)", flush=True)
                oom_at = candidate
                break

        if oom_at is None:
            print(f"[PROBE] never hit OOM up to cap={cap} -- this GPU has a lot of headroom "
                  f"at MAX_SEQ_LENGTH={self.MAX_SEQ_LENGTH}, or cap should be raised", flush=True)
            true_ceiling, ceiling_gb = last_good, last_good_gb
        else:
            lo, hi = last_good, oom_at
            while hi - lo > round_to:
                mid = ((lo + hi) // 2 // round_to) * round_to
                if mid <= lo:
                    break
                try:
                    peak_gb = _probe_step(mid)
                    print(f"[PROBE] batch={mid:5d} -> peak={peak_gb:.2f}GB (fits)", flush=True)
                    lo, last_good, last_good_gb = mid, mid, peak_gb
                except RuntimeError as e:
                    if not _is_oom(e):
                        raise
                    torch.cuda.empty_cache()
                    print(f"[PROBE] batch={mid:5d} -> OOM", flush=True)
                    hi = mid
            true_ceiling, ceiling_gb = last_good, last_good_gb

        if true_ceiling is None:
            print(f"[PROBE] even start={start} OOM'd immediately -- falling back to "
                  f"batch_size={max(4, start // 4)}", flush=True)
            return max(4, start // 4)

        target_gb = total_vram_gb - safety_margin_gb
        if ceiling_gb <= target_gb:
            recommended = true_ceiling
        else:
            scale = target_gb / ceiling_gb
            recommended = max(round_to, (int(true_ceiling * scale) // round_to) * round_to)
        print(f"[PROBE] true ceiling batch={true_ceiling} (peak {ceiling_gb:.2f}GB) -> "
              f"recommended batch_size={recommended} (with {safety_margin_gb:.1f}GB safety margin)", flush=True)
        return recommended

    def tokenize_batch(self, texts):
        enc = self.tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.MAX_SEQ_LENGTH,
        )
        if self.device.type == "cuda":
            enc = {k: v.pin_memory() for k, v in enc.items()}
        return enc

    def forward_batch(self, encoded):
        torch = self._torch
        enc = {k: v.to(self.device, non_blocking=True) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.model(**enc, return_embeddings=True)
        return out.detach().float().cpu().numpy()

    def encode_pairs(self, texts, batch_size):
        n = len(texts)
        result = None
        i = 0
        while i < n:
            batch = texts[i:i + batch_size]
            emb = self.forward_batch(self.tokenize_batch(batch))
            if result is None:
                import numpy as np
                result = np.empty((n, emb.shape[1]), dtype="float32")
            result[i:i + len(batch)] = emb
            i += len(batch)
        return result


def maybe_offline_install(wheels_dir="wheels"):
    """bitsandbytes/einops are almost certainly absent from the baseline
    image (the organizer's own baseline never imports them). Also pins
    transformers/sentence-transformers versions: modeling_gigarembed.py uses
    transformers' ALL_ATTENTION_FUNCTIONS registry, a relatively recent
    internal API. Best-effort: offline pip can't fetch further transitive
    deps, so on failure we log and continue with whatever's already in the
    image rather than hard-failing the submission.
    Must be called before anything imports transformers (this module's own
    heavy imports are deferred into __init__ for that reason).
    """
    if not os.path.isdir(wheels_dir):
        print(f"[SETUP] no wheels dir at {wheels_dir}, skipping offline install", flush=True)
        return
    import subprocess, sys
    pins = ["einops", "bitsandbytes", "accelerate", "transformers==4.51.0", "sentence-transformers==5.1.1"]
    print(f"[SETUP] offline-installing {pins} from {wheels_dir}", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-index",
             "--find-links", wheels_dir, *pins]
        )
        print("[SETUP] offline install ok", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[SETUP] offline install failed ({e}); continuing with whatever transformers/"
              f"bitsandbytes/einops are already in the image", flush=True)
