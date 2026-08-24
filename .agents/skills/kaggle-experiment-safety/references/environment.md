# Kaggle platform notes for this project

Facts about the platform, the images, the CLI, and the accounts. These are
things you cannot derive from the code — each one has cost a run at least once.

## Quotas and caps

| | Weekly pool | Hard per-session cap |
|---|---|---|
| GPU | 30h | **12h** |
| TPU | 20h (separate pool) | **9h** |
| CPU kernels | not metered against either pool | — |

The session cap is hard: the run is cancelled mid-step, `CANCEL_ACKNOWLEDGED`.
It does not extend, warn, or negotiate. The two pools are independent, which is
why RAM-heavy CPU work (tokenization, large joins) is deliberately run on TPU
sessions here — no TPU compute is used, but the allocation is larger and the
hours do not come out of the training budget.

A kernel cancelled while still `QUEUED` costs **zero**, so catching a mistake
before it starts running is fully free.

## Paths

- Input mount carries an extra path segment:
  `/kaggle/input/datasets/<owner>/<slug>/...` — **not** `/kaggle/input/<slug>/...`.
- `/kaggle/working` is the output directory and has a **~20GB ceiling**
  (a real precheck saw `have 20.94GB`). It **persists as kernel output even
  when the session is hard-capped or cancelled**, which is what makes recovery
  via `kaggle kernels output` possible.
- The kernel **log** has no such guarantee: it is truncated to the tail on long
  runs, and one cancellation produced a 0-byte log. Mirror anything important
  into a JSON file under `/kaggle/working`.

## Image differences — GPU/CPU vs TPU

### Реальные host-профили kernel-сессий

| Тип kernel | CPU | Host RAM |
|---|---:|---:|
| TPU v5e-8 | **224 vCPU** | **384 ГБ** |
| CPU-only | **4 ядра** | **30 ГБ** |
| GPU T4 x2 | **4 ядра** | **29 ГБ** |

Переменные `OMP_NUM_THREADS`, `MKL_NUM_THREADS` и `RAYON_NUM_THREADS`
задают число потоков **на процесс**, а не на всю VM. Для обычного
8-процессного TPU v5e-8 kernel безопасная исходная настройка — `28`
(`224 / 8`), а не `224` на каждый rank. Для CPU-only и T4 x2 исходное
значение — `4`, если измерения конкретного workload не требуют меньшего.

The TPU image is not the standard image with an accelerator swapped in. Four
separate incidents came from assuming it was:

| | CPU/GPU image | TPU image |
|---|---|---|
| `polars` | preinstalled | **absent** |
| `kaggle` CLI binary | on PATH | **absent** |
| `transformers`/`tokenizers` | mutually compatible | **mismatched** for `RobertaTokenizer` |

Consequences worth internalizing:

- **`pip install` must run before the module-level `import`** that needs the
  package. Top-level imports execute at module load, before `main()` is ever
  called, so an install placed inside `main()` is already too late.
- **Pin exact versions, not `-U`.** An unpinned `pip install -U transformers
  tokenizers` resolved a *pre-release* `tokenizers==0.23.0rc0` on the TPU image
  and reproduced the exact same `TypeError: RobertaProcessing.__new__() got an
  unexpected keyword argument 'cls'` the upgrade was meant to fix — while the
  same command on the CPU image happened to resolve a working pair. "It worked
  on the CPU image" proves nothing about the TPU image.
- Log `transformers.__version__` / `tokenizers.__version__` on every run; that
  is what surfaced the divergence.
- The cheapest fix of all is **not calling the tokenizer at inside a training
  kernel**: reuse a published token cache and derive `pad_id` from the cache
  itself. That removed this entire failure class from `frida-tpu-human-sft`.

## `kernel-metadata.json`

```jsonc
{
  "id": "<account>/<slug>",        // owner must match the account you push as
  "code_file": "<script>.py",
  "kernel_type": "script",
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["owner/slug", "..."],
  "machine_shape": "NvidiaTeslaT4"  // GPU kernels: omit at your peril
}
```

- **`machine_shape` for GPU kernels.** Omitting it once got a GPU the
  preinstalled torch build had no compiled kernels for; the run died at the
  first CUDA op with
  `torch.AcceleratorError: CUDA error: no kernel image is available for
  execution on the device`. Every other successful GPU kernel in the repo sets
  it. TPU kernels do not need it.
- **`dataset_sources`** is the only thing that needs editing when a checkpoint
  moves, because resume code searches `/kaggle/input` by file content
  (`adapter_config.json`, `train_state.json`) rather than by dataset name.
- **Kernel secrets** (`KAGGLE_USERNAME`/`KAGGLE_KEY` for a kernel that
  self-publishes a dataset) are set per kernel in the Kaggle **web UI** →
  Settings → Secrets. They cannot be configured from `kernel-metadata.json` or
  the CLI push.

## Datasets

- `dataset-metadata.json` requires a **`licenses`** key for
  `kaggle datasets create` — the CLI calls `get_or_fail(meta_data, "licenses")`
  and fails *after* your work is done. Use
  `"licenses": [{"name": "CC0-1.0"}]`, matching what `kaggle datasets init`
  templates.
- `kaggle datasets version -r <skip|zip|tar>` has returned
  `Incompatible Dataset Type` against existing datasets here, in every mode
  tried, uploading real bandwidth each time before failing without committing.
  Creating a new dataset and repointing `dataset_sources` is the working path.

## Accounts

Three accounts, three separate quota pools. Confirm which one a run should
spend before launching rather than inferring it.

| Account | Creds | Scope |
|---|---|---|
| `maxamsecret` | `.kaggle/kaggle_maxamsecret/.kaggle/kaggle.json` | owns the source dataset; FRIDA lineage |
| `rays0n` | `.kaggle/kaggle_rays0n/.kaggle/kaggle.json` | BERTA lineage |
| `maximmarkovbb` | `.kaggle/kaggle_maximmarkovbb/kaggle.json` (**flat, no nested `.kaggle/`**) | FRIDA v2; public by convention |

Updated 2026-08-22: `~/.kaggle/` moved from flat per-account files
(`kaggle.json`, `kaggle_rays0n.json`, ...) to one subdirectory per account —
**including the default account**, `maxamsecret`. The old bare
`~/.kaggle/kaggle.json` no longer exists at all, so a plain `kaggle ...` with
no override now fails auth for every account, default included. The old
`KAGGLE_USERNAME=... KAGGLE_KEY=$(python -c "...json.load(open(...))...")`
override pattern is also dead (`FileNotFoundError` on the old flat path).
Current working pattern, for ANY account including the default:

```bash
KAGGLE_CONFIG_DIR="C:/Users/mba90/.kaggle/kaggle_{account}/.kaggle" kaggle kernels status {account}/<slug>
# maximmarkovbb is the one exception -- flat, no nested .kaggle/:
KAGGLE_CONFIG_DIR="C:/Users/mba90/.kaggle/kaggle_maximmarkovbb" kaggle kernels status maximmarkovbb/<slug>
```

More accounts than these three may exist on disk at any given time (a
`check_all_quotas.py` script under the `kaggle` skill's own setup module
discovers whatever `kaggle_*/` directories are actually present, rather than
assuming this fixed list) — confirm which account a run should spend before
launching rather than inferring it from this table alone.

## CLI lifecycle

```bash
kaggle kernels push   -p kaggle_kernels/<name>
kaggle kernels status <owner>/<slug>
kaggle kernels output <owner>/<slug> -p kaggle_kernels/<name>/output
```

Statuses seen here: `QUEUED` → `RUNNING` → `COMPLETE` / `ERROR` /
`CANCEL_ACKNOWLEDGED` (both a user cancel and a session-cap kill land on the
last one). `kaggle kernels output` returns nothing while a kernel is still
running, so it is not a progress check — use `status`, and read the run's own
JSON sidecars once it finishes.

Check `status` shortly after every push. Confirming the kernel reached
`RUNNING` rather than erroring in its first minute costs one command and
catches the cheapest class of failure while it is still cheap.
