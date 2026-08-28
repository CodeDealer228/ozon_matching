"""Controlled classifier-head sanity check for the 195-row A2 dataset.

The exact submitted FRIDA encoder stays frozen.  Two classifier heads start
from the exact submitted head and receive the same replay rows, number of
steps, weights, ordering and random seed:

* control: repeats each original human-negative source pair;
* A2: repeats the corresponding identity-safe rewritten pair.

The only experimental difference is original B versus rewritten B.  The
untouched canonical validation is used exclusively for evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path


SUBMISSION_URL = (
    "https://storage.yandexcloud.net/ds-ods/files/submissions/"
    "0d0183cf-8864-4331-88be-f78da0c68dd2/8d18ca15/"
    "matching-frida_continue_c_llm3m_188-submit.zip"
)
SUBMISSION_ZIP_NAME = "matching-frida_continue_c_llm3m_188-submit.zip"
EXPECTED_SUBMISSION_SIZE = 1_309_768_430
EXPECTED_SUBMISSION_SHA256 = (
    "30bfc85d44785d44f9316a0b4dda2cb7dfdb7c51e3d3e68c89eb9a5ef064b9a1"
)
EXPECTED_MODEL_SHA256 = (
    "423c9772de237fb9cc1d797b649dff805b547cb9bdf607ef0616c9ab1c7d0591"
)
EXPECTED_CLASSIFIER_SHA256 = (
    "e661edb385e9ea846e8a8783faec1244e2294f99f56b9e090dd8b7649effb771"
)

SPLIT_NAME = "matches_item_disjoint_70_30_seed42.parquet"
EXPECTED_SPLIT_SIZE = 6_192_178
EXPECTED_SPLIT_SHA256 = (
    "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
)
EXPECTED_VAL_FINGERPRINT = (
    "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
)
EXPECTED_N_TRAIN = 256_006
EXPECTED_N_VAL = 109_648

EXPECTED_A2_ROWS = 195
EXPECTED_A2_SELECTED_SHA256 = (
    "fb65c177506e2e272515999e01bc5eafe3c0e6fecd72eca800e540b2590db2c2"
)
EXPECTED_A2_PAIRS_SHA256 = (
    "8e5c8d9f57286a33d1ee70e984c837a62b4abda943ffd08b7cb682fb0ce4cc19"
)
EXPECTED_A2_ITEMS_SHA256 = (
    "296153040f48929e8ee6483ff53f5c8bc4e4ef55b85b40df89027c29d44476fb"
)

SEED = 20260825
REPLAY_PER_CATEGORY_LABEL = 250
A2_REPEAT = 10
A2_SAMPLE_WEIGHT = 0.5
HEAD_LR = 5e-5
HEAD_WEIGHT_DECAY = 0.01
HEAD_BATCH_SIZE = 512
HEAD_EPOCHS = 1
ENCODER_BATCH_SIZE = 48
MAX_RUNTIME_SECONDS = 10.5 * 60 * 60
BASELINE_MACRO_AP = 0.7588808815558529
BASELINE_MICRO_AP = 0.8132819127479105

OUTPUT_DIR = Path("/kaggle/working")
TEMP_DIR = Path("/kaggle/temp/frida-a2-head-ablation")
INPUT_ROOT = Path("/kaggle/input")


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=json_default)
        stream.write("\n")
    os.replace(temporary, path)


def update_status(phase: str, **extra) -> None:
    payload = {"phase": phase, "updated_at_unix": time.time(), **extra}
    write_json(OUTPUT_DIR / "run_status.json", payload)
    print(f"[STATUS] {phase}: {extra}", flush=True)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def find_input_file(filename: str) -> Path | None:
    matches = sorted(INPUT_ROOT.rglob(filename)) if INPUT_ROOT.exists() else []
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[INPUT] {filename}: {len(matches)} copies, using {matches[0]}", flush=True)
    return matches[0]


def require_input_file(filename: str) -> Path:
    path = find_input_file(filename)
    if path is None:
        raise FileNotFoundError(filename)
    return path


def download_submission() -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    destination = TEMP_DIR / SUBMISSION_ZIP_NAME
    if destination.is_file():
        if (
            destination.stat().st_size == EXPECTED_SUBMISSION_SIZE
            and sha256_file(destination) == EXPECTED_SUBMISSION_SHA256
        ):
            return destination
        destination.unlink()
    partial = destination.with_suffix(".zip.part")
    request = urllib.request.Request(SUBMISSION_URL, headers={"User-Agent": "frida-a2/1.0"})
    downloaded = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        reported = response.headers.get("Content-Length")
        if reported and int(reported) != EXPECTED_SUBMISSION_SIZE:
            raise RuntimeError(f"Submission Content-Length mismatch: {reported}")
        while True:
            chunk = response.read(16 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            downloaded += len(chunk)
            if downloaded % (128 * 1024 * 1024) < len(chunk):
                print(f"[MODEL] downloaded {downloaded:,}/{EXPECTED_SUBMISSION_SIZE:,}", flush=True)
    os.replace(partial, destination)
    if (
        destination.stat().st_size != EXPECTED_SUBMISSION_SIZE
        or sha256_file(destination) != EXPECTED_SUBMISSION_SHA256
    ):
        raise RuntimeError("Downloaded submission identity mismatch")
    return destination


def extract_submission(archive: Path) -> Path:
    extract_dir = TEMP_DIR / "submission"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive member: {bad}")
        root = extract_dir.resolve()
        for member in bundle.infolist():
            target = (extract_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(extract_dir)
    models = list(extract_dir.rglob("model.safetensors"))
    if len(models) != 1:
        raise RuntimeError(f"Expected one model.safetensors, got {models}")
    submission_root = models[0].parent.parent.parent
    model_path = models[0]
    classifier_path = submission_root / "classifier.pt"
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Exact model SHA-256 mismatch")
    if sha256_file(classifier_path) != EXPECTED_CLASSIFIER_SHA256:
        raise RuntimeError("Exact classifier SHA-256 mismatch")
    return submission_root


def load_utils(submission_root: Path):
    sys.path.insert(0, str(submission_root))
    for name in ("src.utils", "src"):
        sys.modules.pop(name, None)
    return importlib.import_module("src.utils")


def fingerprint_pairs(frame) -> str:
    payload = "\n".join(
        sorted(f"{int(left)}-{int(right)}" for left, right in zip(frame.id1, frame.id2))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_and_verify_data():
    import numpy as np
    import pandas as pd

    split_path = require_input_file(SPLIT_NAME)
    if split_path.stat().st_size != EXPECTED_SPLIT_SIZE:
        raise RuntimeError("Split size mismatch")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Split SHA-256 mismatch")
    split = pd.read_parquet(split_path)
    train = split[split.split.eq("train")].reset_index(drop=True)
    validation = split[split.split.eq("val")].reset_index(drop=True)
    if len(train) != EXPECTED_N_TRAIN or len(validation) != EXPECTED_N_VAL:
        raise RuntimeError("Split row count mismatch")
    if fingerprint_pairs(validation) != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Validation fingerprint mismatch")
    train_ids = set(np.concatenate((train.id1.to_numpy(), train.id2.to_numpy())).tolist())
    val_ids = set(
        np.concatenate((validation.id1.to_numpy(), validation.id2.to_numpy())).tolist()
    )
    if train_ids & val_ids:
        raise RuntimeError("Train/validation item overlap")

    selected_path = require_input_file("selected_a2.parquet")
    pairs_path = require_input_file("synthetic_pairs.parquet")
    items_path = require_input_file("synthetic_items.parquet")
    if sha256_file(selected_path) != EXPECTED_A2_SELECTED_SHA256:
        raise RuntimeError("selected_a2 SHA-256 mismatch")
    if sha256_file(pairs_path) != EXPECTED_A2_PAIRS_SHA256:
        raise RuntimeError("synthetic_pairs SHA-256 mismatch")
    if sha256_file(items_path) != EXPECTED_A2_ITEMS_SHA256:
        raise RuntimeError("synthetic_items SHA-256 mismatch")
    selected = pd.read_parquet(selected_path)
    synthetic_pairs = pd.read_parquet(pairs_path)
    synthetic_items = pd.read_parquet(items_path)
    if not (len(selected) == len(synthetic_pairs) == len(synthetic_items) == EXPECTED_A2_ROWS):
        raise RuntimeError("A2 row count mismatch")
    original_ids = set(selected.id1.astype(int)) | set(selected.id2.astype(int))
    if original_ids - train_ids or original_ids & val_ids:
        raise RuntimeError("A2 source identity/leak check failed")
    if set(synthetic_pairs.label.astype(int)) != {0}:
        raise RuntimeError("A2 labels must all be zero")
    if set(synthetic_pairs.id2.astype(int)) != set(synthetic_items.id.astype(int)):
        raise RuntimeError("Synthetic pair/item ids differ")
    return train, validation, selected, synthetic_pairs, synthetic_items


def build_replay(train):
    import pandas as pd

    parts = []
    for (_category, _label), group in train.groupby(["category", "label"], sort=True):
        if len(group) < REPLAY_PER_CATEGORY_LABEL:
            raise RuntimeError("Replay stratum too small")
        parts.append(group.sample(n=REPLAY_PER_CATEGORY_LABEL, random_state=SEED))
    replay = pd.concat(parts, ignore_index=True)
    replay = replay.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return replay


def synthetic_texts(utils, synthetic_items) -> dict[int, str]:
    output = {}
    for row in synthetic_items.itertuples(index=False):
        name = str(row.name or "")[: utils.MAX_NAME_CHARS]
        category = str(row.category or "")[: utils.MAX_CATEGORY_CHARS]
        attributes = utils._reorder_and_slice(str(row.attributes or ""))
        text = f"Name: {name} Category: {category} Attributes: {attributes}"
        output[int(row.id)] = text[: utils.MAX_ITEM_CHARS]
    return output


def encode_pairs(name, frame, id_to_item, tokenizer, encoder, device):
    import numpy as np
    import torch

    utils = sys.modules["src.utils"]
    input_ids, attention_mask = utils._build_pairs(frame, id_to_item, tokenizer.pad_token_id)
    if (attention_mask.sum(axis=1) == 0).any():
        raise RuntimeError(f"{name} contains all-pad inputs")
    n = len(frame)
    embeddings = np.empty((n, encoder.config.d_model), dtype=np.float32)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, n, ENCODER_BATCH_SIZE):
            end = min(start + ENCODER_BATCH_SIZE, n)
            valid = end - start
            ids = input_ids[start:end]
            mask = attention_mask[start:end]
            if valid < ENCODER_BATCH_SIZE:
                ids = np.concatenate((ids, np.repeat(ids[-1:], ENCODER_BATCH_SIZE - valid, axis=0)))
                mask = np.concatenate((mask, np.repeat(mask[-1:], ENCODER_BATCH_SIZE - valid, axis=0)))
            ids_t = torch.from_numpy(ids).to(device, non_blocking=True)
            mask_t = torch.from_numpy(mask).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                cls = encoder(input_ids=ids_t, attention_mask=mask_t).last_hidden_state[:, 0, :]
            embeddings[start:end] = cls[:valid].float().cpu().numpy()
            if start == 0 or end == n or end % max(ENCODER_BATCH_SIZE, n // 5) < ENCODER_BATCH_SIZE:
                print(f"[ENCODE] {name}: {end:,}/{n:,} in {time.time()-started:.1f}s", flush=True)
    del input_ids, attention_mask
    return embeddings


def load_head(utils, classifier_path: Path, device):
    import torch

    state_dict = torch.load(classifier_path, map_location="cpu")
    hidden_dim = int(state_dict["blocks.0.0.weight"].shape[1])
    head = utils.FullSftHead(hidden_dim, layer_sizes=(1024, 512, 128), dropout=0.2)
    head.load_state_dict(state_dict)
    return head.float().to(device)


def predict_head(head, features, device):
    import numpy as np
    import torch

    head.eval()
    output = np.empty(len(features), dtype=np.float32)
    batch_size = 4096
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            x = torch.from_numpy(features[start:end]).to(device)
            output[start:end] = torch.sigmoid(head(x).float()).cpu().numpy()
    return output


def metric_payload(validation, probabilities):
    import numpy as np
    from sklearn.metrics import average_precision_score

    rows = []
    for category, group in validation.assign(score=probabilities).groupby("category", sort=True):
        ap = float(average_precision_score(group.label.astype(int), group.score))
        rows.append({"category": str(category), "rows": int(len(group)), "average_precision": ap})
    return {
        "macro_average_precision": float(np.mean([row["average_precision"] for row in rows])),
        "micro_average_precision": float(
            average_precision_score(validation.label.astype(int), probabilities)
        ),
        "per_category": rows,
    }


def train_head(track, utils, classifier_path, replay_embeddings, replay_labels, extra_embeddings, device):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    extras = np.repeat(extra_embeddings, A2_REPEAT, axis=0)
    features = np.concatenate((replay_embeddings, extras), axis=0).astype(np.float32)
    labels = np.concatenate(
        (replay_labels.astype(np.float32), np.zeros(len(extras), dtype=np.float32))
    )
    weights = np.concatenate(
        (
            np.ones(len(replay_embeddings), dtype=np.float32),
            np.full(len(extras), A2_SAMPLE_WEIGHT, dtype=np.float32),
        )
    )
    dataset = TensorDataset(
        torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(weights)
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    head = load_head(utils, classifier_path, device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    head.train()
    losses = []
    for _epoch in range(HEAD_EPOCHS):
        for x, y, weight in loader:
            x, y, weight = x.to(device), y.to(device), weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            raw = functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (raw * weight).sum() / weight.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    checkpoint_dir = OUTPUT_DIR / f"checkpoint_{track}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    classifier_out = checkpoint_dir / "classifier.pt"
    torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, classifier_out)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": HEAD_EPOCHS,
            "mean_train_loss": float(np.mean(losses)),
            "validation_metric": None,
        },
        checkpoint_dir / "optimizer_state.pt",
    )
    write_json(
        checkpoint_dir / "train_state.json",
        {
            "track": track,
            "epochs": HEAD_EPOCHS,
            "rows": len(dataset),
            "mean_train_loss": float(np.mean(losses)),
            "validation_metric": None,
        },
    )
    return head, classifier_out, float(np.mean(losses))


def save_predictions(validation, name, probabilities):
    import numpy as np

    frame = validation[["id1", "id2", "label", "category"]].copy()
    frame["score"] = probabilities
    frame.to_parquet(OUTPUT_DIR / f"val_predictions_{name}.parquet", index=False)
    np.save(OUTPUT_DIR / f"val_predictions_{name}.npy", probabilities)


def build_submission(submission_root: Path, trained_classifier: Path) -> tuple[Path, dict]:
    output = OUTPUT_DIR / "matching-frida-a2-head-ablation-submit.zip"
    root_name = submission_root.name
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(p for p in submission_root.rglob("*") if p.is_file()):
            relative = path.relative_to(submission_root)
            source = trained_classifier if relative.as_posix() == "classifier.pt" else path
            bundle.write(source, f"{root_name}/{relative.as_posix()}")
    with zipfile.ZipFile(output) as bundle:
        bad = bundle.testzip()
        names = bundle.namelist()
        expected_classifier = f"{root_name}/classifier.pt"
        if bad or expected_classifier not in names:
            raise RuntimeError(f"Submission archive failed smoke test: bad={bad}")
        classifier_bytes = bundle.read(expected_classifier)
    return output, {
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "entries": len(names),
        "classifier_sha256": hashlib.sha256(classifier_bytes).hexdigest(),
        "zip_test": "ok",
    }


def main() -> None:
    started = time.time()
    deadline = started + MAX_RUNTIME_SECONDS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    update_status("preflight")
    write_json(OUTPUT_DIR / "write_probe.json", {"ok": True})
    (OUTPUT_DIR / "write_probe.json").unlink()
    probe = OUTPUT_DIR / "zip_probe.zip"
    with zipfile.ZipFile(probe, "w") as bundle:
        bundle.writestr("ok.txt", "ok")
    with zipfile.ZipFile(probe) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Zip teardown preflight failed")
    probe.unlink()

    import numpy as np
    import pandas as pd
    import pyarrow
    import sklearn
    import torch
    import transformers
    from transformers import T5EncoderModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 7:
        raise RuntimeError(f"Unsupported GPU capability: {capability}")
    preflight = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": device_name,
        "gpu_capability": capability,
        "output_free_bytes": shutil.disk_usage(OUTPUT_DIR).free,
        "temp_free_bytes": shutil.disk_usage(TEMP_DIR).free,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
    }
    if preflight["output_free_bytes"] < 4_000_000_000:
        raise RuntimeError("Insufficient output disk for checkpoint and submission")
    write_json(OUTPUT_DIR / "preflight.json", preflight)

    train, validation, selected, synthetic_pairs, synthetic_items = load_and_verify_data()
    replay = build_replay(train)
    control_pairs = selected[["id1", "id2", "label", "category"]].copy()
    a2_pairs = synthetic_pairs[["id1", "id2", "label", "category"]].copy()
    replay_fingerprint = hashlib.sha256(
        "\n".join(f"{int(a)}-{int(b)}" for a, b in zip(replay.id1, replay.id2)).encode()
    ).hexdigest()
    write_json(
        OUTPUT_DIR / "data_identity.json",
        {
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "replay_rows": len(replay),
            "replay_fingerprint": replay_fingerprint,
            "a2_rows": len(a2_pairs),
            "a2_selected_sha256": EXPECTED_A2_SELECTED_SHA256,
            "a2_pairs_sha256": EXPECTED_A2_PAIRS_SHA256,
            "a2_items_sha256": EXPECTED_A2_ITEMS_SHA256,
            "validation_used_as_training": False,
        },
    )
    update_status("data_verified", replay_rows=len(replay), a2_rows=len(a2_pairs))

    archive = download_submission()
    submission_root = extract_submission(archive)
    utils = load_utils(submission_root)
    model_dir = submission_root / "models/frida-full-sft"
    classifier_path = submission_root / "classifier.pt"
    tokenizer = utils._load_tokenizer(model_dir)

    needed_original_ids = set(
        np.concatenate(
            (
                validation.id1.to_numpy(), validation.id2.to_numpy(),
                replay.id1.to_numpy(), replay.id2.to_numpy(),
                control_pairs.id1.to_numpy(), control_pairs.id2.to_numpy(),
            )
        ).tolist()
    )
    items_human_path = require_input_file("items_human.parquet")
    id_to_text = utils._load_needed_item_texts(items_human_path, needed_original_ids)
    if set(id_to_text) != needed_original_ids:
        raise RuntimeError("Missing original item text")
    id_to_text.update(synthetic_texts(utils, synthetic_items))
    id_to_item = utils._tokenize_items(tokenizer, id_to_text)
    update_status("items_tokenized", unique_items=len(id_to_item))

    try:
        encoder = T5EncoderModel.from_pretrained(
            model_dir, attn_implementation="eager", torch_dtype=torch.float16
        ).to("cuda")
    except (TypeError, ValueError):
        encoder = T5EncoderModel.from_pretrained(
            model_dir, torch_dtype=torch.float16
        ).to("cuda")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    validation_embeddings = encode_pairs(
        "validation", validation, id_to_item, tokenizer, encoder, "cuda"
    )
    replay_embeddings = encode_pairs(
        "replay", replay, id_to_item, tokenizer, encoder, "cuda"
    )
    control_embeddings = encode_pairs(
        "control_extras", control_pairs, id_to_item, tokenizer, encoder, "cuda"
    )
    a2_embeddings = encode_pairs(
        "a2_extras", a2_pairs, id_to_item, tokenizer, encoder, "cuda"
    )
    del encoder, tokenizer, id_to_item, id_to_text
    torch.cuda.empty_cache()
    update_status("embeddings_complete")

    baseline_head = load_head(utils, classifier_path, "cuda")
    baseline_predictions = predict_head(baseline_head, validation_embeddings, "cuda")
    baseline_metrics = metric_payload(validation, baseline_predictions)
    save_predictions(validation, "baseline", baseline_predictions)
    del baseline_head
    if abs(baseline_metrics["macro_average_precision"] - BASELINE_MACRO_AP) > 0.01:
        raise RuntimeError(f"Baseline reproduction drift too large: {baseline_metrics}")
    update_status("baseline_evaluated", **baseline_metrics)

    replay_labels = replay.label.to_numpy(dtype=np.float32)
    control_head, control_classifier, control_loss = train_head(
        "control", utils, classifier_path, replay_embeddings, replay_labels,
        control_embeddings, "cuda"
    )
    control_predictions = predict_head(control_head, validation_embeddings, "cuda")
    control_metrics = metric_payload(validation, control_predictions)
    save_predictions(validation, "control", control_predictions)
    write_json(
        OUTPUT_DIR / "checkpoint_control/train_state.json",
        {
            "track": "control", "epochs": HEAD_EPOCHS,
            "mean_train_loss": control_loss,
            "validation_metric": control_metrics,
        },
    )
    del control_head
    torch.cuda.empty_cache()
    update_status("control_evaluated", **control_metrics)

    a2_head, a2_classifier, a2_loss = train_head(
        "a2", utils, classifier_path, replay_embeddings, replay_labels,
        a2_embeddings, "cuda"
    )
    a2_predictions = predict_head(a2_head, validation_embeddings, "cuda")
    a2_metrics = metric_payload(validation, a2_predictions)
    save_predictions(validation, "a2", a2_predictions)
    write_json(
        OUTPUT_DIR / "checkpoint_a2/train_state.json",
        {
            "track": "a2", "epochs": HEAD_EPOCHS,
            "mean_train_loss": a2_loss,
            "validation_metric": a2_metrics,
        },
    )
    del a2_head
    torch.cuda.empty_cache()
    update_status("a2_evaluated", **a2_metrics)

    comparison = {
        "status": "complete",
        "experiment": "frozen_encoder_head_only_control_vs_a2",
        "baseline_reference": {
            "macro_average_precision": BASELINE_MACRO_AP,
            "micro_average_precision": BASELINE_MICRO_AP,
        },
        "baseline_same_gpu": baseline_metrics,
        "control": control_metrics,
        "a2": a2_metrics,
        "a2_minus_control_macro": (
            a2_metrics["macro_average_precision"] - control_metrics["macro_average_precision"]
        ),
        "a2_minus_control_micro": (
            a2_metrics["micro_average_precision"] - control_metrics["micro_average_precision"]
        ),
        "a2_minus_original_macro": (
            a2_metrics["macro_average_precision"] - baseline_metrics["macro_average_precision"]
        ),
        "a2_minus_original_micro": (
            a2_metrics["micro_average_precision"] - baseline_metrics["micro_average_precision"]
        ),
        "a2_improved_over_control": bool(
            a2_metrics["macro_average_precision"] > control_metrics["macro_average_precision"]
        ),
    }
    write_json(OUTPUT_DIR / "metrics_comparison.json", comparison)

    if time.time() >= deadline:
        raise RuntimeError("Deadline reached before submission packaging")
    submission_zip, submission_identity = build_submission(submission_root, a2_classifier)
    write_json(OUTPUT_DIR / "submission_identity.json", submission_identity)
    training_meta = {
        "status": "complete",
        "seed": SEED,
        "encoder_frozen": True,
        "exact_model_sha256": EXPECTED_MODEL_SHA256,
        "starting_classifier_sha256": EXPECTED_CLASSIFIER_SHA256,
        "control_classifier_sha256": sha256_file(control_classifier),
        "a2_classifier_sha256": sha256_file(a2_classifier),
        "head_lr": HEAD_LR,
        "head_weight_decay": HEAD_WEIGHT_DECAY,
        "head_epochs": HEAD_EPOCHS,
        "head_batch_size": HEAD_BATCH_SIZE,
        "encoder_batch_size": ENCODER_BATCH_SIZE,
        "replay_rows": len(replay),
        "extra_rows_unique": EXPECTED_A2_ROWS,
        "extra_repeat": A2_REPEAT,
        "extra_sample_weight": A2_SAMPLE_WEIGHT,
        "control_mean_train_loss": control_loss,
        "a2_mean_train_loss": a2_loss,
        "runtime_seconds": time.time() - started,
        "submission_archive": submission_identity,
    }
    write_json(OUTPUT_DIR / "training_meta.json", training_meta)
    update_status(
        "complete",
        macro_a2=a2_metrics["macro_average_precision"],
        macro_control=control_metrics["macro_average_precision"],
        a2_minus_control=comparison["a2_minus_control_macro"],
        submission=str(submission_zip),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT_DIR / "failure.json",
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_unix": time.time(),
            },
        )
        update_status("error", error_type=type(exc).__name__, error_message=str(exc))
        raise
