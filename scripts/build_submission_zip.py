"""
Zip a submits/<name>/ directory into a submission archive, the same way
submits/baseline/matching-baseline-submit.zip is structured (flat files at
the archive root: metadata.json, run.py, classifier file, src/, models/) --
except lean: only the files run.py actually loads (safetensors + tokenizer +
config), none of the onnx/openvino/flax/pytorch_model.bin duplicates the
baseline zip carries.

Usage:
    python scripts/build_submission_zip.py                          # submits/lora (default)
    python scripts/build_submission_zip.py --submission-dir submits/berta_probe
"""

import argparse
import os
import zipfile

SKIP_NAMES = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
SKIP_EXT = {".pyc"}


def iter_files(src_dir, out_path_abs):
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        for f in files:
            if f in SKIP_NAMES or os.path.splitext(f)[1] in SKIP_EXT:
                continue
            full = os.path.join(root, f)
            if os.path.abspath(full) == out_path_abs:
                continue
            yield full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", default=os.path.join("submits", "lora"))
    parser.add_argument("--out-path", default=None, help="defaults to <submission-dir>/matching-<name>-submit.zip")
    args = parser.parse_args()

    src_dir = args.submission_dir
    dir_name = os.path.basename(os.path.normpath(src_dir))
    out_path = args.out_path or os.path.join(src_dir, f"matching-{dir_name}-submit.zip")
    out_path_abs = os.path.abspath(out_path)

    if os.path.exists(out_path):
        os.remove(out_path)

    files = list(iter_files(src_dir, out_path_abs))
    total_size = sum(os.path.getsize(f) for f in files)

    print(f"[ZIP] {src_dir}: {len(files)} files, {total_size / 1e6:.1f} MB uncompressed")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            arcname = os.path.relpath(f, src_dir)
            zf.write(f, arcname)
            print(f"  + {arcname} ({os.path.getsize(f):,} bytes)")

    zip_size = os.path.getsize(out_path)
    print(f"[ZIP] wrote {out_path} -- {zip_size / 1e6:.1f} MB "
          f"({zip_size / (5 * 1024**3) * 100:.1f}% of the 5GB limit)")

    with zipfile.ZipFile(out_path) as zf:
        bad = zf.testzip()
        assert bad is None, f"corrupt entry in zip: {bad}"
        names = {n.replace("\\", "/") for n in zf.namelist()}

    required = {"metadata.json", "run.py", "src/utils.py"}
    missing = required - names
    assert not missing, f"zip is missing required files: {missing}"

    has_classifier = any(n.endswith((".joblib", ".pt")) and "/" not in n for n in names)
    if not has_classifier:
        print("[ZIP] note: no top-level classifier file (*.joblib or *.pt) -- fine for a "
              "cosine-similarity bi-encoder submission (e.g. giga3b_probe), expected for a "
              "trained-classifier one (baseline/lora-style)")

    model_files = {n for n in names if n.startswith("models/")}
    assert any(n.endswith("config.json") for n in model_files), "zip is missing models/**/config.json"
    assert any(n.endswith((".safetensors", ".bin")) for n in model_files), "zip is missing model weights under models/"
    assert any(n.endswith("tokenizer.json") for n in model_files) or any(n.endswith("tokenizer_config.json") for n in model_files), \
        "zip is missing a tokenizer file under models/"

    print(f"[ZIP] integrity OK ({len(required)} required top-level files + classifier + model weights/config/tokenizer present)")


if __name__ == "__main__":
    main()
