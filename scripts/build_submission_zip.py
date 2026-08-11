"""
Zip submits/lora/ into a submission archive, the same way
submits/baseline/matching-baseline-submit.zip is structured (flat files at
the archive root: metadata.json, run.py, classifier.pt, src/, models/) --
except lean: only the files run.py actually loads (safetensors + tokenizer +
config), none of the onnx/openvino/flax/pytorch_model.bin duplicates the
baseline zip carries.

Usage:
    python scripts/build_submission_zip.py
"""

import os
import zipfile

SRC_DIR = os.path.join("submits", "lora")
OUT_PATH = os.path.join("submits", "lora", "matching-lora-submit.zip")

SKIP_NAMES = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
SKIP_EXT = {".pyc"}


def iter_files(src_dir):
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        for f in files:
            if f in SKIP_NAMES or os.path.splitext(f)[1] in SKIP_EXT:
                continue
            if os.path.join(root, f) == os.path.abspath(OUT_PATH):
                continue
            yield os.path.join(root, f)


def main():
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    files = list(iter_files(SRC_DIR))
    total_size = sum(os.path.getsize(f) for f in files)

    print(f"[ZIP] {len(files)} files, {total_size / 1e6:.1f} MB uncompressed")

    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            arcname = os.path.relpath(f, SRC_DIR)
            zf.write(f, arcname)
            print(f"  + {arcname} ({os.path.getsize(f):,} bytes)")

    zip_size = os.path.getsize(OUT_PATH)
    print(f"[ZIP] wrote {OUT_PATH} -- {zip_size / 1e6:.1f} MB "
          f"({zip_size / (5 * 1024**3) * 100:.1f}% of the 5GB limit)")

    with zipfile.ZipFile(OUT_PATH) as zf:
        bad = zf.testzip()
        assert bad is None, f"corrupt entry in zip: {bad}"
        names = zf.namelist()

    required = {
        "metadata.json", "run.py", "classifier.pt",
        "src/utils.py",
        "models/lora-merged-ms-marco-MiniLM-L12-v2/config.json",
        "models/lora-merged-ms-marco-MiniLM-L12-v2/model.safetensors",
        "models/lora-merged-ms-marco-MiniLM-L12-v2/tokenizer.json",
        "models/lora-merged-ms-marco-MiniLM-L12-v2/tokenizer_config.json",
    }
    missing = required - set(n.replace("\\", "/") for n in names)
    assert not missing, f"zip is missing required files: {missing}"

    print(f"[ZIP] integrity OK, all {len(required)} required files present")


if __name__ == "__main__":
    main()
