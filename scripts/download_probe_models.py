"""
One-off local download of the 3 cross-encoder-size-probe candidate models
(see CLAUDE.md "In progress" / the cross-encoder baseline optimization task)
into their submission dirs' models/ subfolder. Skips repo files the
submission pipeline never touches (images, sentence-transformers pooling
config, raw pre-fast-tokenizer vocab dumps when a tokenizer.json is present).

Usage:
    python scripts/download_probe_models.py
"""

import os

from huggingface_hub import snapshot_download

TARGETS = [
    ("omarelshehy/arabic-english-sts-matryoshka", "submits/matryoshka_probe/models/arabic-english-sts-matryoshka"),
    ("ai-forever/FRIDA", "submits/frida_probe/models/FRIDA"),
    ("sergeyzh/BERTA", "submits/berta_probe/models/BERTA"),
]

IGNORE_PATTERNS = [
    "*.png", "*.jpg", "*.jpeg", "*.gitattributes",
    "1_Pooling/*", "modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
    "onnx/*", "openvino/*", "*.onnx", "*.msgpack", "*.bin", "*.h5",
]


def main():
    for repo_id, local_dir in TARGETS:
        print(f"[DL] {repo_id} -> {local_dir}", flush=True)
        os.makedirs(local_dir, exist_ok=True)
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=IGNORE_PATTERNS,
        )
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(path) for f in files
        )
        print(f"[DL] done: {path} ({size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
