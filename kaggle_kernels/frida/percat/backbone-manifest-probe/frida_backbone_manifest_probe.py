import hashlib
import json
import os

EXPECTED = {
    "classifier.pt": (4_340_725, "0fd9134e8cd4a9b945aee7fb3d12d991208c9ff93d293e958c0ec3b35eb94612"),
    "models/frida-format-adapter-v4/config.json": (846, "f42bc8b9f9706487667056e8bb5c2151a995fbf7d056c2461936d147e24e3651"),
    "models/frida-format-adapter-v4/model.safetensors": (1_646_829_336, "3978a7b9c0641ab75248257adc0abe2b2c0007a98202b2ce22040bf40b42b3d1"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    roots = []
    caches = {}
    files_seen = set()
    for root, _dirs, files in os.walk("/kaggle/input"):
        files_seen.update(files)
        if ("classifier.pt" in files and
                all(os.path.isfile(os.path.join(root, *name.split("/")))
                    for name in EXPECTED)):
            roots.append(root)
        if {"cache_meta.json", "ids.bin", "input_ids.bin", "attention_mask.bin"} <= set(files):
            with open(os.path.join(root, "cache_meta.json"), encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta.get("format_version") == 4:
                caches[meta.get("scope")] = {"root": root, "meta": meta}
    if len(roots) != 1:
        raise RuntimeError(f"expected one merged backbone root, got {roots}")
    if set(caches) != {"human", "llm"}:
        raise RuntimeError(f"expected human+llm v4 caches, got {caches}")
    for scope, entry in caches.items():
        meta = entry["meta"]
        if (meta.get("max_length") != 256 or meta.get("prefix") != ""
                or meta.get("category_in_text") is not False
                or meta.get("add_special_tokens") is not True):
            raise RuntimeError(f"wrong {scope} cache contract: {meta}")
    for filename in ("matches_item_disjoint_70_30_seed42.parquet",
                     "llm_train_val_split.parquet"):
        if filename not in files_seen:
            raise RuntimeError(f"missing {filename}")

    root = roots[0]
    content = {}
    for name, (expected_size, expected_hash) in EXPECTED.items():
        path = os.path.join(root, *name.split("/"))
        actual = {"size": os.path.getsize(path), "sha256": sha256(path)}
        content[name] = actual
        if (actual["size"], actual["sha256"]) != (expected_size, expected_hash):
            raise RuntimeError(f"content mismatch for {name}: {actual}")

    report = {"backbone_root": root, "content": content, "caches": caches}
    with open("/kaggle/working/preflight_contract.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("PREFLIGHT CONTRACT OK", flush=True)


if __name__ == "__main__":
    main()
