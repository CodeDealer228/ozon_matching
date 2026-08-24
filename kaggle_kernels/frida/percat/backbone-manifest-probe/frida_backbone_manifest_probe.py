import hashlib
import json
import os
import zipfile

ZIP_NAME = "matching-frida_continue_c_last_188-submit.zip"
EXPECTED = {
    "classifier.pt": (4_340_725, "0fd9134e8cd4a9b945aee7fb3d12d991208c9ff93d293e958c0ec3b35eb94612"),
    "models/frida-full-sft/config.json": (846, "f42bc8b9f9706487667056e8bb5c2151a995fbf7d056c2461936d147e24e3651"),
    "models/frida-full-sft/model.safetensors": (1_646_829_336, "ffae8888af57ee3d5337607033dbb972728bcec07937441fbc95414e43d9d758"),
}


def sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    matches = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if ZIP_NAME in files:
            matches.append(os.path.join(root, ZIP_NAME))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {ZIP_NAME}, found {matches}")

    path = matches[0]
    with open(path, "rb") as stream:
        archive_sha = sha256(stream)
    result = {"archive": path, "archive_sha256": archive_sha, "members": {}}
    with zipfile.ZipFile(path) as archive:
        for name, (expected_size, expected_sha) in EXPECTED.items():
            info = archive.getinfo(name)
            with archive.open(name) as stream:
                actual_sha = sha256(stream)
            result["members"][name] = {"size": info.file_size, "sha256": actual_sha}
            if (info.file_size, actual_sha) != (expected_size, expected_sha):
                raise RuntimeError(f"content mismatch for {name}: {result['members'][name]}")

    with open("/kaggle/working/backbone_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print("BACKBONE CONTENT MATCHES LOCAL ARCHIVE", flush=True)


if __name__ == "__main__":
    main()
