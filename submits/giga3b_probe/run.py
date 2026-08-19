import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import predict_pipeline

# Pre-quantized 4-bit NF4 Giga-Embeddings-instruct weights shipped inside the
# archive (iMiW/Giga-Embeddings-instruct-4bit-nf4 on HuggingFace -- see
# src/model.py's module docstring for why this replaced this project's own
# int8 quantize-and-save pipeline: smaller on disk, near-identical modeling
# code, own baked-in quantization_config).
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "giga_4bit_nf4")
CLASSIFIER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier.joblib")

# Cross-encoder mode (one forward pass per PAIR, not per item -- see
# src/model.py's docstring for why the earlier bi-encoder version was wrong
# for this ~1.94-items/pair dataset): 2048 hidden, 36 layers, seq=128.
#
# Batch size is NOT a hand-derived guess anymore -- GigaCrossEncoder.
# probe_batch_size() measures it for real, on whatever GPU this run actually
# lands on (the judge stand's H100), by growing batch size to a real OOM and
# binary-searching the boundary (same algorithm as kaggle_kernels/
# frida-batch-probe). The old static BATCH_SIZE=1024 constant was flagged in
# its own comment as "still unverified on H100 itself" -- this replaces that
# guess with a measurement instead of a better guess.
PROBE_START_BATCH = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None,
                         help="skip the startup VRAM auto-probe and force this batch size")
    parser.add_argument("--weights_dir", type=str, default=WEIGHTS_DIR)
    parser.add_argument("--classifier_path", type=str, default=CLASSIFIER_PATH)
    parser.add_argument("--no_torch_compile", action="store_true",
                         help="skip the torch.compile canary attempt at load time")
    args = parser.parse_args()

    from src.model import GigaCrossEncoder, maybe_offline_install
    maybe_offline_install(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wheels"))

    # Loads eager. Probing happens against the eager model on purpose (see
    # model.py's benchmark_and_maybe_compile docstring for why compiling
    # before/during the probe is actively harmful) -- compiling, if it wins,
    # happens once afterward at the one batch size the real run will use.
    encoder = GigaCrossEncoder(args.weights_dir)

    batch_size = args.batch_size or encoder.probe_batch_size(start=PROBE_START_BATCH)

    if not args.no_torch_compile:
        encoder.benchmark_and_maybe_compile(batch_size)

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        encoder=encoder,
        classifier_path=args.classifier_path,
        output_csv_path=args.output_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
