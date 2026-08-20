import argparse
import os

from src.utils import predict_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "frida-base")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "frida-lora-adapter")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pt")
WHEELS_DIR = os.path.join(BASE_DIR, "wheels")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        base_model_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        classifier_path=CLASSIFIER_PATH,
        output_csv_path=args.output_path,
        wheels_dir=WHEELS_DIR,
        # batch_size intentionally omitted -- uses predict_pipeline's own
        # BATCH_SIZE_FIXED=6912 default, calibrated against a real OOM at
        # batch=8192/seq=192 -- see its comment in src/utils.py.
    )


if __name__ == "__main__":
    main()
