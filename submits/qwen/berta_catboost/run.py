import argparse
import os

from src.utils import predict_pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QWEN_MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen3-embedding-0.6b-merged")
BERTA_BASE_PATH = os.path.join(BASE_DIR, "models", "BERTA")
BERTA_ADAPTER_PATH = os.path.join(BASE_DIR, "models", "berta-lora-adapter")
BERTA_CLASSIFIER_PATH = os.path.join(BASE_DIR, "berta_classifier.pt")
CATBOOST_MODEL_PATH = os.path.join(BASE_DIR, "catboost_model.cbm")
WHEELS_DIR = os.path.join(BASE_DIR, "wheels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", "--output-path", dest="output_path", type=str, required=True)
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        output_csv_path=args.output_path,
        qwen_model_path=QWEN_MODEL_PATH,
        berta_base_path=BERTA_BASE_PATH,
        berta_adapter_path=BERTA_ADAPTER_PATH,
        berta_classifier_path=BERTA_CLASSIFIER_PATH,
        catboost_model_path=CATBOOST_MODEL_PATH,
        wheels_dir=WHEELS_DIR,
    )


if __name__ == "__main__":
    main()
