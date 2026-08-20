import argparse
import os

from src.utils import predict_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "berta-base")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "berta-lora-adapter")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pt")
TFIDF_DIR = os.path.join(BASE_DIR, "tfidf_vectorizers")
PCA_PATH = os.path.join(BASE_DIR, "pca_transformer.joblib")
CATBOOST_MODEL_PATH = os.path.join(BASE_DIR, "catboost_model.cbm")
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
        tfidf_dir=TFIDF_DIR,
        pca_path=PCA_PATH,
        catboost_model_path=CATBOOST_MODEL_PATH,
        output_csv_path=args.output_path,
        wheels_dir=WHEELS_DIR,
        batch_size=1024,
    )


if __name__ == "__main__":
    main()
