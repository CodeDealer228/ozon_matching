import argparse
import os
from src.utils import predict_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", required=True)
parser.add_argument("--items_path", required=True)
parser.add_argument("--matches_path", required=True)
args = parser.parse_args()
predict_pipeline(args.items_path, args.matches_path,
                 os.path.join(ROOT, "models", "frida-format-adapter-v4"),
                 os.path.join(ROOT, "classifier.pt"),
                 os.path.join(ROOT, "experts"), args.output_path,
                 os.path.join(ROOT, "wheels"), batch_size=4096)
