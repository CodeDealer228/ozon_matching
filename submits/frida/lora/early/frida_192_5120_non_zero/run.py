import argparse
import os
import time
import sys
import threading

from src.utils import predict_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "frida-base")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "frida-lora-adapter")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pt")
WHEELS_DIR = os.path.join(BASE_DIR, "wheels")


# ===== DEBUG: искусственный таймаут через 5 минут =====
# os._exit(1) убивает процесс немедленно, даже если основной поток
# заблокирован в CUDA-ядре или держит GIL.
# sys.exit() НЕ подойдет — он бросает SystemExit, который может быть
# перехвачен try/except в transformers/peft/torch.
TIMEOUT_SECONDS = 300  # 5 минут

def _timeout_watchdog():
    time.sleep(TIMEOUT_SECONDS)
    # Пишем в stderr, чтобы judging-лог показал причину
    sys.stderr.write(f"\n[DEBUG TIMEOUT] Process killed after {TIMEOUT_SECONDS}s "
                     f"(simulating non-zero exit)\n")
    sys.stderr.flush()
    os._exit(1)  # exit code 1 = non-zero

# Стартуем watchdog ДО всего. Daemon=False чтобы процесс не завершился
# раньше времени, если main thread вдруг завершится быстро.
_watchdog = threading.Thread(target=_timeout_watchdog, daemon=False, name="timeout-watchdog")
_watchdog.start()
# ===== END DEBUG =====

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
        # BATCH_SIZE_FIXED=5120 default (fixed-shape batching +
        # PYTORCH_ALLOC_CONF=expandable_segments:True both already applied) --
        # see comments in src/utils.py for the three real OOM data points
        # this is calibrated against.
    )


if __name__ == "__main__":
    main()
