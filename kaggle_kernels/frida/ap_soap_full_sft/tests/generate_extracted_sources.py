"""Regenerates generated/streaming_ap_loss.py and generated/soap_optimizer.py
as exact copies of StreamingAPLoss and SOAP out of the real kernel file.

Run this by hand whenever frida_ap_soap_full_sft.py's SOAP class,
StreamingAPLoss class, or the constants either one defaults to changes:

    .venv/Scripts/python kaggle_kernels/frida/ap_soap_full_sft/tests/generate_extracted_sources.py

test_extraction_matches_kernel.py fails loudly if someone forgets -- it
recomputes the same extraction and diffs it against what's on disk.
"""
from pathlib import Path

from extract_from_kernel import extract

OUT_DIR = Path(__file__).resolve().parent / "generated"

_AP_LOSS_CONSTANTS = ("AP_GAMMA", "AP_SIGMOID_SCALE", "AP_EPS")
_SOAP_CONSTANTS = (
    "SOAP_BETAS", "SOAP_EPS", "WEIGHT_DECAY", "SOAP_SHAMPOO_BETA",
    "SOAP_PRECONDITION_FREQUENCY", "SOAP_MAX_PRECOND_DIM", "SOAP_CLIP_VALUE",
)

_AP_LOSS_HEADER = '''"""Exact copy of StreamingAPLoss, extracted from frida_ap_soap_full_sft.py.
Regenerate with generate_extracted_sources.py -- do not hand-edit.
"""
import torch
import torch.nn as nn


'''

_SOAP_HEADER = '''"""Exact copy of SOAP, extracted from frida_ap_soap_full_sft.py.
Regenerate with generate_extracted_sources.py -- do not hand-edit.
"""
import torch
from torch.optim import Optimizer


'''


def build_ap_loss_source():
    sources = extract(*_AP_LOSS_CONSTANTS, "StreamingAPLoss")
    constants = "\n".join(sources[name] for name in _AP_LOSS_CONSTANTS)
    return _AP_LOSS_HEADER + constants + "\n\n\n" + sources["StreamingAPLoss"] + "\n"


def build_soap_source():
    sources = extract(*_SOAP_CONSTANTS, "SOAP")
    constants = "\n".join(sources[name] for name in _SOAP_CONSTANTS)
    return _SOAP_HEADER + constants + "\n\n\n" + sources["SOAP"] + "\n"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "__init__.py").touch(exist_ok=True)
    (OUT_DIR / "streaming_ap_loss.py").write_text(build_ap_loss_source(), encoding="utf-8")
    (OUT_DIR / "soap_optimizer.py").write_text(build_soap_source(), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'streaming_ap_loss.py'}")
    print(f"wrote {OUT_DIR / 'soap_optimizer.py'}")


if __name__ == "__main__":
    main()
