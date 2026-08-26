"""If frida_ap_soap_full_sft.py changes SOAP/StreamingAPLoss (or a constant
either one defaults to) and nobody reruns generate_extracted_sources.py, the
functional tests below would silently test stale code. This is the tripwire:
it recomputes the same extraction fresh and diffs it against what's on disk.
"""
from generate_extracted_sources import OUT_DIR, build_ap_loss_source, build_soap_source


def test_streaming_ap_loss_copy_matches_kernel():
    on_disk = (OUT_DIR / "streaming_ap_loss.py").read_text(encoding="utf-8")
    assert on_disk == build_ap_loss_source(), (
        "generated/streaming_ap_loss.py is stale -- rerun generate_extracted_sources.py"
    )


def test_soap_copy_matches_kernel():
    on_disk = (OUT_DIR / "soap_optimizer.py").read_text(encoding="utf-8")
    assert on_disk == build_soap_source(), (
        "generated/soap_optimizer.py is stale -- rerun generate_extracted_sources.py"
    )
