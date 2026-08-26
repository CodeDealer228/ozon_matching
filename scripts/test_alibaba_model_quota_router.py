import json
import tempfile
from pathlib import Path

from alibaba_model_quota_router import ModelQuotaRouter, classify_provider_error


class ProviderError(Exception):
    def __init__(self, status_code=None, code="", message=""):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"code": code, "message": message}}


with tempfile.TemporaryDirectory() as tmp_dir:
    root = Path(tmp_dir)
    registry = root / "models.json"
    events = root / "events.jsonl"
    registry.write_text(json.dumps({
        "provider": "Alibaba Cloud Model Studio",
        "models": [
            {"id": "new", "rank": 1},
            {"id": "fallback", "rank": 2},
        ],
    }), encoding="utf-8")
    router = ModelQuotaRouter([registry], events)
    assert router.current_model == "new"
    quota = ProviderError(429, "AllocationQuota.FreeTierOnly", "free quota exhausted")
    assert classify_provider_error(quota) == "quota"
    assert router.register_error("new", quota) == "quota"
    assert router.current_model == "fallback"
    assert ModelQuotaRouter([registry], events).current_model == "fallback"

    unavailable = ProviderError(404, "model_not_found", "missing")
    assert router.register_error("fallback", unavailable) == "model_unavailable"
    assert router.current_model is None

assert classify_provider_error(ProviderError(None, "", "connection reset")) == "uncertain"
assert classify_provider_error(ProviderError(400, "invalid_request", "bad")) == "terminal"
