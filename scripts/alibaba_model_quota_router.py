"""Ordered Alibaba model rotation with durable quota events."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXACT_QUOTA_CODES = {
    "allocationquota.freetieronly",
    "allocationquota.exceeded",
    "quotaexceeded",
}


def _error_details(error: BaseException) -> tuple[int | None, str, str]:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    code = ""
    message = str(error)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict):
            code = str(nested.get("code", ""))
            message = str(nested.get("message", message))
    return status if isinstance(status, int) else None, code.casefold(), message


def classify_provider_error(error: BaseException) -> str:
    """Return quota, model_unavailable, uncertain, or terminal."""
    status, code, message = _error_details(error)
    message_folded = message.casefold()
    if code in EXACT_QUOTA_CODES or "allocationquota.freetieronly" in message_folded:
        return "quota"
    if status == 404 and code == "model_not_found":
        return "model_unavailable"
    if status is None:
        return "uncertain"
    return "terminal"


class ModelQuotaRouter:
    """Select models in registry order and persist exhausted-model decisions."""

    def __init__(self, registry_paths: Iterable[Path], events_path: Path):
        self.registry_paths = [Path(path) for path in registry_paths]
        if not self.registry_paths:
            raise ValueError("at_least_one_model_registry_required")
        self.events_path = Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.models = self._load_models()
        self.exhausted = self._load_exhausted()

    def _load_models(self) -> list[str]:
        ordered: list[tuple[int, str]] = []
        seen: set[str] = set()
        for path_index, path in enumerate(self.registry_paths):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("provider") != "Alibaba Cloud Model Studio":
                raise ValueError(f"unexpected_provider:{path}")
            models = value.get("models")
            if not isinstance(models, list) or not models:
                raise ValueError(f"empty_model_registry:{path}")
            for local_index, entry in enumerate(models):
                if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                    raise ValueError(f"invalid_model_entry:{path}")
                model = entry["id"]
                if model in seen:
                    raise ValueError(f"duplicate_model:{model}")
                seen.add(model)
                rank = int(entry.get("rank", local_index + 1))
                ordered.append((path_index * 10000 + rank, model))
        return [model for _, model in sorted(ordered)]

    def _load_exhausted(self) -> set[str]:
        exhausted: set[str] = set()
        if not self.events_path.exists():
            return exhausted
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") == "model_exhausted":
                exhausted.add(str(event["model"]))
        return exhausted

    @property
    def current_model(self) -> str | None:
        with self._lock:
            return next((model for model in self.models if model not in self.exhausted), None)

    def register_error(self, model: str, error: BaseException) -> str:
        """Persist an exact rotation event and return the error classification."""
        classification = classify_provider_error(error)
        if classification not in {"quota", "model_unavailable"}:
            return classification
        status, code, message = _error_details(error)
        with self._lock:
            if model not in self.exhausted:
                event: dict[str, Any] = {
                    "event": "model_exhausted",
                    "model": model,
                    "reason": classification,
                    "http_status": status,
                    "provider_code": code,
                    "message": message[:500],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                with self.events_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                self.exhausted.add(model)
        return classification

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "models": self.models,
                "exhausted_models": [model for model in self.models if model in self.exhausted],
                "current_model": next(
                    (model for model in self.models if model not in self.exhausted), None
                ),
            }
