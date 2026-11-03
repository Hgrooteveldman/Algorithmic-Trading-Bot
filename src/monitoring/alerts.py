"""Operator alerts for risk events.

Every alert is written to the log at CRITICAL with ``event`` and the fields
as structured context (so it lands in the JSON log too). If
``ALERT_WEBHOOK_URL`` is set, the same payload is POSTed there (Slack,
Discord, ntfy and most chat tools accept a JSON webhook). A failing webhook
never raises: alerting must not be able to crash the trading loop.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Callable

from loguru import logger

Poster = Callable[[str, bytes], None]


def _default_poster(url: str, body: bytes) -> None:
    import requests  # imported lazily: tests and backtests never need it

    requests.post(url, data=body, headers={"Content-Type": "application/json"}, timeout=5)


class AlertSink:
    def __init__(self, webhook_url: str | None = None, poster: Poster | None = None) -> None:
        self.webhook_url = webhook_url if webhook_url is not None else os.getenv("ALERT_WEBHOOK_URL") or None
        self._post = poster or _default_poster
        self.sent: list[dict[str, Any]] = []

    def send(self, event: str, message: str, **fields: Any) -> dict[str, Any]:
        payload = {"event": event, "message": message, "at": datetime.now(UTC).isoformat(), **fields}
        logger.bind(event=event, alert=True, **fields).critical(message)
        self.sent.append(payload)
        if self.webhook_url:
            try:
                body = json.dumps({"text": f"[{event}] {message}", **payload}, default=str).encode()
                self._post(self.webhook_url, body)
            except Exception as exc:  # pragma: no cover - exercised via a failing fake poster in tests
                logger.bind(event="alert_delivery_failed").error(f"Alert webhook failed: {exc}")
        return payload


alerts = AlertSink()
"""Process-wide sink used by the risk layer."""
