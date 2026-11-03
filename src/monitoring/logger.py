"""Centralized logging via loguru.

Three sinks:
    - stdout, human-readable, INFO and up;
    - ``logs/bot_YYYY-MM-DD.log``, human-readable, DEBUG and up;
    - ``logs/bot_YYYY-MM-DD.jsonl``, one JSON object per line with every
      ``bind()`` field under ``record.extra`` — the audit trail.

Structured context is attached with :func:`audit`, e.g.::

    audit("order_submitted", trade_id=order.id, symbol="AAPL", qty=10,
          reference_price=187.12, est_slippage_bps=1.4, latency_ms=82.0)

so a later question like "how far off were our slippage estimates last
month?" is a ``jq`` one-liner instead of a regex over prose.
"""
from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from config.settings import LOGS_DIR


def setup_logger(json_logs: bool = True) -> None:
    """Call once at startup."""
    logger.remove()  # clear default handler
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
    )
    logger.add(
        LOGS_DIR / "bot_{time:YYYY-MM-DD}.log",
        rotation="00:00",          # new file at midnight UTC
        retention="60 days",       # keep two months of history
        level="DEBUG",             # full detail in the file
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {name}:{line} | {message}",
        enqueue=True,              # async-safe
    )
    if json_logs:
        logger.add(
            LOGS_DIR / "bot_{time:YYYY-MM-DD}.jsonl",
            rotation="00:00",
            retention="60 days",
            level="INFO",
            serialize=True,        # loguru writes the full record, extras included
            enqueue=True,
        )


def audit(event: str, level: str = "INFO", **fields: Any) -> None:
    """Log one structured audit event. ``fields`` become JSON keys."""
    logger.bind(event=event, **fields).log(level, event + " " + " ".join(f"{k}={v}" for k, v in fields.items()))
