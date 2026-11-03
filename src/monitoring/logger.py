"""Centralized logging via loguru.

Why a logger module?
    - Every module just does `from loguru import logger` and gets the same sinks.
    - One place to change log format, rotation, retention.
    - Trade decisions and rejections are auditable after the fact.
"""
import sys
from loguru import logger

from config.settings import LOGS_DIR


def setup_logger() -> None:
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
