"""Helpers for NYSE regular-session timing with DST awareness."""
from __future__ import annotations

from datetime import UTC, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = dtime(9, 30)
REGULAR_CLOSE = dtime(16, 0)


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_market_time(timestamp: datetime | None = None) -> datetime:
    current = timestamp or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(MARKET_TZ)


def attach_market_timezone(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=MARKET_TZ)
    return timestamp.astimezone(MARKET_TZ)


def market_session_date(timestamp: datetime | None = None):
    return to_market_time(timestamp).date()


def minutes_until_close(timestamp: datetime | None = None) -> float:
    market_now = to_market_time(timestamp)
    close_dt = datetime.combine(market_now.date(), REGULAR_CLOSE, MARKET_TZ)
    return (close_dt - market_now).total_seconds() / 60.0


def is_regular_session_weekday(timestamp: datetime | None = None) -> bool:
    return to_market_time(timestamp).weekday() < 5


def is_opening_no_trade_window(
    timestamp: datetime | None = None,
    minutes: int = 5,
) -> bool:
    if not is_regular_session_weekday(timestamp):
        return False
    market_now = to_market_time(timestamp)
    opening_start = datetime.combine(market_now.date(), REGULAR_OPEN, MARKET_TZ)
    opening_end = opening_start + timedelta(minutes=minutes)
    return opening_start <= market_now < opening_end


def is_flatten_window(
    timestamp: datetime | None = None,
    flatten_before_close_minutes: int = 15,
    session_close: datetime | None = None,
) -> bool:
    if not is_regular_session_weekday(timestamp):
        return False
    if session_close is not None:
        market_now = to_market_time(timestamp)
        remaining = (attach_market_timezone(session_close) - market_now).total_seconds() / 60.0
    else:
        remaining = minutes_until_close(timestamp)
    return 0 <= remaining <= flatten_before_close_minutes
