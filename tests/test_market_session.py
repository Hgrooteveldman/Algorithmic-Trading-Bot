from datetime import UTC, datetime

from src.market.session import is_flatten_window, is_opening_no_trade_window


def test_opening_window_is_dst_aware_in_winter():
    winter_open = datetime(2026, 1, 15, 14, 32, tzinfo=UTC)
    assert is_opening_no_trade_window(winter_open, minutes=5)


def test_opening_window_is_dst_aware_in_summer():
    summer_open = datetime(2026, 7, 15, 13, 32, tzinfo=UTC)
    assert is_opening_no_trade_window(summer_open, minutes=5)


def test_flatten_window_triggers_before_close():
    winter_close = datetime(2026, 1, 15, 20, 50, tzinfo=UTC)
    assert is_flatten_window(
        winter_close,
        flatten_before_close_minutes=15,
    )
