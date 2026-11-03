"""Typed broker models, retry policy and the reconnecting bar stream."""
from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.broker.models import Bar, Fill, OrderAck, Quote, validate_bars
from src.broker.retry import RetryPolicy, with_retry
from src.data.bar_stream import BarStream

NOW = datetime(2026, 11, 3, 15, 30, tzinfo=UTC)


# --- models ---------------------------------------------------------------------

def test_quote_mid_spread_and_dict_shape() -> None:
    q = Quote(symbol="SPY", bid=99.99, ask=100.01)
    assert q.mid == pytest.approx(100.0)
    assert q.spread_bps == pytest.approx(2.0)
    assert q.as_dict() == {"bid": 99.99, "ask": 100.01, "mid": pytest.approx(100.0)}


@pytest.mark.parametrize("bid,ask", [(101.0, 100.0), (0.0, 0.0), (-1.0, 100.0)])
def test_invalid_quotes_rejected(bid, ask) -> None:
    with pytest.raises(ValidationError):
        Quote(symbol="SPY", bid=bid, ask=ask)


def test_quote_reads_sdk_object() -> None:
    raw = SimpleNamespace(bid_price=10.0, ask_price=10.02, timestamp=NOW)
    assert Quote.from_alpaca("X", raw).mid == pytest.approx(10.01)


def test_models_are_frozen() -> None:
    q = Quote(symbol="SPY", bid=1.0, ask=1.1)
    with pytest.raises(ValidationError):
        q.bid = 2.0  # type: ignore[misc]


def test_bar_from_attributes_and_envelope() -> None:
    raw = SimpleNamespace(symbol="SPY", timestamp=NOW, open=10, high=11, low=9.5, close=10.5, volume=1000)
    assert Bar.model_validate(raw).close == 10.5
    with pytest.raises(ValidationError):
        Bar(symbol="SPY", timestamp=NOW, open=10, high=10.2, low=9.5, close=10.5, volume=1)


def test_order_ack_normalises_sdk_enums() -> None:
    from alpaca.trading.enums import OrderSide, OrderStatus

    raw = SimpleNamespace(id="abc", client_order_id="algobot-spy-1", symbol="SPY", side=OrderSide.BUY,
                          qty="10", status=OrderStatus.ACCEPTED, submitted_at=NOW)
    ack = OrderAck.model_validate(raw)
    assert ack.side.value == "buy" and ack.qty == 10 and ack.status == "accepted"


def test_fill_from_order() -> None:
    from alpaca.trading.enums import OrderSide

    order = SimpleNamespace(id=1, symbol="SPY", side=OrderSide.SELL, filled_qty="5", filled_avg_price="101.2", filled_at=NOW)
    fill = Fill.from_order(order)
    assert fill.order_id == "1" and fill.qty == 5 and fill.side.value == "sell"


def test_validate_bars_drops_bad_rows_and_duplicates() -> None:
    idx = pd.to_datetime(["2026-11-03 15:00", "2026-11-03 14:00", "2026-11-03 16:00", "2026-11-03 16:00", "2026-11-03 17:00"], utc=True)
    bars = pd.DataFrame({
        "open":   [10, 10, 10, 10, 10],
        "high":   [11, 11, 9, 11, 11],       # third row: high below open -> bad
        "low":    [9, 9, 8, 9, 9],
        "close":  [10.5, 10.2, 8.5, 10.6, np.nan],  # last row: NaN -> bad
        "volume": [100, 100, 100, 100, 100],
    }, index=idx)
    clean = validate_bars(bars, "SPY")
    assert list(clean.index) == list(pd.to_datetime(["2026-11-03 14:00", "2026-11-03 15:00", "2026-11-03 16:00"], utc=True))
    assert clean.loc[pd.Timestamp("2026-11-03 16:00", tz="UTC"), "close"] == 10.6  # duplicate: last one kept


# --- retry ------------------------------------------------------------------------

class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(status_code)
        self.status_code = status_code


def test_retry_recovers_from_transient_errors(mocker) -> None:
    sleeps: list[float] = []
    fn = mocker.Mock(side_effect=[ConnectionError("reset"), _HttpError(503), "ok"])
    fn.__name__ = "get_account"
    wrapped = with_retry(RetryPolicy(attempts=4, base_delay=0.1), sleep=sleeps.append)(fn)
    assert wrapped() == "ok"
    assert fn.call_count == 3 and len(sleeps) == 2


def test_retry_does_not_retry_client_errors(mocker) -> None:
    fn = mocker.Mock(side_effect=_HttpError(403))
    fn.__name__ = "get_account"
    with pytest.raises(_HttpError):
        with_retry(RetryPolicy(attempts=4), sleep=lambda s: None)(fn)()
    assert fn.call_count == 1


def test_retry_gives_up_after_attempts(mocker) -> None:
    fn = mocker.Mock(side_effect=TimeoutError("slow"))
    fn.__name__ = "get_clock"
    with pytest.raises(TimeoutError):
        with_retry(RetryPolicy(attempts=3), sleep=lambda s: None)(fn)()
    assert fn.call_count == 3


def test_backoff_is_bounded_and_grows() -> None:
    import random

    policy = RetryPolicy(base_delay=0.5, max_delay=4.0)
    rng = random.Random(0)
    assert all(0 <= policy.delay(a, rng) <= min(4.0, 0.5 * 2**a) for a in range(8))


def test_submit_orders_are_not_wrapped_in_retry() -> None:
    from src.broker.alpaca_client import AlpacaClient

    assert not hasattr(AlpacaClient.submit_entry_order, "__wrapped__")
    assert hasattr(AlpacaClient.get_account, "__wrapped__")


def test_client_order_ids_are_unique_and_traceable() -> None:
    from src.broker.alpaca_client import _client_order_id

    a, b = _client_order_id("SPY"), _client_order_id("SPY")
    assert a != b and a.startswith("algobot-spy-")


# --- bar stream ------------------------------------------------------------------

class _FakeStream:
    """Delivers given bars, then 'disconnects' by returning from run()."""

    def __init__(self, bars, fail: bool = False) -> None:
        self.bars, self.fail, self.handler = bars, fail, None

    def subscribe_bars(self, handler, *symbols) -> None:
        self.handler = handler

    def run(self) -> None:
        if self.fail:
            raise ConnectionError("socket closed")
        import asyncio

        for bar in self.bars:
            asyncio.run(self.handler(bar))

    def stop(self) -> None:
        pass


def _raw_bar(minutes: int, symbol: str = "SPY"):
    return SimpleNamespace(symbol=symbol, timestamp=NOW + timedelta(minutes=minutes), open=10, high=11, low=9, close=10, volume=5)


def test_stream_wakes_loop_on_new_bar_and_ignores_stale() -> None:
    stream = BarStream(["SPY"], factory=lambda: _FakeStream([]))
    assert stream.handle_bar(_raw_bar(1)) is not None
    assert stream.wait_for_bar(0.01) is True
    stream.handle_bar(_raw_bar(0))            # older bar: no wake-up
    assert stream.wait_for_bar(0.01) is False
    assert stream.latest_bar_time("spy") == NOW + timedelta(minutes=1)


def test_stream_drops_invalid_bars() -> None:
    stream = BarStream(["SPY"], factory=lambda: _FakeStream([]))
    bad = SimpleNamespace(symbol="SPY", timestamp=NOW, open=10, high=9, low=8, close=10, volume=1)
    assert stream.handle_bar(bad) is None and not stream.last_bars


def test_stream_reconnects_with_backoff_then_stops() -> None:
    sessions = iter([_FakeStream([], fail=True), _FakeStream([], fail=True), _FakeStream([_raw_bar(5)])])
    waits: list[float] = []
    stream = BarStream(["SPY"], base_backoff=1.0, max_backoff=8.0)

    def factory():
        try:
            return next(sessions)
        except StopIteration:
            stream.stop()
            return _FakeStream([])

    stream.factory = factory
    stream.sleep = waits.append
    runner = threading.Thread(target=stream.run_forever)
    runner.start()
    runner.join(timeout=5)
    assert not runner.is_alive()
    assert stream.reconnects >= 3
    assert all(0 <= w <= 8.0 for w in waits)
    assert stream.latest_bar_time("SPY") == NOW + timedelta(minutes=5)
