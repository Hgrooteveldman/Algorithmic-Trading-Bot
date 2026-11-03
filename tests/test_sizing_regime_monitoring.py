"""Dynamic sizing, regime detection, performance kill-switch, alerts and
structured audit logging (with pytest-mock fakes for the broker)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from alpaca.trading.enums import OrderSide, OrderStatus
from loguru import logger

from src.execution.order_executor import OrderExecutor
from src.execution.trade_reconciliation import reconcile_closed_trades
from src.monitoring.alerts import AlertSink
from src.monitoring.logger import audit
from src.regime.detector import (
    GaussianHMM2,
    Regime,
    RegimePolicy,
    adx,
    regime_frame,
    regime_multipliers,
)
from src.risk.performance_monitor import PerformanceMonitor
from src.risk.risk_manager import RiskManager
from src.risk.sizing import DynamicSizer, kelly_from_trade_returns
from src.runtime.state_store import PositionSnapshot
from tests.conftest import make_synthetic_bars


# --- sizing ---------------------------------------------------------------------

def test_vol_target_halves_size_when_atr_doubles() -> None:
    sizer = DynamicSizer(risk_per_trade_pct=0.002, atr_stop_mult=2.0, max_position_pct=0.25,
                         max_sector_pct=1.0, round_trip_friction_pct=0.0)
    calm = sizer.size(symbol="X", equity=100_000, price=100, stop_price=None, atr=1.0)
    wild = sizer.size(symbol="X", equity=100_000, price=100, stop_price=None, atr=2.0)
    assert calm.binding == "vol_target" and calm.qty == 100   # 200 risk / (2 * 1.0)
    assert wild.qty == 50


def test_stop_wider_than_atr_takes_precedence() -> None:
    sizer = DynamicSizer(risk_per_trade_pct=0.01, atr_stop_mult=1.0, max_position_pct=0.25,
                         max_sector_pct=1.0, round_trip_friction_pct=0.0)
    decision = sizer.size(symbol="X", equity=100_000, price=100, stop_price=95, atr=1.0)
    assert decision.qty == 200  # 1000 / 5


def test_regime_multiplier_scales_and_blocks() -> None:
    sizer = DynamicSizer(risk_per_trade_pct=0.002, max_position_pct=0.25, max_sector_pct=1.0, round_trip_friction_pct=0.0)
    full = sizer.size(symbol="X", equity=100_000, price=100, stop_price=98, atr=None)
    half = sizer.size(symbol="X", equity=100_000, price=100, stop_price=98, atr=None, regime_multiplier=0.5)
    blocked = sizer.size(symbol="X", equity=100_000, price=100, stop_price=98, atr=None, regime_multiplier=0.0)
    assert full.binding == "vol_target" and full.qty == 100
    assert half.qty == 50
    assert blocked.rejected and blocked.binding == "regime"


def test_sector_cap_counts_existing_exposure() -> None:
    sizer = DynamicSizer(risk_per_trade_pct=0.02, max_position_pct=0.25, max_sector_pct=0.20,
                         round_trip_friction_pct=0.0, sector_by_symbol={"AAPL": "tech", "MSFT": "tech"})
    decision = sizer.size(symbol="AAPL", equity=100_000, price=100, stop_price=99, atr=None,
                          notional_by_symbol={"MSFT": 15_000, "XOM": 50_000})
    assert decision.binding == "sector_cap"
    assert decision.qty == 50  # 20k cap - 15k tech already held = 5k / 100


def test_kelly_formula_and_cap() -> None:
    est = kelly_from_trade_returns([0.04, -0.02] * 20)   # p=.5, win 4%, loss 2%
    assert est is not None
    assert est.full_kelly == pytest.approx(0.5 / 0.02 - 0.5 / 0.04)  # 12.5x
    sizer = DynamicSizer(risk_per_trade_pct=0.02, max_position_pct=0.25, max_sector_pct=1.0,
                         kelly_fraction=0.25, min_trades_for_kelly=30, round_trip_friction_pct=0.0)
    losing = [0.01, -0.02] * 20  # negative edge
    decision = sizer.size(symbol="X", equity=100_000, price=100, stop_price=99, atr=None, trade_returns=losing)
    assert decision.binding == "kelly_cap" and decision.qty == 0


def test_kelly_ignored_below_min_trades() -> None:
    sizer = DynamicSizer(min_trades_for_kelly=30)
    decision = sizer.size(symbol="X", equity=100_000, price=100, stop_price=98, atr=None, trade_returns=[-0.01] * 10)
    assert "kelly_cap" not in decision.components


@pytest.mark.parametrize("kwargs", [{"kelly_fraction": 1.0}, {"kelly_fraction": 0.1}, {"risk_per_trade_pct": 0.05}])
def test_sizer_rejects_aggressive_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        DynamicSizer(**kwargs)


# --- regime -----------------------------------------------------------------------

def _trending_bars(n: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100 * np.exp(np.cumsum(np.full(n, 0.002) + np.random.default_rng(0).normal(0, 0.001, n)))
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close}, index=idx)


def test_adx_high_in_a_clean_trend() -> None:
    assert adx(_trending_bars()).iloc[-1] > 25


def test_regime_is_causal() -> None:
    bars = make_synthetic_bars(7, n=1500)
    full = regime_frame(bars)["regime"]
    cut = regime_frame(bars.iloc[:900])["regime"]
    pd.testing.assert_series_equal(full.iloc[:900], cut)


def test_regime_labels_and_warmup() -> None:
    frame = regime_frame(make_synthetic_bars(8, n=1200))
    assert set(frame["regime"].unique()) <= {r.value for r in Regime}
    assert (frame["regime"].iloc[:50] == Regime.NEUTRAL.value).all()


def test_policy_pauses_mean_reversion_in_volatile_trend_and_momentum_in_chop() -> None:
    policy = RegimePolicy()
    assert policy.multiplier("mean_reversion", Regime.HIGH_VOL_TREND) == 0.0
    assert policy.multiplier("mean_reversion", Regime.TREND) == 0.5
    assert policy.multiplier("momentum", Regime.LOW_VOL_CHOP) == 0.0
    assert policy.multiplier("momentum", "neutral") == 1.0


def test_regime_multipliers_series_aligned() -> None:
    bars = make_synthetic_bars(9, n=800)
    mult = regime_multipliers(bars, "mean_reversion")
    assert mult.index.equals(bars.index) and mult.between(0, 1).all()


def test_hmm_separates_calm_and_volatile_and_stays_causal() -> None:
    rng = np.random.default_rng(10)
    returns = np.concatenate([rng.normal(0, 0.002, 400), rng.normal(0, 0.02, 200), rng.normal(0, 0.002, 400)])
    model = GaussianHMM2().fit(returns[:600])
    assert model.variances[1] > model.variances[0]
    prob = model.filtered_high_vol_probability(returns)
    assert prob[450:550].mean() > 0.8 and prob[100:300].mean() < 0.2
    # filtered probabilities at t do not change when later data is appended
    np.testing.assert_allclose(prob[:500], model.filtered_high_vol_probability(returns[:500]))


# --- performance kill-switch and alerts --------------------------------------------

def test_kill_switch_trips_after_confirmed_degradation(tmp_path) -> None:
    sink = AlertSink(webhook_url="")
    stop_file = tmp_path / "STOP"
    monitor = PerformanceMonitor(window=30, min_trades=20, confirmations=3, kill_switch_path=stop_file, alert_sink=sink)
    trips = [monitor.record(r) for r in [-0.01, 0.002] * 15]
    assert any(trips) and monitor.tripped
    assert stop_file.exists()
    assert sink.sent[-1]["event"] == "performance_kill_switch"


def test_kill_switch_quiet_for_healthy_strategy(tmp_path) -> None:
    monitor = PerformanceMonitor(window=30, min_trades=20, kill_switch_path=tmp_path / "STOP",
                                 alert_sink=AlertSink(webhook_url=""))
    for r in [0.01, -0.005] * 30:
        monitor.record(r)
    assert not monitor.tripped and not (tmp_path / "STOP").exists()


def test_monitor_state_round_trip(tmp_path) -> None:
    a = PerformanceMonitor(window=30, min_trades=20, alert_sink=AlertSink(webhook_url=""))
    for r in [0.01, -0.02, 0.005]:
        a.record(r)
    b = PerformanceMonitor(window=30, min_trades=20, alert_sink=AlertSink(webhook_url=""))
    b.restore(a.snapshot())
    assert b.trade_returns == a.trade_returns


def test_webhook_alert_posts_json(mocker) -> None:
    poster = mocker.Mock()
    sink = AlertSink(webhook_url="https://hooks.example/x", poster=poster)
    sink.send("daily_loss_limit", "halted", daily_pnl_pct=-0.021)
    url, body = poster.call_args.args
    assert url == "https://hooks.example/x"
    assert json.loads(body)["daily_pnl_pct"] == -0.021


def test_failing_webhook_never_raises(mocker) -> None:
    sink = AlertSink(webhook_url="https://hooks.example/x", poster=mocker.Mock(side_effect=OSError("down")))
    sink.send("circuit_breaker", "halted")  # must not raise


def test_risk_manager_alerts_once_on_daily_loss(tmp_path) -> None:
    from config.settings import settings

    sink = AlertSink(webhook_url="")
    rm = RiskManager(starting_equity=100_000, alert_sink=sink)
    rm.performance.kill_switch_path = tmp_path / "STOP"
    bad = 100_000 * (1 - settings.max_daily_loss_pct - 0.001)
    rm.can_trade(bad, 0)
    rm.can_trade(bad, 0)
    assert [a["event"] for a in sink.sent] == ["daily_loss_limit"]


def test_risk_manager_feeds_monitor_and_persists(tmp_path) -> None:
    rm = RiskManager(starting_equity=100_000, alert_sink=AlertSink(webhook_url=""))
    rm.performance.kill_switch_path = tmp_path / "STOP"
    rm.record_trade_outcome(-50.0, -0.01)
    rm.record_trade_outcome(80.0, 0.016)
    restored = RiskManager(starting_equity=100_000, alert_sink=AlertSink(webhook_url=""))
    restored.restore(rm.snapshot())
    assert list(restored.trade_returns) == [-0.01, 0.016]


def test_size_entry_never_exceeds_legacy_size(tmp_path) -> None:
    rm = RiskManager(starting_equity=100_000, alert_sink=AlertSink(webhook_url=""))
    decision = rm.size_entry(symbol="SPY", equity=100_000, entry_price=500.0, stop_loss_price=490.0, atr=2.0)
    assert decision.qty <= rm.calculate_position_size(100_000, 500.0, 490.0)
    assert "legacy_fixed_risk" in decision.components


# --- structured logging -------------------------------------------------------------

def test_audit_writes_json_with_context(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    sink_id = logger.add(path, serialize=True, level="INFO")
    try:
        audit("order_submitted", trade_id="abc", symbol="SPY", est_slippage_bps=1.2, latency_ms=40.0)
    finally:
        logger.remove(sink_id)
    record = json.loads(path.read_text().splitlines()[-1])["record"]
    assert record["extra"]["event"] == "order_submitted"
    assert record["extra"]["trade_id"] == "abc"
    assert record["extra"]["latency_ms"] == 40.0


# --- broker-facing flow with pytest-mock ------------------------------------------

def _fake_broker(mocker, equity: float = 100_000.0):
    broker = mocker.Mock()
    broker.get_equity.return_value = equity
    broker.get_buying_power.return_value = equity
    broker.get_positions.return_value = []
    broker.get_position.return_value = None
    broker.get_latest_quote.return_value = {"mid": 100.0, "bid": 99.99, "ask": 100.01}
    broker.submit_entry_order.return_value = SimpleNamespace(id="order-1")
    return broker


def test_executor_sizes_with_atr_and_records_estimate(mocker, tmp_path) -> None:
    mocker.patch("src.risk.risk_manager.RiskManager.in_no_trade_window", return_value=False)
    rm = RiskManager(starting_equity=100_000, alert_sink=AlertSink(webhook_url=""))
    rm.performance.kill_switch_path = tmp_path / "STOP"
    mocker.patch("config.settings.settings.kill_switch_path", tmp_path / "NO_STOP")
    broker = _fake_broker(mocker)
    executor = OrderExecutor(broker=broker, risk=rm)
    bars = make_synthetic_bars(3, n=300)
    submission = executor.open_long_with_levels("SPY", stop_loss_price=98.0, take_profit_price=104.0, bars=bars)
    assert submission is not None and submission.qty > 0
    assert "order-1" in executor.entry_estimates
    assert executor.entry_estimates["order-1"]["est_slippage_bps"] == pytest.approx(1.0, abs=0.01)


def test_executor_skips_when_regime_blocks(mocker, tmp_path) -> None:
    mocker.patch("src.risk.risk_manager.RiskManager.in_no_trade_window", return_value=False)
    mocker.patch("config.settings.settings.kill_switch_path", tmp_path / "NO_STOP")
    rm = RiskManager(starting_equity=100_000, alert_sink=AlertSink(webhook_url=""))
    broker = _fake_broker(mocker)
    executor = OrderExecutor(broker=broker, risk=rm)
    assert executor.open_long_with_levels("SPY", 98.0, 104.0, regime_multiplier=0.0) is None
    broker.submit_entry_order.assert_not_called()


def test_reconciliation_returns_and_slippage_audit(mocker) -> None:
    now = datetime.now(UTC)
    buy = SimpleNamespace(id="b1", symbol="SPY", side=OrderSide.BUY, status=OrderStatus.FILLED,
                          filled_at=now, filled_qty=10, filled_avg_price=100.05)
    sell = SimpleNamespace(id="s1", symbol="SPY", side=OrderSide.SELL, status=OrderStatus.FILLED,
                           filled_at=now + timedelta(minutes=5), filled_qty=10, filled_avg_price=102.0)
    broker = mocker.Mock()
    broker.get_closed_orders.return_value = [buy, sell]
    audit_spy = mocker.patch("src.execution.trade_reconciliation.audit")
    estimates = {"b1": {"reference_price": 100.0, "est_slippage_bps": 1.0, "submitted_at": now - timedelta(seconds=2)}}
    result = reconcile_closed_trades(broker, {"SPY": PositionSnapshot(qty=10, avg_entry_price=100.0)}, [], None,
                                     entry_estimates=estimates)
    assert result.realized_pnls == [pytest.approx(20.0)]
    assert result.realized_returns == [pytest.approx(0.02)]
    fill_event = next(c for c in audit_spy.call_args_list if c.args[0] == "entry_filled")
    assert fill_event.kwargs["actual_slippage_bps"] == pytest.approx(5.0)
    assert fill_event.kwargs["slippage_error_bps"] == pytest.approx(4.0)
    assert estimates == {}
