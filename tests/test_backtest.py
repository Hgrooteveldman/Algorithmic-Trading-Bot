import numpy as np
import pandas as pd

from src.backtest import backtester as backtester_module
from src.backtest.backtester import backtest_portfolio, backtest_sma
from src.strategy.base import Signal


def test_backtest_summary_contains_richer_metrics():
    periods = 320
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    base = 100 + np.linspace(0, 25, periods) + 1.5 * np.sin(np.linspace(0, 30, periods))
    bars = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.8,
            "low": base - 0.8,
            "close": base + 0.2 * np.sin(np.linspace(0, 12, periods)),
            "volume": np.full(periods, 1_000_000),
        },
        index=index,
    )

    result = backtest_sma(bars, bars_per_year=252 * 7)
    summary = result.summary()

    assert "Buy & hold" in summary
    assert "Profit factor" in summary
    assert "Exposure" in summary
    assert 0.0 <= result.exposure_pct <= 1.0
    assert isinstance(result.trades, pd.DataFrame)


def test_portfolio_backtest_supports_multiple_symbols():
    periods = 320
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    bars_by_symbol = {}
    for offset, symbol in enumerate(["SPY", "QQQ"]):
        base = 100 + offset * 5 + np.linspace(0, 20, periods)
        wave = 1.2 * np.sin(np.linspace(0, 24, periods) + offset)
        close = base + wave
        bars_by_symbol[symbol] = pd.DataFrame(
            {
                "open": close - 0.3,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": np.full(periods, 1_000_000),
            },
            index=index,
        )

    result = backtest_portfolio(bars_by_symbol, bars_per_year=252 * 7)

    assert isinstance(result.trades, pd.DataFrame)
    assert 0.0 <= result.exposure_pct <= 1.0
    assert "Buy & hold" in result.summary()


def test_portfolio_backtest_prefers_stronger_symbol_when_rank_filter_enabled(monkeypatch):
    periods = 8
    target_idx = periods - 2
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")

    def make_bars(close_values):
        close = np.array(close_values, dtype=float)
        return pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": np.full(periods, 1_000_000),
            },
            index=index,
        )

    bars_by_symbol = {
        "GLD": make_bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]),
        "SPY": make_bars([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4]),
    }

    def fake_signal(frame, idx, **_kwargs):
        return Signal.BUY if idx == target_idx else Signal.HOLD

    monkeypatch.setattr(backtester_module, "signal_for_index", fake_signal)

    result = backtest_portfolio(
        bars_by_symbol,
        short_window=2,
        long_window=3,
        trend_filter_window=4,
        pullback_lookback=2,
        trend_slope_lookback=1,
        atr_window=2,
        bars_per_year=252 * 7,
        relative_strength_lookback=3,
        max_relative_strength_rank=1,
        min_relative_strength_return=0.0,
    )

    assert set(result.trades["symbol"]) == {"GLD"}
