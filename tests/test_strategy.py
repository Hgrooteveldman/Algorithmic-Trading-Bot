import pandas as pd

from src.strategy.base import Signal
from src.strategy.sma_crossover import (
    passes_relative_strength_filter,
    relative_strength_snapshot,
    signal_for_index,
)


def test_signal_exit_ignores_minor_fast_ma_wobble():
    bars = pd.DataFrame(
        [
            {
                "close": 101.0,
                "short_sma": 100.5,
                "long_sma": 99.5,
                "trend_sma": 98.0,
                "trend_slope": 0.5,
                "atr_pct": 0.01,
                "pullback_low": 100.0,
            },
            {
                "close": 100.2,
                "short_sma": 100.8,
                "long_sma": 99.8,
                "trend_sma": 98.4,
                "trend_slope": 0.4,
                "atr_pct": 0.01,
                "pullback_low": 99.9,
            },
        ]
    )

    signal = signal_for_index(
        bars,
        idx=1,
        pullback_reclaim_tolerance_pct=0.001,
        min_atr_pct=0.002,
        max_atr_pct=0.03,
    )

    assert signal == Signal.HOLD


def test_signal_exit_triggers_on_confirmed_trend_break():
    bars = pd.DataFrame(
        [
            {
                "close": 97.5,
                "short_sma": 100.5,
                "long_sma": 99.5,
                "trend_sma": 98.0,
                "trend_slope": 0.2,
                "atr_pct": 0.01,
                "pullback_low": 97.0,
            },
            {
                "close": 96.8,
                "short_sma": 100.1,
                "long_sma": 99.4,
                "trend_sma": 97.6,
                "trend_slope": 0.1,
                "atr_pct": 0.01,
                "pullback_low": 96.5,
            },
        ]
    )

    signal = signal_for_index(
        bars,
        idx=1,
        pullback_reclaim_tolerance_pct=0.001,
        min_atr_pct=0.002,
        max_atr_pct=0.03,
    )

    assert signal == Signal.SELL


def test_relative_strength_snapshot_ranks_symbols_by_recent_return():
    strong = pd.DataFrame({"close": [100.0, 101.0, 102.0, 105.0]})
    weak = pd.DataFrame({"close": [100.0, 100.4, 100.8, 101.0]})

    snapshot = relative_strength_snapshot(
        {"GLD": strong, "SPY": weak},
        idx_by_symbol={"GLD": 3, "SPY": 3},
        lookback_bars=2,
    )

    assert snapshot["GLD"].rank == 1
    assert snapshot["SPY"].rank == 2
    assert snapshot["GLD"].score > snapshot["SPY"].score
    assert passes_relative_strength_filter(snapshot["GLD"], max_rank=1, min_return=0.0)
    assert not passes_relative_strength_filter(snapshot["SPY"], max_rank=1, min_return=0.0)
