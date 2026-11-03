import pandas as pd

from src.backtest.volatility_mean_reversion import MeanReversionConfig
from src.runtime.state_store import ManagedPositionSnapshot
from src.strategy.mean_reversion import (
    MeanReversionSnapshot,
    advance_managed_position,
    mean_reversion_lookback_days,
    seed_managed_position,
)


def test_mean_reversion_lookback_days_covers_regime_history():
    config = MeanReversionConfig(regime_window=200)
    assert mean_reversion_lookback_days(config, timeframe_minutes=60) >= 30


def test_seed_managed_position_uses_atr_based_levels():
    config = MeanReversionConfig(
        atr_stop_mult=2.0,
        atr_target_mult=1.5,
        mean_exit_atr_buffer=0.2,
    )
    snapshot = MeanReversionSnapshot(
        timestamp=pd.Timestamp("2026-04-29T14:00:00Z"),
        close=100.0,
        high=101.0,
        low=99.0,
        atr=2.0,
        bb_mid=101.0,
        zscore=-2.0,
        long_entry_setup=True,
        long_exit_signal=False,
    )

    managed = seed_managed_position(
        qty=10,
        entry_price=100.0,
        snapshot=snapshot,
        config=config,
    )

    assert managed.stop_price == 96.0
    assert managed.target_price == 103.0
    assert managed.highest_high == 101.0


def test_advance_managed_position_trails_stop_forward():
    config = MeanReversionConfig(
        atr_stop_mult=1.8,
        atr_trailing_stop_mult=2.0,
        atr_target_mult=1.1,
        mean_exit_atr_buffer=0.15,
        max_holding_bars=18,
    )
    position = ManagedPositionSnapshot(
        qty=10.0,
        entry_price=100.0,
        entry_time="2026-04-29T14:00:00+00:00",
        entry_atr=2.0,
        stop_price=96.4,
        target_price=110.0,
        highest_high=100.5,
        lowest_low=99.5,
        holding_bars=4,
    )
    snapshot = MeanReversionSnapshot(
        timestamp=pd.Timestamp("2026-04-29T15:00:00Z"),
        close=102.0,
        high=104.0,
        low=101.0,
        atr=1.5,
        bb_mid=101.0,
        zscore=-1.0,
        long_entry_setup=False,
        long_exit_signal=False,
    )

    updated, exit_reason = advance_managed_position(position, snapshot, config)

    assert exit_reason is None
    assert updated.holding_bars == 5
    assert updated.stop_price == 101.0
    assert updated.highest_high == 104.0
