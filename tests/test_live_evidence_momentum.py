import numpy as np
import pandas as pd

from src.backtest.evidence_momentum import EvidenceMomentumConfig
from src.runtime.state_store import ManagedPositionSnapshot
from src.strategy.evidence_momentum import (
    EvidenceMomentumSnapshot,
    advance_managed_position,
    evidence_momentum_lookback_days,
    seed_managed_position,
)


def test_evidence_momentum_lookback_days_covers_long_history():
    config = EvidenceMomentumConfig(
        long_sma_sessions=200,
        high_lookback_sessions=252,
    )
    assert evidence_momentum_lookback_days(config, timeframe_minutes=60) >= 200


def test_seed_managed_position_uses_atr_targets():
    config = EvidenceMomentumConfig(
        atr_stop_mult=2.0,
        atr_trailing_stop_mult=3.0,
    )
    snapshot = EvidenceMomentumSnapshot(
        timestamp=pd.Timestamp("2026-04-29T14:00:00Z"),
        close=100.0,
        high=101.5,
        low=99.5,
        atr=2.0,
        score=1.25,
        raw_score=0.9,
        high_ratio=0.96,
        rebalance_due=True,
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
    assert managed.target_price == 108.0
    assert managed.highest_high == 101.5


def test_advance_managed_position_trails_stop_forward_for_momentum():
    config = EvidenceMomentumConfig(
        atr_stop_mult=2.0,
        atr_trailing_stop_mult=2.5,
    )
    position = ManagedPositionSnapshot(
        qty=10.0,
        entry_price=100.0,
        entry_time="2026-04-29T14:00:00+00:00",
        entry_atr=2.0,
        stop_price=96.0,
        target_price=108.0,
        highest_high=101.0,
        lowest_low=99.0,
        holding_bars=3,
    )
    snapshot = EvidenceMomentumSnapshot(
        timestamp=pd.Timestamp("2026-04-29T15:00:00Z"),
        close=104.0,
        high=106.0,
        low=103.5,
        atr=1.6,
        score=1.5,
        raw_score=1.0,
        high_ratio=0.98,
        rebalance_due=False,
        long_entry_setup=False,
        long_exit_signal=False,
    )

    updated, exit_reason = advance_managed_position(position, snapshot, config)

    assert exit_reason is None
    assert updated.holding_bars == 4
    assert updated.highest_high == 106.0
    assert np.isclose(updated.stop_price, 102.0)
