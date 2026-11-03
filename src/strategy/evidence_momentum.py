"""Live helpers for the evidence-backed momentum strategy.

The PDF review pointed away from retail-style oscillators and toward the small
set of strategy families that have held up more often in top-tier finance
journals after costs: cross-sectional momentum, time-series momentum, and
52-week-high persistence. This module keeps the live path aligned with that
research engine while leaving execution and portfolio risk controls in the
shared broker/risk layers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import numpy as np
import pandas as pd

from config.settings import settings
from src.backtest.evidence_momentum import (
    EvidenceMomentumConfig,
    build_indicator_frame,
    sessions_to_bars,
)
from src.runtime.state_store import ManagedPositionSnapshot


@dataclass(frozen=True, slots=True)
class EvidenceMomentumSnapshot:
    timestamp: pd.Timestamp
    close: float
    high: float
    low: float
    atr: float
    score: float
    raw_score: float
    high_ratio: float
    rebalance_due: bool
    long_entry_setup: bool
    long_exit_signal: bool


def build_live_config() -> EvidenceMomentumConfig:
    return EvidenceMomentumConfig(
        short_lookback_sessions=settings.em_short_lookback_sessions,
        medium_lookback_sessions=settings.em_medium_lookback_sessions,
        long_lookback_sessions=settings.em_long_lookback_sessions,
        skip_recent_sessions=settings.em_skip_recent_sessions,
        long_sma_sessions=settings.em_long_sma_sessions,
        high_lookback_sessions=settings.em_high_lookback_sessions,
        vol_lookback_sessions=settings.em_vol_lookback_sessions,
        min_high_ratio=settings.em_min_high_ratio,
        min_momentum_score=settings.em_min_momentum_score,
        require_market_regime=settings.em_require_market_regime,
        market_regime_buffer_pct=settings.em_market_regime_buffer_pct,
        rebalance_interval_sessions=settings.em_rebalance_interval_sessions,
        top_n=settings.em_top_n,
        atr_length=settings.em_atr_length,
        atr_stop_mult=settings.em_atr_stop_mult,
        atr_trailing_stop_mult=settings.em_atr_trailing_stop_mult,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_open_positions=settings.max_open_positions,
        circuit_breaker_drawdown_pct=settings.em_circuit_breaker_drawdown_pct,
        allow_short=False,
    )


def evidence_momentum_lookback_days(
    config: EvidenceMomentumConfig,
    timeframe_minutes: int | None = None,
) -> int:
    timeframe = timeframe_minutes or settings.bar_timeframe_minutes
    bars_per_session = max(1.0, 390.0 / float(timeframe))
    required_bars = max(
        sessions_to_bars(config.long_sma_sessions, timeframe),
        sessions_to_bars(config.high_lookback_sessions, timeframe),
        sessions_to_bars(config.vol_lookback_sessions, timeframe),
        sessions_to_bars(
            config.long_lookback_sessions + max(config.skip_recent_sessions, 0),
            timeframe,
        ),
        config.atr_length,
    ) + 5
    sessions_needed = ceil(required_bars / bars_per_session)
    return max(40, sessions_needed + 5)


def latest_snapshot(frame: pd.DataFrame) -> EvidenceMomentumSnapshot | None:
    if frame.empty:
        return None
    row = frame.iloc[-1]
    required_columns = [
        "close",
        "high",
        "low",
        "atr",
        "score",
        "raw_score",
        "high_ratio",
        "rebalance_due",
        "long_entry_setup",
        "long_exit_signal",
    ]
    if any(pd.isna(row.get(column)) for column in required_columns[:-3]):
        return None

    timestamp = frame.index[-1]
    if "timestamp" in frame.columns:
        timestamp = row["timestamp"]

    return EvidenceMomentumSnapshot(
        timestamp=pd.Timestamp(timestamp),
        close=float(row["close"]),
        high=float(row["high"]),
        low=float(row["low"]),
        atr=float(row["atr"]),
        score=float(row["score"]),
        raw_score=float(row["raw_score"]),
        high_ratio=float(row["high_ratio"]),
        rebalance_due=bool(row["rebalance_due"]),
        long_entry_setup=bool(row["long_entry_setup"]),
        long_exit_signal=bool(row["long_exit_signal"]),
    )


def build_snapshot(
    bars: pd.DataFrame,
    proxy_bars: pd.DataFrame,
    config: EvidenceMomentumConfig,
) -> EvidenceMomentumSnapshot | None:
    frame = build_indicator_frame(
        bars,
        proxy_bars=proxy_bars,
        config=config,
        timeframe_minutes=settings.bar_timeframe_minutes,
    )
    return latest_snapshot(frame)


def seed_managed_position(
    qty: float,
    entry_price: float,
    snapshot: EvidenceMomentumSnapshot,
    config: EvidenceMomentumConfig,
) -> ManagedPositionSnapshot:
    atr = snapshot.atr if np.isfinite(snapshot.atr) and snapshot.atr > 0 else entry_price * 0.01
    stop_price = entry_price - (config.atr_stop_mult * atr)
    target_price = entry_price + (2.0 * config.atr_stop_mult * atr)
    return ManagedPositionSnapshot(
        qty=float(qty),
        entry_price=float(entry_price),
        entry_time=snapshot.timestamp.isoformat(),
        entry_atr=float(atr),
        stop_price=float(stop_price),
        target_price=float(target_price),
        highest_high=max(snapshot.high, entry_price),
        lowest_low=min(snapshot.low, entry_price),
        holding_bars=0,
    )


def advance_managed_position(
    position: ManagedPositionSnapshot,
    snapshot: EvidenceMomentumSnapshot,
    config: EvidenceMomentumConfig,
) -> tuple[ManagedPositionSnapshot, str | None]:
    highest_high = max(position.highest_high, snapshot.high)
    lowest_low = min(position.lowest_low, snapshot.low)
    holding_bars = position.holding_bars + 1

    updated = replace(
        position,
        highest_high=highest_high,
        lowest_low=lowest_low,
        holding_bars=holding_bars,
    )

    if snapshot.low <= position.stop_price:
        return updated, "STOP"
    if snapshot.high >= position.target_price:
        return updated, "TARGET"
    if snapshot.long_exit_signal:
        return updated, "SIGNAL"

    current_atr = snapshot.atr
    if np.isfinite(current_atr) and current_atr > 0:
        trailing_stop = highest_high - (config.atr_trailing_stop_mult * current_atr)
        initial_stop = position.entry_price - (config.atr_stop_mult * position.entry_atr)
        updated = replace(
            updated,
            stop_price=max(position.stop_price, initial_stop, trailing_stop),
            target_price=max(
                position.target_price,
                position.entry_price + (2.0 * config.atr_stop_mult * position.entry_atr),
            ),
        )

    return updated, None
