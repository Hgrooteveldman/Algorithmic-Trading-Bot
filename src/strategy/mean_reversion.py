"""Live helpers for the volatility-adjusted mean-reversion strategy.

The research engine in `src.backtest.volatility_mean_reversion` already owns
the indicator math. This module keeps the live path thin by reusing that math
and only adding the stateful pieces that a broker-connected bot needs:

- how much history to fetch for the configured timeframe
- how to translate the latest closed bar into an actionable snapshot
- how to persist and advance ATR-based trailing state across loop iterations
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import numpy as np
import pandas as pd

from config.settings import settings
from src.backtest.volatility_mean_reversion import MeanReversionConfig, build_indicator_frame
from src.runtime.state_store import ManagedPositionSnapshot


@dataclass(frozen=True, slots=True)
class MeanReversionSnapshot:
    timestamp: pd.Timestamp
    close: float
    high: float
    low: float
    atr: float
    bb_mid: float
    zscore: float
    long_entry_setup: bool
    long_exit_signal: bool


def build_live_config() -> MeanReversionConfig:
    return MeanReversionConfig(
        bollinger_window=settings.mr_bollinger_window,
        bollinger_std_mult=settings.mr_bollinger_std_mult,
        keltner_window=settings.mr_keltner_window,
        keltner_atr_mult=settings.mr_keltner_atr_mult,
        rsi_length=settings.mr_rsi_length,
        rsi_entry_threshold=settings.mr_rsi_entry_threshold,
        rsi_exit_threshold=settings.mr_rsi_exit_threshold,
        regime_window=settings.mr_regime_window,
        regime_buffer_pct=settings.mr_regime_buffer_pct,
        regime_slope_lookback=settings.mr_regime_slope_lookback,
        regime_max_down_slope_pct=settings.mr_regime_max_down_slope_pct,
        require_asset_regime=settings.mr_require_asset_regime,
        asset_regime_buffer_pct=settings.mr_asset_regime_buffer_pct,
        atr_length=settings.mr_atr_length,
        atr_stop_mult=settings.mr_atr_stop_mult,
        atr_trailing_stop_mult=settings.mr_atr_trailing_stop_mult,
        atr_target_mult=settings.mr_atr_target_mult,
        mean_exit_atr_buffer=settings.mr_mean_exit_atr_buffer,
        max_holding_bars=settings.mr_max_holding_bars,
        require_reversal_bar=settings.mr_require_reversal_bar,
        entry_zscore_threshold=settings.mr_entry_zscore_threshold,
        entry_reclaim_atr_buffer=settings.mr_entry_reclaim_atr_buffer,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_open_positions=settings.max_open_positions,
        circuit_breaker_drawdown_pct=settings.mr_circuit_breaker_drawdown_pct,
        allow_short=False,
    )


def mean_reversion_lookback_days(
    config: MeanReversionConfig,
    timeframe_minutes: int | None = None,
) -> int:
    timeframe = timeframe_minutes or settings.bar_timeframe_minutes
    bars_per_session = max(1.0, 390.0 / float(timeframe))
    required_bars = max(
        config.regime_window,
        config.bollinger_window,
        config.keltner_window,
        config.atr_length,
        config.rsi_length,
    ) + max(config.regime_slope_lookback, config.max_holding_bars) + 5
    sessions_needed = ceil(required_bars / bars_per_session)
    return max(20, sessions_needed + 5)


def latest_snapshot(frame: pd.DataFrame) -> MeanReversionSnapshot | None:
    if frame.empty:
        return None
    row = frame.iloc[-1]
    required_columns = [
        "close",
        "high",
        "low",
        "atr",
        "bb_mid",
        "zscore",
        "long_entry_setup",
        "long_exit_signal",
    ]
    if any(pd.isna(row.get(column)) for column in required_columns):
        return None

    timestamp = frame.index[-1]
    if "timestamp" in frame.columns:
        timestamp = row["timestamp"]

    return MeanReversionSnapshot(
        timestamp=pd.Timestamp(timestamp),
        close=float(row["close"]),
        high=float(row["high"]),
        low=float(row["low"]),
        atr=float(row["atr"]),
        bb_mid=float(row["bb_mid"]),
        zscore=float(row["zscore"]),
        long_entry_setup=bool(row["long_entry_setup"]),
        long_exit_signal=bool(row["long_exit_signal"]),
    )


def build_snapshot(
    bars: pd.DataFrame,
    proxy_bars: pd.DataFrame,
    config: MeanReversionConfig,
) -> MeanReversionSnapshot | None:
    frame = build_indicator_frame(bars, proxy_bars=proxy_bars, config=config)
    return latest_snapshot(frame)


def mean_target_price(
    snapshot: MeanReversionSnapshot,
    config: MeanReversionConfig,
) -> float:
    return snapshot.bb_mid + (config.mean_exit_atr_buffer * snapshot.atr)


def seed_managed_position(
    qty: float,
    entry_price: float,
    snapshot: MeanReversionSnapshot,
    config: MeanReversionConfig,
) -> ManagedPositionSnapshot:
    atr = snapshot.atr if np.isfinite(snapshot.atr) and snapshot.atr > 0 else entry_price * 0.01
    stop_price = entry_price - (config.atr_stop_mult * atr)
    target_price = max(
        entry_price + (config.atr_target_mult * atr),
        mean_target_price(snapshot, config),
    )
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
    snapshot: MeanReversionSnapshot,
    config: MeanReversionConfig,
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
    if holding_bars >= config.max_holding_bars:
        return updated, "TIME"

    current_atr = snapshot.atr
    if np.isfinite(current_atr) and current_atr > 0:
        trailing_stop = highest_high - (config.atr_trailing_stop_mult * current_atr)
        initial_stop = position.entry_price - (config.atr_stop_mult * position.entry_atr)
        updated = replace(
            updated,
            stop_price=max(position.stop_price, initial_stop, trailing_stop),
            target_price=max(
                position.target_price,
                position.entry_price + (config.atr_target_mult * position.entry_atr),
                mean_target_price(snapshot, config),
            ),
        )

    return updated, None
