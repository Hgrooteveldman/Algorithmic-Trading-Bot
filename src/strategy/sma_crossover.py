"""Trend-following pullback strategy.

This started life as a plain SMA crossover, but the current version is more
selective:

    - Only buys in an established uptrend
    - Waits for a pullback back into the fast moving average
    - Requires a reclaim of that moving average before entering
    - Filters out unusually dead or chaotic volatility regimes with ATR%

The goal is not to promise alpha, but to reduce noisy entries compared with
buying every moving-average crossover.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.strategy.base import Signal, Strategy
from config.settings import settings


@dataclass(frozen=True, slots=True)
class RelativeStrengthSnapshot:
    score: float
    rank: int


def build_signal_frame(
    bars: pd.DataFrame,
    short: int,
    long: int,
    trend_filter: int,
    pullback_lookback: int,
    trend_slope_lookback: int,
    atr_window: int,
) -> pd.DataFrame:
    df = bars.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["short_sma"] = close.rolling(short).mean()
    df["long_sma"] = close.rolling(long).mean()
    if trend_filter:
        df["trend_sma"] = close.rolling(trend_filter).mean()
        trend_anchor = df["trend_sma"]
    else:
        df["trend_sma"] = df["long_sma"]
        trend_anchor = df["long_sma"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.rolling(atr_window).mean()
    df["atr_pct"] = df["atr"] / close
    df["trend_slope"] = trend_anchor - trend_anchor.shift(trend_slope_lookback)
    df["pullback_low"] = low.rolling(pullback_lookback).min()
    return df


def relative_strength_score(
    bars: pd.DataFrame,
    idx: int,
    lookback_bars: int,
) -> float | None:
    if lookback_bars <= 0 or idx < lookback_bars or "close" not in bars:
        return None

    current_close = bars.iloc[idx]["close"]
    baseline_close = bars.iloc[idx - lookback_bars]["close"]
    if pd.isna(current_close) or pd.isna(baseline_close):
        return None

    current_close = float(current_close)
    baseline_close = float(baseline_close)
    if baseline_close <= 0:
        return None
    return (current_close / baseline_close) - 1.0


def relative_strength_snapshot(
    bars_by_symbol: dict[str, pd.DataFrame],
    idx_by_symbol: dict[str, int],
    lookback_bars: int,
) -> dict[str, RelativeStrengthSnapshot]:
    scores: dict[str, float] = {}
    for symbol, idx in idx_by_symbol.items():
        bars = bars_by_symbol.get(symbol)
        if bars is None or bars.empty:
            continue
        score = relative_strength_score(bars, idx=idx, lookback_bars=lookback_bars)
        if score is None:
            continue
        scores[symbol] = score

    ranked_symbols = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    return {
        symbol: RelativeStrengthSnapshot(score=scores[symbol], rank=rank)
        for rank, symbol in enumerate(ranked_symbols, start=1)
    }


def passes_relative_strength_filter(
    snapshot: RelativeStrengthSnapshot | None,
    max_rank: int,
    min_return: float,
) -> bool:
    if max_rank <= 0:
        return True
    if snapshot is None:
        return False
    return snapshot.rank <= max_rank and snapshot.score >= min_return


def signal_for_index(
    bars: pd.DataFrame,
    idx: int,
    pullback_reclaim_tolerance_pct: float,
    min_atr_pct: float,
    max_atr_pct: float,
) -> Signal:
    if idx < 1:
        return Signal.HOLD

    current = bars.iloc[idx]
    previous = bars.iloc[idx - 1]
    required = [
        "close",
        "short_sma",
        "long_sma",
        "trend_sma",
        "trend_slope",
        "atr_pct",
        "pullback_low",
    ]
    if any(pd.isna(current[column]) or pd.isna(previous[column]) for column in required):
        return Signal.HOLD

    tolerance = 1 + pullback_reclaim_tolerance_pct
    in_uptrend = (
        current["close"] > current["trend_sma"]
        and current["short_sma"] > current["long_sma"]
        and current["trend_slope"] > 0
    )
    pullback_recent = previous["pullback_low"] <= previous["short_sma"] * tolerance
    reclaim = (
        previous["close"] <= previous["short_sma"] * tolerance
        and current["close"] > current["short_sma"]
    )
    follow_through = current["close"] > previous["close"]
    volatility_ok = min_atr_pct <= current["atr_pct"] <= max_atr_pct

    if in_uptrend and pullback_recent and reclaim and follow_through and volatility_ok:
        logger.debug(
            "Trend pullback BUY: close={:.2f}, short={:.2f}, long={:.2f}, atr_pct={:.2%}",
            current["close"],
            current["short_sma"],
            current["long_sma"],
            current["atr_pct"],
        )
        return Signal.BUY

    trend_break = (
        current["close"] < current["trend_sma"]
        and previous["close"] < previous["trend_sma"]
    )
    crossover_break = (
        current["short_sma"] < current["long_sma"]
        and current["close"] < current["long_sma"]
    )

    # Be patient on exits. The stop-loss / take-profit already protect the
    # trade, so the strategy-level SELL should represent a genuine regime
    # deterioration rather than a brief wobble under the fast moving average.
    if trend_break or crossover_break:
        logger.debug(
            "Trend pullback SELL: close={:.2f}, short={:.2f}, long={:.2f}, trend={:.2f}",
            current["close"],
            current["short_sma"],
            current["long_sma"],
            current["trend_sma"],
        )
        return Signal.SELL

    return Signal.HOLD


class TrendPullbackStrategy(Strategy):
    def __init__(
        self,
        short: int | None = None,
        long: int | None = None,
        trend_filter: int | None = None,
    ):
        self.short = short or settings.short_sma_window
        self.long = long or settings.long_sma_window
        self.trend_filter = (
            trend_filter
            if trend_filter is not None
            else settings.regime_filter_window
        )
        self.pullback_lookback = settings.pullback_lookback_bars
        self.trend_slope_lookback = settings.trend_slope_lookback_bars
        self.atr_window = settings.atr_window
        self.min_atr_pct = settings.min_atr_pct
        self.max_atr_pct = settings.max_atr_pct
        self.pullback_reclaim_tolerance_pct = settings.pullback_reclaim_tolerance_pct
        if self.short >= self.long:
            raise ValueError("short window must be < long window")
        if self.trend_filter and self.trend_filter <= self.long:
            raise ValueError("trend_filter window must be > long window (or 0 to disable)")

    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        min_bars = (
            max(
                self.long,
                self.trend_filter,
                self.pullback_lookback,
                self.trend_slope_lookback,
                self.atr_window,
            )
            + 2
        )
        if len(bars) < min_bars:
            return Signal.HOLD

        frame = build_signal_frame(
            bars=bars,
            short=self.short,
            long=self.long,
            trend_filter=self.trend_filter,
            pullback_lookback=self.pullback_lookback,
            trend_slope_lookback=self.trend_slope_lookback,
            atr_window=self.atr_window,
        )
        return signal_for_index(
            frame,
            idx=len(frame) - 1,
            pullback_reclaim_tolerance_pct=self.pullback_reclaim_tolerance_pct,
            min_atr_pct=self.min_atr_pct,
            max_atr_pct=self.max_atr_pct,
        )


class SMACrossover(TrendPullbackStrategy):
    """Backward-compatible alias for the evolved default strategy."""
