"""Evidence-backed momentum backtester and optimizer.

This module pivots the research path toward the small set of strategy families
that the supplied literature review judged most credible after costs:

1. Cross-sectional momentum:
   Rank liquid assets by medium-horizon returns and favor the strongest names.

2. Time-series momentum / trend:
   Only hold assets whose own trend is positive and whose broad market proxy is
   not in a structural downtrend.

3. Low-turnover implementation:
   Rebalance on a slower cadence than every bar and use persistence features
   such as 52-week-high proximity to avoid hyperactive churn.

This is not a claim of guaranteed alpha. It is an attempt to align the bot's
research engine with the document's methodological conclusions:
    - retail RSI / MA-cross / candlestick rules are weak after costs
    - momentum and trend survive more often, but only with crash/risk controls
    - turnover, trading friction, and multiple-testing discipline matter
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import ceil, floor, isinf
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger


def bars_per_year(timeframe_minutes: int) -> float:
    return 252.0 * (390.0 / timeframe_minutes)


def sessions_to_bars(sessions: int, timeframe_minutes: int) -> int:
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    bars_per_session = max(1.0, 390.0 / float(timeframe_minutes))
    return max(1, int(ceil(sessions * bars_per_session)))


@dataclass(frozen=True, slots=True)
class EvidenceMomentumConfig:
    short_lookback_sessions: int = 21
    medium_lookback_sessions: int = 63
    long_lookback_sessions: int = 126
    skip_recent_sessions: int = 5
    long_sma_sessions: int = 200
    high_lookback_sessions: int = 252
    vol_lookback_sessions: int = 21

    min_high_ratio: float = 0.85
    min_momentum_score: float = 0.0
    require_market_regime: bool = True
    market_regime_buffer_pct: float = 0.02
    rebalance_interval_sessions: int = 5
    top_n: int = 2

    atr_length: int = 14
    atr_stop_mult: float = 2.0
    atr_trailing_stop_mult: float = 3.0

    risk_per_trade_pct: float = 0.003
    max_position_pct: float = 0.18
    max_gross_exposure_pct: float = 0.75
    max_open_positions: int = 3
    circuit_breaker_drawdown_pct: float = 0.25

    allow_short: bool = False

    def label(self) -> str:
        return (
            f"mom{self.short_lookback_sessions}-{self.medium_lookback_sessions}-{self.long_lookback_sessions}"
            f"_skip{self.skip_recent_sessions}"
            f"_top{self.top_n}"
            f"_reb{self.rebalance_interval_sessions}"
            f"_atr{self.atr_stop_mult:.1f}-{self.atr_trailing_stop_mult:.1f}"
        ).replace(".", "p")


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    side: int
    qty: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_index: int
    entry_atr: float
    stop_price: float
    target_price: float
    highest_high: float
    lowest_low: float
    holding_bars: int = 0


@dataclass(slots=True)
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    buy_and_hold_return: float
    outperformance_vs_buy_and_hold: float
    profit_factor: float
    average_win: float
    average_loss: float
    exposure_pct: float
    circuit_breaker_triggered: bool

    def summary(self) -> str:
        profit_factor = "inf" if isinf(self.profit_factor) else f"{self.profit_factor:.2f}"
        return (
            f"Trades:            {self.num_trades}\n"
            f"Total return:      {self.total_return:.2%}\n"
            f"Annualized return: {self.annualized_return:.2%}\n"
            f"Buy & hold:        {self.buy_and_hold_return:.2%}\n"
            f"Vs buy/hold:       {self.outperformance_vs_buy_and_hold:+.2%}\n"
            f"Sharpe (ann.):     {self.sharpe:.2f}\n"
            f"Max drawdown:      {self.max_drawdown:.2%}\n"
            f"Win rate:          {self.win_rate:.2%}\n"
            f"Profit factor:     {profit_factor}\n"
            f"Avg win/loss:      ${self.average_win:,.2f} / ${self.average_loss:,.2f}\n"
            f"Exposure:          {self.exposure_pct:.2%}\n"
            f"Circuit breaker:   {'TRIGGERED' if self.circuit_breaker_triggered else 'not triggered'}"
        )


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _momentum_return(close: pd.Series, lookback_bars: int, skip_bars: int) -> pd.Series:
    anchor = close.shift(lookback_bars + skip_bars)
    reference = close.shift(skip_bars)
    return (reference / anchor) - 1.0


def build_indicator_frame(
    bars: pd.DataFrame,
    proxy_bars: pd.DataFrame,
    config: EvidenceMomentumConfig,
    timeframe_minutes: int,
) -> pd.DataFrame:
    df = bars.sort_index().copy()
    proxy = proxy_bars.sort_index()[["close"]].rename(columns={"close": "proxy_close"})
    df = df.join(proxy, how="left")
    df["proxy_close"] = df["proxy_close"].ffill().bfill()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    proxy_close = df["proxy_close"].astype(float)

    short_bars = sessions_to_bars(config.short_lookback_sessions, timeframe_minutes)
    medium_bars = sessions_to_bars(config.medium_lookback_sessions, timeframe_minutes)
    long_bars = sessions_to_bars(config.long_lookback_sessions, timeframe_minutes)
    # Zero is a valid optimizer choice here: it means "measure momentum all the
    # way through the latest bar" instead of skipping the most recent session.
    skip_bars = (
        0
        if config.skip_recent_sessions <= 0
        else sessions_to_bars(config.skip_recent_sessions, timeframe_minutes)
    )
    long_sma_bars = sessions_to_bars(config.long_sma_sessions, timeframe_minutes)
    high_bars = sessions_to_bars(config.high_lookback_sessions, timeframe_minutes)
    vol_bars = sessions_to_bars(config.vol_lookback_sessions, timeframe_minutes)
    rebalance_interval_bars = sessions_to_bars(
        config.rebalance_interval_sessions,
        timeframe_minutes,
    )

    atr = _wilder_atr(high, low, close, config.atr_length)
    realized_vol = close.pct_change().rolling(vol_bars).std() * np.sqrt(
        bars_per_year(timeframe_minutes)
    )

    mom_short = _momentum_return(close, short_bars, skip_bars)
    mom_medium = _momentum_return(close, medium_bars, skip_bars)
    mom_long = _momentum_return(close, long_bars, skip_bars)
    proxy_mom_long = _momentum_return(proxy_close, long_bars, skip_bars)

    asset_long_sma = close.rolling(long_sma_bars).mean()
    proxy_long_sma = proxy_close.rolling(long_sma_bars).mean()
    high_ratio = close / close.rolling(high_bars).max()

    raw_score = (
        (0.30 * mom_short)
        + (0.40 * mom_medium)
        + (0.30 * mom_long)
        + (0.10 * (high_ratio - 1.0))
    )
    score = raw_score / realized_vol.replace(0.0, np.nan)

    proxy_regime_ok = (
        (proxy_close >= proxy_long_sma * (1 - config.market_regime_buffer_pct))
        & (proxy_mom_long > 0)
    )
    trend_ok = (close >= asset_long_sma) & (mom_long > 0)
    long_entry_setup = (
        trend_ok
        & (mom_medium > 0)
        & (high_ratio >= config.min_high_ratio)
        & (score >= config.min_momentum_score)
    )
    if config.require_market_regime:
        long_entry_setup = long_entry_setup & proxy_regime_ok

    long_exit_signal = (
        (close < asset_long_sma)
        | (mom_short < 0)
        | (raw_score < 0)
    )
    if config.require_market_regime:
        long_exit_signal = long_exit_signal | (~proxy_regime_ok)

    short_entry_setup = (
        config.allow_short
        & (close < asset_long_sma)
        & (mom_medium < 0)
        & (raw_score < 0)
        & (high_ratio <= min(config.min_high_ratio, 0.95))
    )
    short_exit_signal = (
        (close > asset_long_sma)
        | (mom_short > 0)
        | (raw_score > 0)
    )

    df["atr"] = atr
    df["realized_vol"] = realized_vol
    df["mom_short"] = mom_short
    df["mom_medium"] = mom_medium
    df["mom_long"] = mom_long
    df["proxy_mom_long"] = proxy_mom_long
    df["asset_long_sma"] = asset_long_sma
    df["proxy_long_sma"] = proxy_long_sma
    df["high_ratio"] = high_ratio
    df["raw_score"] = raw_score
    df["score"] = score
    df["proxy_regime_ok"] = proxy_regime_ok
    df["trend_ok"] = trend_ok
    df["rebalance_due"] = (np.arange(len(df)) % rebalance_interval_bars) == 0
    df["long_entry_setup"] = long_entry_setup.fillna(False)
    df["long_exit_signal"] = long_exit_signal.fillna(False)
    df["short_entry_setup"] = short_entry_setup.fillna(False)
    df["short_exit_signal"] = short_exit_signal.fillna(False)
    return df


def _result_from_components(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    starting_cash: float,
    buy_and_hold_return: float,
    exposure_pct: float,
    bars_per_year_value: float,
    circuit_breaker_triggered: bool,
) -> BacktestResult:
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / starting_cash - 1 if len(equity) else 0.0
    if len(equity) > 1 and equity.iloc[-1] > 0 and starting_cash > 0:
        annualized_return = float(
            (equity.iloc[-1] / starting_cash) ** (bars_per_year_value / max(len(equity), 1)) - 1
        )
    else:
        annualized_return = 0.0
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(bars_per_year_value))
        if returns.std() > 0
        else 0.0
    )
    drawdown = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    win_rate = float((trades_df["pnl"] > 0).mean()) if len(trades_df) else 0.0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if len(trades_df) else 0.0
    gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) if len(trades_df) else 0.0
    profit_factor = (
        float(gross_profit / gross_loss)
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    average_win = (
        float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean())
        if len(trades_df) and (trades_df["pnl"] > 0).any()
        else 0.0
    )
    average_loss = (
        float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean())
        if len(trades_df) and (trades_df["pnl"] < 0).any()
        else 0.0
    )
    return BacktestResult(
        equity_curve=equity,
        trades=trades_df,
        total_return=float(total_return),
        annualized_return=annualized_return,
        sharpe=sharpe,
        max_drawdown=drawdown,
        win_rate=win_rate,
        num_trades=len(trades_df),
        buy_and_hold_return=float(buy_and_hold_return),
        outperformance_vs_buy_and_hold=float(total_return - buy_and_hold_return),
        profit_factor=profit_factor,
        average_win=average_win,
        average_loss=average_loss,
        exposure_pct=float(exposure_pct),
        circuit_breaker_triggered=circuit_breaker_triggered,
    )


def backtest_evidence_momentum_portfolio(
    bars_by_symbol: dict[str, pd.DataFrame],
    proxy_bars: pd.DataFrame,
    config: EvidenceMomentumConfig,
    timeframe_minutes: int,
    starting_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    bars_per_year_value: float = 252.0 * 6.5,
) -> BacktestResult:
    frames: dict[str, pd.DataFrame] = {}
    events: list[tuple[pd.Timestamp, str, int]] = []
    latest_prices: dict[str, float] = {}

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        frame = build_indicator_frame(
            bars=bars,
            proxy_bars=proxy_bars,
            config=config,
            timeframe_minutes=timeframe_minutes,
        ).reset_index(drop=False)
        if "timestamp" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        frames[symbol] = frame
        latest_prices[symbol] = float(frame["close"].iloc[0])
        events.extend((frame.iloc[idx]["timestamp"], symbol, idx) for idx in range(1, len(frame)))

    if not events:
        return _result_from_components(
            equity=pd.Series(dtype=float),
            trades_df=pd.DataFrame(),
            starting_cash=starting_cash,
            buy_and_hold_return=0.0,
            exposure_pct=0.0,
            bars_per_year_value=bars_per_year_value,
            circuit_breaker_triggered=False,
        )

    cost_per_side = (commission_bps + slippage_bps) / 10_000.0
    rebalance_interval_bars = sessions_to_bars(
        config.rebalance_interval_sessions,
        timeframe_minutes,
    )
    cash = starting_cash
    positions: dict[str, OpenPosition] = {}
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_index: list[pd.Timestamp] = []
    exposure_samples: list[float] = []

    peak_equity = starting_cash
    circuit_breaker_triggered = False
    entries_halted = False

    def portfolio_equity() -> float:
        return cash + sum(
            position.side * position.qty * latest_prices.get(symbol, position.entry_price)
            for symbol, position in positions.items()
        )

    def gross_exposure() -> float:
        return sum(
            abs(position.qty * latest_prices.get(symbol, position.entry_price))
            for symbol, position in positions.items()
        )

    event_groups: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for timestamp, symbol, idx in events:
        event_groups.setdefault(timestamp, []).append((symbol, idx))

    for timestamp in sorted(event_groups):
        symbol_events = event_groups[timestamp]
        for symbol, idx in symbol_events:
            latest_prices[symbol] = float(frames[symbol].iloc[idx]["close"])

        equity_before_entries = portfolio_equity()
        peak_equity = max(peak_equity, equity_before_entries)
        if peak_equity > 0 and equity_before_entries <= peak_equity * (1 - config.circuit_breaker_drawdown_pct):
            entries_halted = True
            circuit_breaker_triggered = True

        for symbol, idx in symbol_events:
            position = positions.get(symbol)
            if position is None:
                continue

            row = frames[symbol].iloc[idx]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            position.holding_bars += 1
            position.highest_high = max(position.highest_high, high)
            position.lowest_low = min(position.lowest_low, low)

            exit_trigger = None
            reported_exit_price = None
            reason = ""

            if position.side == 1:
                if low <= position.stop_price:
                    exit_trigger = position.stop_price
                    reported_exit_price = position.stop_price
                    reason = "STOP"
                elif high >= position.target_price:
                    exit_trigger = position.target_price
                    reported_exit_price = position.target_price
                    reason = "TARGET"
                elif bool(row["long_exit_signal"]):
                    exit_trigger = close
                    reported_exit_price = close * (1 - cost_per_side)
                    reason = "SIGNAL"
            else:
                if high >= position.stop_price:
                    exit_trigger = position.stop_price
                    reported_exit_price = position.stop_price
                    reason = "STOP"
                elif low <= position.target_price:
                    exit_trigger = position.target_price
                    reported_exit_price = position.target_price
                    reason = "TARGET"
                elif bool(row["short_exit_signal"]):
                    exit_trigger = close
                    reported_exit_price = close * (1 + cost_per_side)
                    reason = "SIGNAL"

            if exit_trigger is None:
                current_atr = float(row["atr"])
                if np.isfinite(current_atr) and current_atr > 0:
                    if position.side == 1:
                        trailing_stop = position.highest_high - (config.atr_trailing_stop_mult * current_atr)
                        initial_stop = position.entry_price - (config.atr_stop_mult * position.entry_atr)
                        position.stop_price = max(position.stop_price, initial_stop, trailing_stop)
                        target_price = position.entry_price + (2.0 * config.atr_stop_mult * position.entry_atr)
                        position.target_price = max(position.target_price, target_price)
                    else:
                        trailing_stop = position.lowest_low + (config.atr_trailing_stop_mult * current_atr)
                        initial_stop = position.entry_price + (config.atr_stop_mult * position.entry_atr)
                        position.stop_price = min(position.stop_price, initial_stop, trailing_stop)
                        target_price = position.entry_price - (2.0 * config.atr_stop_mult * position.entry_atr)
                        position.target_price = min(position.target_price, target_price)
                continue

            if position.side == 1:
                proceeds = position.qty * exit_trigger * (1 - cost_per_side)
                pnl = proceeds - (position.qty * position.entry_price)
                cash += proceeds
                pnl_pct = (exit_trigger / position.entry_price) - 1.0
                side_label = "LONG"
            else:
                cover_cost = position.qty * exit_trigger * (1 + cost_per_side)
                pnl = (position.qty * position.entry_price) - cover_cost
                cash -= cover_cost
                pnl_pct = (position.entry_price - (exit_trigger * (1 + cost_per_side))) / position.entry_price
                side_label = "SHORT"
            trades.append(
                {
                    "symbol": symbol,
                    "side": side_label,
                    "entry_time": position.entry_time,
                    "exit_time": timestamp,
                    "entry": position.entry_price,
                    "exit": reported_exit_price if reported_exit_price is not None else exit_trigger,
                    "qty": position.qty,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "hold_bars": position.holding_bars,
                    "reason": reason,
                }
            )
            positions.pop(symbol, None)

        rebalance_due = any((idx % rebalance_interval_bars) == 0 for _, idx in symbol_events)
        if rebalance_due:
            long_rankings: list[tuple[float, str, int]] = []
            short_rankings: list[tuple[float, str, int]] = []
            for symbol, idx in symbol_events:
                row = frames[symbol].iloc[idx]
                score = row.get("score")
                atr = row.get("atr")
                if pd.isna(score) or pd.isna(atr):
                    continue
                if bool(row["long_entry_setup"]):
                    long_rankings.append((float(score), symbol, idx))
                if config.allow_short and bool(row["short_entry_setup"]):
                    short_rankings.append((float(score), symbol, idx))

            long_rankings.sort(key=lambda item: (-item[0], item[1]))
            short_rankings.sort(key=lambda item: (item[0], item[1]))

            desired_longs = {symbol for _, symbol, _ in long_rankings[: config.top_n]}
            desired_shorts = {symbol for _, symbol, _ in short_rankings[: config.top_n]}

            for symbol, idx in symbol_events:
                position = positions.get(symbol)
                if position is None:
                    continue
                row = frames[symbol].iloc[idx]
                close = float(row["close"])
                if position.side == 1 and symbol not in desired_longs:
                    exit_price = close * (1 - cost_per_side)
                    cash += position.qty * exit_price
                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "LONG",
                            "entry_time": position.entry_time,
                            "exit_time": timestamp,
                            "entry": position.entry_price,
                            "exit": exit_price,
                            "qty": position.qty,
                            "pnl": position.qty * (exit_price - position.entry_price),
                            "pnl_pct": (exit_price / position.entry_price) - 1.0,
                            "hold_bars": position.holding_bars,
                            "reason": "RANK",
                        }
                    )
                    positions.pop(symbol, None)
                elif position.side == -1 and symbol not in desired_shorts:
                    exit_price = close * (1 + cost_per_side)
                    cash -= position.qty * exit_price
                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "SHORT",
                            "entry_time": position.entry_time,
                            "exit_time": timestamp,
                            "entry": position.entry_price,
                            "exit": exit_price,
                            "qty": position.qty,
                            "pnl": (position.qty * position.entry_price) - (position.qty * exit_price),
                            "pnl_pct": (position.entry_price - exit_price) / position.entry_price,
                            "hold_bars": position.holding_bars,
                            "reason": "RANK",
                        }
                    )
                    positions.pop(symbol, None)

            if not entries_halted:
                entry_candidates: list[tuple[float, str, int, int]] = []
                for score, symbol, idx in long_rankings[: config.top_n]:
                    if symbol not in positions:
                        entry_candidates.append((abs(score), symbol, idx, 1))
                for score, symbol, idx in short_rankings[: config.top_n]:
                    if symbol not in positions:
                        entry_candidates.append((abs(score), symbol, idx, -1))
                entry_candidates.sort(key=lambda item: (-item[0], item[1], -item[3]))

                for _, symbol, idx, side in entry_candidates:
                    if symbol in positions or len(positions) >= config.max_open_positions:
                        continue
                    row = frames[symbol].iloc[idx]
                    close = float(row["close"])
                    high = float(row["high"])
                    low = float(row["low"])
                    atr = float(row["atr"])
                    if not np.isfinite(atr) or atr <= 0 or close <= 0:
                        continue

                    equity = portfolio_equity()
                    current_gross_exposure = gross_exposure()
                    available_notional = max((equity * config.max_gross_exposure_pct) - current_gross_exposure, 0.0)
                    entry_price = close * (1 + cost_per_side) if side == 1 else close * (1 - cost_per_side)
                    risk_budget = equity * config.risk_per_trade_pct
                    per_share_risk = (config.atr_stop_mult * atr) + (entry_price * (2 * cost_per_side))
                    qty_by_risk = floor(risk_budget / per_share_risk) if per_share_risk > 0 else 0
                    qty_by_position_cap = floor((equity * config.max_position_pct) / entry_price)
                    qty_by_available = (
                        floor(min(cash, available_notional) / entry_price)
                        if side == 1
                        else floor(available_notional / max(close, 1e-9))
                    )
                    qty = max(min(qty_by_risk, qty_by_position_cap, qty_by_available), 0)
                    if qty <= 0:
                        continue

                    if side == 1:
                        cash -= qty * entry_price
                        stop_price = entry_price - (config.atr_stop_mult * atr)
                        target_price = entry_price + (2.0 * config.atr_stop_mult * atr)
                    else:
                        cash += qty * entry_price
                        stop_price = entry_price + (config.atr_stop_mult * atr)
                        target_price = entry_price - (2.0 * config.atr_stop_mult * atr)

                    positions[symbol] = OpenPosition(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        entry_price=entry_price,
                        entry_time=timestamp,
                        entry_index=idx,
                        entry_atr=atr,
                        stop_price=stop_price,
                        target_price=target_price,
                        highest_high=high,
                        lowest_low=low,
                    )

        equity = portfolio_equity()
        equity_curve.append(equity)
        equity_index.append(timestamp)
        exposure_samples.append(gross_exposure() / equity if equity > 0 else 0.0)

    for symbol, position in list(positions.items()):
        frame = frames[symbol]
        final_bar = frame.iloc[-1]
        final_time = final_bar["timestamp"]
        final_close = float(final_bar["close"])
        if position.side == 1:
            exit_price = final_close * (1 - cost_per_side)
            pnl = (position.qty * exit_price) - (position.qty * position.entry_price)
            cash += position.qty * exit_price
            pnl_pct = (exit_price / position.entry_price) - 1.0
            side_label = "LONG"
        else:
            exit_price = final_close * (1 + cost_per_side)
            pnl = (position.qty * position.entry_price) - (position.qty * exit_price)
            cash -= position.qty * exit_price
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
            side_label = "SHORT"
        trades.append(
            {
                "symbol": symbol,
                "side": side_label,
                "entry_time": position.entry_time,
                "exit_time": final_time,
                "entry": position.entry_price,
                "exit": exit_price,
                "qty": position.qty,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "hold_bars": position.holding_bars,
                "reason": "END",
            }
        )

    if equity_curve:
        equity_curve[-1] = cash

    buy_and_hold_returns = []
    for frame in frames.values():
        if len(frame) > 1:
            buy_and_hold_returns.append(float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1))
    buy_and_hold_return = float(np.mean(buy_and_hold_returns)) if buy_and_hold_returns else 0.0

    return _result_from_components(
        equity=pd.Series(equity_curve, index=equity_index),
        trades_df=pd.DataFrame(trades),
        starting_cash=starting_cash,
        buy_and_hold_return=buy_and_hold_return,
        exposure_pct=float(np.mean(exposure_samples)) if exposure_samples else 0.0,
        bars_per_year_value=bars_per_year_value,
        circuit_breaker_triggered=circuit_breaker_triggered,
    )


def default_optimization_grid(
    timeframe_minutes: int,
    profile: str = "balanced",
) -> dict[str, list]:
    profile = profile.strip().lower()
    if profile not in {"balanced", "aggressive", "active"}:
        raise ValueError("profile must be 'balanced', 'aggressive', or 'active'")

    if profile == "active":
        return {
            "short_lookback_sessions": [5, 10, 21],
            "medium_lookback_sessions": [21, 42, 63],
            "long_lookback_sessions": [63, 84, 126],
            "skip_recent_sessions": [0, 2, 5],
            "min_high_ratio": [0.75, 0.80, 0.85, 0.90],
            "rebalance_interval_sessions": [1, 2, 3],
            "top_n": [2, 3, 4],
            "atr_stop_mult": [1.2, 1.5, 2.0],
            "atr_trailing_stop_mult": [1.8, 2.2, 2.6],
            "risk_per_trade_pct": [0.0025, 0.0040, 0.0060],
            "max_position_pct": [0.10, 0.15, 0.18],
            "max_gross_exposure_pct": [0.60, 0.80, 1.00],
            "max_open_positions": [3, 4],
            "allow_short": [False, True],
        }

    if profile == "aggressive":
        return {
            "short_lookback_sessions": [21, 42],
            "medium_lookback_sessions": [63, 84, 126],
            "long_lookback_sessions": [126, 189, 252],
            "skip_recent_sessions": [0, 5, 10],
            "min_high_ratio": [0.80, 0.90, 0.95],
            "rebalance_interval_sessions": [1, 5, 10],
            "top_n": [1, 2, 3],
            "atr_stop_mult": [1.5, 2.0, 2.5],
            "atr_trailing_stop_mult": [2.5, 3.0, 3.5],
            "risk_per_trade_pct": [0.002, 0.004, 0.006],
            "max_position_pct": [0.10, 0.15, 0.20],
            "max_gross_exposure_pct": [0.40, 0.60, 0.80],
            "allow_short": [False, True],
        }

    return {
        "short_lookback_sessions": [21, 42],
        "medium_lookback_sessions": [63, 84],
        "long_lookback_sessions": [126, 189],
        "skip_recent_sessions": [0, 5],
        "min_high_ratio": [0.85, 0.90],
        "rebalance_interval_sessions": [1, 5],
        "top_n": [1, 2],
        "atr_stop_mult": [1.5, 2.0, 2.5],
        "atr_trailing_stop_mult": [2.5, 3.0],
        "risk_per_trade_pct": [0.0025, 0.0040],
        "max_position_pct": [0.10, 0.15],
        "max_gross_exposure_pct": [0.40, 0.60],
        "allow_short": [False],
    }


def generate_candidate_configs(
    base_config: EvidenceMomentumConfig,
    parameter_grid: dict[str, Iterable] | None = None,
) -> list[EvidenceMomentumConfig]:
    grid = parameter_grid or default_optimization_grid(timeframe_minutes=60)
    keys = list(grid.keys())
    if not keys:
        return [base_config]
    configs: list[EvidenceMomentumConfig] = []
    for values in product(*(grid[key] for key in keys)):
        config = base_config
        for key, value in zip(keys, values):
            config = replace(config, **{key: value})
        configs.append(config)
    return configs


def limit_candidate_configs(
    candidate_configs: list[EvidenceMomentumConfig],
    max_configs: int | None,
) -> list[EvidenceMomentumConfig]:
    if not max_configs or max_configs <= 0 or len(candidate_configs) <= max_configs:
        return candidate_configs
    indices = np.linspace(0, len(candidate_configs) - 1, num=max_configs, dtype=int)
    unique_indices = []
    seen = set()
    for index in indices.tolist():
        if index not in seen:
            seen.add(index)
            unique_indices.append(index)
    return [candidate_configs[index] for index in unique_indices]


def default_min_trade_count(timeframe_minutes: int) -> int:
    if timeframe_minutes <= 5:
        return 400
    if timeframe_minutes <= 15:
        return 180
    if timeframe_minutes <= 30:
        return 120
    return 80


def _optimization_score(
    result: BacktestResult,
    target_annual_return: float,
    max_drawdown_limit: float,
    evaluation_mode: str = "balanced",
    min_trade_count: int = 0,
) -> tuple:
    trade_goal_met = result.num_trades >= max(min_trade_count, 0)
    meets_target = (
        result.annualized_return >= target_annual_return
        and abs(result.max_drawdown) <= max_drawdown_limit
        and trade_goal_met
    )
    annual_return_gap = target_annual_return - result.annualized_return
    drawdown_gap = abs(result.max_drawdown) - max_drawdown_limit
    trade_gap = max(min_trade_count - result.num_trades, 0)
    penalty = max(annual_return_gap, 0.0) * 4.0 + max(drawdown_gap, 0.0) * 3.0
    if evaluation_mode == "active" and min_trade_count > 0:
        penalty += (trade_gap / min_trade_count) * 0.75
    if evaluation_mode == "active":
        return (
            1 if meets_target else 0,
            1 if trade_goal_met else 0,
            result.annualized_return - penalty,
            result.num_trades,
            result.exposure_pct,
            result.sharpe,
            result.profit_factor,
            -abs(result.max_drawdown),
        )
    return (
        1 if meets_target else 0,
        result.annualized_return - penalty,
        result.sharpe,
        result.profit_factor,
        -abs(result.max_drawdown),
    )


def optimize_evidence_momentum(
    bars_by_symbol: dict[str, pd.DataFrame],
    proxy_bars: pd.DataFrame,
    base_config: EvidenceMomentumConfig,
    timeframe_minutes: int,
    parameter_grid: dict[str, Iterable] | None = None,
    target_annual_return: float = 0.08,
    max_drawdown_limit: float = 0.25,
    starting_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    bars_per_year_value: float = 252.0 * 6.5,
    max_configs: int | None = None,
    progress_every: int = 100,
    evaluation_mode: str = "balanced",
    min_trade_count: int | None = None,
) -> tuple[pd.DataFrame, EvidenceMomentumConfig, BacktestResult]:
    evaluation_mode = evaluation_mode.strip().lower()
    if evaluation_mode not in {"balanced", "active"}:
        raise ValueError("evaluation_mode must be 'balanced' or 'active'")
    if min_trade_count is None:
        min_trade_count = (
            default_min_trade_count(timeframe_minutes)
            if evaluation_mode == "active"
            else 0
        )

    candidate_configs = generate_candidate_configs(base_config, parameter_grid=parameter_grid)
    full_grid_count = len(candidate_configs)
    if max_configs is None and full_grid_count > 50_000:
        max_configs = min(2_000, full_grid_count)
        logger.warning(
            "Optimizer grid contains {:,} combinations. Auto-limiting this run to {:,}.",
            full_grid_count,
            max_configs,
        )
    candidate_configs = limit_candidate_configs(candidate_configs, max_configs=max_configs)
    logger.info(
        "Optimizer evaluating {} configuration(s){}.",
        len(candidate_configs),
        (
            f" out of {full_grid_count} total grid combinations"
            if len(candidate_configs) != full_grid_count
            else ""
        ),
    )

    rows: list[dict] = []
    best_result: BacktestResult | None = None
    best_config: EvidenceMomentumConfig | None = None
    best_score: tuple | None = None
    total_candidates = len(candidate_configs)
    progress_every = max(int(progress_every), 1)

    for index, config in enumerate(candidate_configs, start=1):
        result = backtest_evidence_momentum_portfolio(
            bars_by_symbol=bars_by_symbol,
            proxy_bars=proxy_bars,
            config=config,
            timeframe_minutes=timeframe_minutes,
            starting_cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            bars_per_year_value=bars_per_year_value,
        )
        score = _optimization_score(
            result,
            target_annual_return=target_annual_return,
            max_drawdown_limit=max_drawdown_limit,
            evaluation_mode=evaluation_mode,
            min_trade_count=min_trade_count,
        )
        trade_goal_met = result.num_trades >= max(min_trade_count, 0)
        meets_target = (
            result.annualized_return >= target_annual_return
            and abs(result.max_drawdown) <= max_drawdown_limit
            and trade_goal_met
        )
        row = asdict(config)
        row.update(
            {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "buy_and_hold_return": result.buy_and_hold_return,
                "outperformance_vs_buy_and_hold": result.outperformance_vs_buy_and_hold,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
                "win_rate": result.win_rate,
                "num_trades": result.num_trades,
                "exposure_pct": result.exposure_pct,
                "circuit_breaker_triggered": result.circuit_breaker_triggered,
                "meets_trade_goal": trade_goal_met,
                "evaluation_mode": evaluation_mode,
                "min_trade_count_target": min_trade_count,
                "meets_target": meets_target,
            }
        )
        rows.append(row)
        if best_score is None or score > best_score:
            best_score = score
            best_config = config
            best_result = result
        if index == 1 or index == total_candidates or index % progress_every == 0:
            logger.info(
                "Optimizer progress: {}/{} ({:.1%}) | best annualized={:+.2%} | best drawdown={:.2%}",
                index,
                total_candidates,
                index / total_candidates,
                best_result.annualized_return if best_result is not None else 0.0,
                best_result.max_drawdown if best_result is not None else 0.0,
            )

    if best_config is None or best_result is None:
        raise RuntimeError("Optimization produced no candidate results.")

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        sort_columns = ["meets_target", "annualized_return", "sharpe", "profit_factor", "max_drawdown"]
        ascending = [False, False, False, False, False]
        if evaluation_mode == "active":
            sort_columns = [
                "meets_target",
                "meets_trade_goal",
                "annualized_return",
                "num_trades",
                "exposure_pct",
                "sharpe",
                "profit_factor",
                "max_drawdown",
            ]
            ascending = [False, False, False, False, False, False, False, False]
        results_df = results_df.sort_values(
            by=sort_columns,
            ascending=ascending,
        ).reset_index(drop=True)
        results_df.insert(0, "rank", range(1, len(results_df) + 1))
    return results_df, best_config, best_result
