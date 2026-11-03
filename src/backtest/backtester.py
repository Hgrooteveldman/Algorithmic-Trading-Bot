"""Event-driven backtester for the default trend strategy.

Limits to be aware of (read these before trusting any result):
    - Fills happen at the close of the signal bar with a flat slippage cost.
      Real markets have variable slippage that depends on size and volatility.
    - Stop and target are checked against the bar's high/low, ignoring
      intra-bar path. If both are hit in the same bar, we pessimistically
      assume the stop fired first.
    - Commissions modelled as a flat basis-point cost per side. Alpaca's
      stock commissions are zero, but slippage and bid/ask spread are real.
    - No survivorship bias correction, no dividend/corporate-action handling.
    - A profitable backtest is NECESSARY but NOT SUFFICIENT. Validate with
      out-of-sample data, walk-forward analysis, and weeks of paper trading
      before risking real capital.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import List

import numpy as np
import pandas as pd

from src.strategy.base import Signal
from src.strategy.sma_crossover import (
    RelativeStrengthSnapshot,
    build_signal_frame,
    passes_relative_strength_filter,
    relative_strength_score,
    relative_strength_snapshot,
    signal_for_index,
)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    total_return: float
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

    def summary(self) -> str:
        profit_factor = "inf" if isinf(self.profit_factor) else f"{self.profit_factor:.2f}"
        return (
            f"Trades:        {self.num_trades}\n"
            f"Total return:  {self.total_return:.2%}\n"
            f"Buy & hold:    {self.buy_and_hold_return:.2%}\n"
            f"Vs buy/hold:   {self.outperformance_vs_buy_and_hold:+.2%}\n"
            f"Sharpe (ann.): {self.sharpe:.2f}\n"
            f"Max drawdown:  {self.max_drawdown:.2%}\n"
            f"Win rate:      {self.win_rate:.2%}\n"
            f"Profit factor: {profit_factor}\n"
            f"Avg win/loss:  ${self.average_win:,.2f} / ${self.average_loss:,.2f}\n"
            f"Exposure:      {self.exposure_pct:.2%}"
        )


def _result_from_components(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    starting_cash: float,
    buy_and_hold_return: float,
    exposure_pct: float,
    bars_per_year: int,
) -> BacktestResult:
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / starting_cash - 1 if len(equity) else 0
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0
    drawdown = (equity / equity.cummax() - 1).min() if len(equity) else 0
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
        if (len(trades_df) and (trades_df["pnl"] > 0).any())
        else 0.0
    )
    average_loss = (
        float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean())
        if (len(trades_df) and (trades_df["pnl"] < 0).any())
        else 0.0
    )
    return BacktestResult(
        equity_curve=equity,
        trades=trades_df,
        total_return=float(total_return),
        sharpe=float(sharpe),
        max_drawdown=float(drawdown),
        win_rate=win_rate,
        num_trades=len(trades_df),
        buy_and_hold_return=float(buy_and_hold_return),
        outperformance_vs_buy_and_hold=float(total_return - buy_and_hold_return),
        profit_factor=profit_factor,
        average_win=average_win,
        average_loss=average_loss,
        exposure_pct=float(exposure_pct),
    )


def backtest_sma(
    bars: pd.DataFrame,
    starting_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    short_window: int = 30,
    long_window: int = 100,
    trend_filter_window: int = 200,
    pullback_lookback: int = 3,
    trend_slope_lookback: int = 5,
    atr_window: int = 14,
    min_atr_pct: float = 0.002,
    max_atr_pct: float = 0.030,
    pullback_reclaim_tolerance_pct: float = 0.001,
    relative_strength_lookback: int = 20,
    max_relative_strength_rank: int = 3,
    min_relative_strength_return: float = 0.0,
    max_position_pct: float = 0.10,
    risk_per_trade_pct: float = 0.0015,
    bars_per_year: int = 252 * 78,  # ~5-min bars in a US trading year
) -> BacktestResult:
    df = build_signal_frame(
        bars=bars,
        short=short_window,
        long=long_window,
        trend_filter=trend_filter_window,
        pullback_lookback=pullback_lookback,
        trend_slope_lookback=trend_slope_lookback,
        atr_window=atr_window,
    ).reset_index(drop=False)

    cost_per_side = (commission_bps + slippage_bps) / 10_000
    effective_stop_pct = stop_loss_pct + (2 * cost_per_side)
    target_position_pct = min(
        risk_per_trade_pct / max(effective_stop_pct, 1e-9),
        max_position_pct,
    )

    cash = starting_cash
    qty = 0
    entry_price = 0.0
    stop = 0.0
    target = 0.0
    entry_time = None
    entry_index = 0

    equity_curve: List[float] = []
    equity_index: List = []
    trades: List[dict] = []
    bars_in_market = 0

    for i in range(1, len(df)):
        bar = df.iloc[i]
        timestamp = bar["timestamp"] if "timestamp" in bar else bar.name
        price = float(bar["close"])
        high, low = float(bar["high"]), float(bar["low"])

        # --- 1) Manage the open position first ---
        if qty > 0:
            bars_in_market += 1
            exit_price = None
            reason = ""
            # Pessimistic assumption: if both touched, stop fires first.
            if low <= stop:
                exit_price, reason = stop, "STOP"
            elif high >= target:
                exit_price, reason = target, "TARGET"
            if exit_price is not None:
                proceeds = qty * exit_price * (1 - cost_per_side)
                cost_basis = qty * entry_price
                pnl = proceeds - cost_basis
                cash += proceeds
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry": entry_price,
                        "exit": exit_price,
                        "qty": qty,
                        "pnl": pnl,
                        "pnl_pct": (exit_price / entry_price) - 1,
                        "hold_bars": i - entry_index,
                        "reason": reason,
                    }
                )
                qty = 0

        # --- 2) Generate new entries / exits on signals ---
        signal = signal_for_index(
            df,
            idx=i,
            pullback_reclaim_tolerance_pct=pullback_reclaim_tolerance_pct,
            min_atr_pct=min_atr_pct,
            max_atr_pct=max_atr_pct,
        )
        strength_score = relative_strength_score(
            df,
            idx=i,
            lookback_bars=relative_strength_lookback,
        )
        strength_snapshot = (
            RelativeStrengthSnapshot(score=strength_score, rank=1)
            if strength_score is not None
            else None
        )

        if signal == Signal.BUY and qty == 0:
            if not passes_relative_strength_filter(
                strength_snapshot,
                max_rank=max_relative_strength_rank,
                min_return=min_relative_strength_return,
            ):
                continue
            entry_price = price * (1 + cost_per_side)
            qty = int((cash * target_position_pct) // entry_price)
            if qty > 0:
                cash -= qty * entry_price
                stop = round(entry_price * (1 - stop_loss_pct), 2)
                target = round(entry_price * (1 + take_profit_pct), 2)
                entry_time = timestamp
                entry_index = i
        elif signal == Signal.SELL and qty > 0:
            exit_price = price * (1 - cost_per_side)
            cash += qty * exit_price
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": qty * (exit_price - entry_price),
                    "pnl_pct": (exit_price / entry_price) - 1,
                    "hold_bars": i - entry_index,
                    "reason": "SIGNAL",
                }
            )
            qty = 0

        equity_curve.append(cash + qty * price)
        equity_index.append(timestamp)

    if qty > 0 and len(df):
        final_bar = df.iloc[-1]
        final_time = final_bar["timestamp"] if "timestamp" in final_bar else final_bar.name
        final_price = float(final_bar["close"]) * (1 - cost_per_side)
        cash += qty * final_price
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": final_time,
                "entry": entry_price,
                "exit": final_price,
                "qty": qty,
                "pnl": qty * (final_price - entry_price),
                "pnl_pct": (final_price / entry_price) - 1,
                "hold_bars": max(len(df) - 1 - entry_index, 0),
                "reason": "END",
            }
        )
        qty = 0
        if equity_curve:
            equity_curve[-1] = cash

    # --- Metrics ---
    equity = pd.Series(equity_curve, index=equity_index)
    buy_and_hold_return = (
        df["close"].iloc[-1] / df["close"].iloc[0] - 1
        if len(df) > 1
        else 0.0
    )
    trades_df = pd.DataFrame(trades)
    exposure_pct = bars_in_market / max(len(df) - 1, 1) if len(df) else 0.0

    return _result_from_components(
        equity=equity,
        trades_df=trades_df,
        starting_cash=starting_cash,
        buy_and_hold_return=buy_and_hold_return,
        exposure_pct=exposure_pct,
        bars_per_year=bars_per_year,
    )


def backtest_portfolio(
    bars_by_symbol: dict[str, pd.DataFrame],
    starting_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    short_window: int = 30,
    long_window: int = 100,
    trend_filter_window: int = 200,
    pullback_lookback: int = 3,
    trend_slope_lookback: int = 5,
    atr_window: int = 14,
    min_atr_pct: float = 0.002,
    max_atr_pct: float = 0.030,
    pullback_reclaim_tolerance_pct: float = 0.001,
    relative_strength_lookback: int = 20,
    max_relative_strength_rank: int = 3,
    min_relative_strength_return: float = 0.0,
    max_position_pct: float = 0.10,
    max_gross_exposure_pct: float = 0.40,
    max_open_positions: int = 4,
    risk_per_trade_pct: float = 0.0015,
    bars_per_year: int = 252 * 78,
) -> BacktestResult:
    frames: dict[str, pd.DataFrame] = {}
    events: list[tuple[pd.Timestamp, str, int]] = []
    latest_prices: dict[str, float] = {}

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        frame = build_signal_frame(
            bars=bars,
            short=short_window,
            long=long_window,
            trend_filter=trend_filter_window,
            pullback_lookback=pullback_lookback,
            trend_slope_lookback=trend_slope_lookback,
            atr_window=atr_window,
        ).reset_index(drop=False)
        if "timestamp" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        frames[symbol] = frame
        latest_prices[symbol] = float(frame["close"].iloc[0])
        events.extend(
            [
                (frame.iloc[idx]["timestamp"], symbol, idx)
                for idx in range(1, len(frame))
            ]
        )

    if not events:
        empty_series = pd.Series(dtype=float)
        empty_trades = pd.DataFrame()
        return _result_from_components(
            equity=empty_series,
            trades_df=empty_trades,
            starting_cash=starting_cash,
            buy_and_hold_return=0.0,
            exposure_pct=0.0,
            bars_per_year=bars_per_year,
        )

    events.sort(key=lambda item: (item[0], item[1]))
    cost_per_side = (commission_bps + slippage_bps) / 10_000
    effective_stop_pct = stop_loss_pct + (2 * cost_per_side)
    target_position_pct = min(
        risk_per_trade_pct / max(effective_stop_pct, 1e-9),
        max_position_pct,
    )

    cash = starting_cash
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_index: list[pd.Timestamp] = []
    exposure_samples: list[float] = []

    def portfolio_equity() -> float:
        return cash + sum(
            position["qty"] * latest_prices.get(symbol, position["entry_price"])
            for symbol, position in positions.items()
        )

    def portfolio_gross_exposure() -> float:
        return sum(
            position["qty"] * latest_prices.get(symbol, position["entry_price"])
            for symbol, position in positions.items()
        )

    event_groups: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for timestamp, symbol, idx in events:
        event_groups.setdefault(timestamp, []).append((symbol, idx))

    for timestamp in sorted(event_groups):
        symbol_events = event_groups[timestamp]
        for symbol, idx in symbol_events:
            latest_prices[symbol] = float(frames[symbol].iloc[idx]["close"])

        signal_by_symbol = {
            symbol: signal_for_index(
                frames[symbol],
                idx=idx,
                pullback_reclaim_tolerance_pct=pullback_reclaim_tolerance_pct,
                min_atr_pct=min_atr_pct,
                max_atr_pct=max_atr_pct,
            )
            for symbol, idx in symbol_events
        }
        strength_by_symbol = relative_strength_snapshot(
            frames,
            idx_by_symbol={symbol: idx for symbol, idx in symbol_events},
            lookback_bars=relative_strength_lookback,
        )

        for symbol, idx in symbol_events:
            position = positions.get(symbol)
            if position is None:
                continue
            bar = frames[symbol].iloc[idx]
            high = float(bar["high"])
            low = float(bar["low"])
            exit_price = None
            reason = ""
            if low <= position["stop"]:
                exit_price, reason = position["stop"], "STOP"
            elif high >= position["target"]:
                exit_price, reason = position["target"], "TARGET"
            if exit_price is None:
                continue
            proceeds = position["qty"] * exit_price * (1 - cost_per_side)
            pnl = proceeds - (position["qty"] * position["entry_price"])
            cash += proceeds
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "entry": position["entry_price"],
                    "exit": exit_price,
                    "qty": position["qty"],
                    "pnl": pnl,
                    "pnl_pct": (exit_price / position["entry_price"]) - 1,
                    "hold_bars": idx - position["entry_index"],
                    "reason": reason,
                }
            )
            positions.pop(symbol, None)

        for symbol, idx in symbol_events:
            bar = frames[symbol].iloc[idx]
            signal = signal_by_symbol[symbol]
            price = float(bar["close"])

            if signal == Signal.SELL and symbol in positions:
                position = positions.pop(symbol)
                exit_price = price * (1 - cost_per_side)
                cash += position["qty"] * exit_price
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry": position["entry_price"],
                        "exit": exit_price,
                        "qty": position["qty"],
                        "pnl": position["qty"] * (exit_price - position["entry_price"]),
                        "pnl_pct": (exit_price / position["entry_price"]) - 1,
                        "hold_bars": idx - position["entry_index"],
                        "reason": "SIGNAL",
                    }
                )
                continue

        buy_candidates: list[tuple[float, str, int]] = []
        for symbol, idx in symbol_events:
            if signal_by_symbol[symbol] != Signal.BUY or symbol in positions:
                continue
            strength_snapshot = strength_by_symbol.get(symbol)
            if not passes_relative_strength_filter(
                strength_snapshot,
                max_rank=max_relative_strength_rank,
                min_return=min_relative_strength_return,
            ):
                continue
            score = strength_snapshot.score if strength_snapshot is not None else 0.0
            buy_candidates.append((score, symbol, idx))

        buy_candidates.sort(key=lambda item: (-item[0], item[1]))

        for _, symbol, idx in buy_candidates:
            if symbol in positions or len(positions) >= max_open_positions:
                continue
            bar = frames[symbol].iloc[idx]
            price = float(bar["close"])
            equity = portfolio_equity()
            gross_exposure = portfolio_gross_exposure()
            available_notional = max((equity * max_gross_exposure_pct) - gross_exposure, 0.0)
            desired_notional = min(
                equity * target_position_pct,
                available_notional,
                cash,
            )
            entry_price = price * (1 + cost_per_side)
            qty = int(desired_notional // entry_price)
            if qty <= 0:
                continue
            cash -= qty * entry_price
            positions[symbol] = {
                "qty": qty,
                "entry_price": entry_price,
                "stop": round(entry_price * (1 - stop_loss_pct), 2),
                "target": round(entry_price * (1 + take_profit_pct), 2),
                "entry_time": timestamp,
                "entry_index": idx,
            }

        equity = portfolio_equity()
        gross_exposure = portfolio_gross_exposure()
        equity_curve.append(equity)
        equity_index.append(timestamp)
        exposure_samples.append(gross_exposure / equity if equity > 0 else 0.0)

    for symbol, position in list(positions.items()):
        frame = frames[symbol]
        final_bar = frame.iloc[-1]
        final_time = final_bar["timestamp"]
        final_price = float(final_bar["close"]) * (1 - cost_per_side)
        cash += position["qty"] * final_price
        trades.append(
            {
                "symbol": symbol,
                "entry_time": position["entry_time"],
                "exit_time": final_time,
                "entry": position["entry_price"],
                "exit": final_price,
                "qty": position["qty"],
                "pnl": position["qty"] * (final_price - position["entry_price"]),
                "pnl_pct": (final_price / position["entry_price"]) - 1,
                "hold_bars": max(len(frame) - 1 - position["entry_index"], 0),
                "reason": "END",
            }
        )

    if equity_curve:
        equity_curve[-1] = cash

    buy_and_hold_return = float(
        np.mean(
            [
                frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
                for frame in frames.values()
                if len(frame) > 1
            ]
        )
    )
    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, index=equity_index)
    exposure_pct = float(np.mean(exposure_samples)) if exposure_samples else 0.0

    return _result_from_components(
        equity=equity_series,
        trades_df=trades_df,
        starting_cash=starting_cash,
        buy_and_hold_return=buy_and_hold_return,
        exposure_pct=exposure_pct,
        bars_per_year=bars_per_year,
    )
