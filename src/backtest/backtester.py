"""Event-driven backtester for the default trend strategy.

Limits to be aware of (read these before trusting any result):
    - Signals are evaluated on a bar's close and filled at the NEXT bar's
      open (see src/backtest/execution_model.py). No fill ever uses the
      close that produced its own signal.
    - Friction per fill = fee tier + half spread (range-based proxy, with a
      floor) + linear market impact on participation of bar volume.
    - Stop and target are checked against the bar's high/low, ignoring
      intra-bar path. If both are hit in the same bar, we pessimistically
      assume the stop fired first. Gaps through a level fill at the open.
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

from src.backtest.execution_model import (
    ExecutionModel,
    FillResult,
    require_columns,
    sigma_proxy,
    stop_fill_reference,
)

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


def _bar_context(bar: pd.Series) -> dict:
    """Fields the execution model needs from one bar."""
    open_price = float(bar["open"])
    atr = bar.get("atr")
    atr_value = float(atr) if atr is not None and pd.notna(atr) else None
    volume = bar.get("volume")
    return {
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "bar_volume": float(volume) if volume is not None and pd.notna(volume) else None,
        "sigma_bar": sigma_proxy(atr_value, open_price),
    }


def _trade_record(
    *,
    entry_time,
    exit_time,
    entry_fill: FillResult,
    exit_fill: FillResult,
    qty: int,
    hold_bars: int,
    reason: str,
    symbol: str | None = None,
) -> dict:
    record = {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry": entry_fill.price,
        "exit": exit_fill.price,
        "qty": qty,
        "pnl": qty * (exit_fill.price - entry_fill.price),
        "pnl_pct": (exit_fill.price / entry_fill.price) - 1,
        "hold_bars": hold_bars,
        "reason": reason,
        "entry_cost_bps": entry_fill.total_cost_bps,
        "exit_cost_bps": exit_fill.total_cost_bps,
    }
    if symbol is not None:
        record = {"symbol": symbol, **record}
    return record


def _position_fraction(risk_per_trade_pct: float, stop_loss_pct: float, est_cost: float, max_position_pct: float) -> float:
    effective_stop_pct = stop_loss_pct + (2 * est_cost)
    return min(risk_per_trade_pct / max(effective_stop_pct, 1e-9), max_position_pct)


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
    execution: ExecutionModel | None = None,
) -> BacktestResult:
    """Single-symbol backtest.

    Timing: the signal is evaluated on bar ``i``'s close and executed at bar
    ``i+1``'s open. Stops and targets are checked from the fill bar onward,
    with gaps through a level filled at the open.
    """
    require_columns(bars, ["open", "high", "low", "close"])
    df = build_signal_frame(
        bars=bars,
        short=short_window,
        long=long_window,
        trend_filter=trend_filter_window,
        pullback_lookback=pullback_lookback,
        trend_slope_lookback=trend_slope_lookback,
        atr_window=atr_window,
    ).reset_index(drop=False)

    model = execution or ExecutionModel.from_legacy(commission_bps, slippage_bps)
    est_cost = (commission_bps + slippage_bps) / 10_000
    target_position_pct = _position_fraction(risk_per_trade_pct, stop_loss_pct, est_cost, max_position_pct)

    cash = starting_cash
    qty = 0
    entry_fill: FillResult | None = None
    stop = 0.0
    target = 0.0
    entry_time = None
    entry_index = 0
    pending: str | None = None  # "BUY" or "SELL", decided on the previous bar

    equity_curve: List[float] = []
    equity_index: List = []
    trades: List[dict] = []
    bars_in_market = 0

    for i in range(1, len(df)):
        bar = df.iloc[i]
        timestamp = bar["timestamp"] if "timestamp" in bar else bar.name
        open_price = float(bar["open"])
        price = float(bar["close"])
        high, low = float(bar["high"]), float(bar["low"])
        context = _bar_context(bar)

        # --- 1) Execute yesterday's decision at today's open ---
        if pending == "BUY" and qty == 0:
            qty = int((cash * target_position_pct) // (open_price * (1 + est_cost)))
            if qty > 0:
                entry_fill = model.fill(1, open_price, qty, **context)
                cash -= qty * entry_fill.price
                stop = round(entry_fill.price * (1 - stop_loss_pct), 2)
                target = round(entry_fill.price * (1 + take_profit_pct), 2)
                entry_time = timestamp
                entry_index = i
        elif pending == "SELL" and qty > 0 and entry_fill is not None:
            exit_fill = model.fill(-1, open_price, qty, **context)
            cash += qty * exit_fill.price
            trades.append(_trade_record(entry_time=entry_time, exit_time=timestamp, entry_fill=entry_fill,
                                        exit_fill=exit_fill, qty=qty, hold_bars=i - entry_index, reason="SIGNAL"))
            qty = 0
        pending = None

        # --- 2) Manage the open position inside the bar ---
        if qty > 0 and entry_fill is not None:
            bars_in_market += 1
            reference = None
            reason = ""
            # Pessimistic assumption: if both touched, stop fires first.
            if low <= stop:
                reference, reason = stop_fill_reference(1, open_price, stop, "stop"), "STOP"
            elif high >= target:
                reference, reason = stop_fill_reference(1, open_price, target, "target"), "TARGET"
            if reference is not None:
                exit_fill = model.fill(-1, reference, qty, **context)
                cash += qty * exit_fill.price
                trades.append(_trade_record(entry_time=entry_time, exit_time=timestamp, entry_fill=entry_fill,
                                            exit_fill=exit_fill, qty=qty, hold_bars=i - entry_index, reason=reason))
                qty = 0

        # --- 3) Decide on this bar's close; executed at the next open ---
        signal = signal_for_index(
            df,
            idx=i,
            pullback_reclaim_tolerance_pct=pullback_reclaim_tolerance_pct,
            min_atr_pct=min_atr_pct,
            max_atr_pct=max_atr_pct,
        )
        if signal == Signal.BUY and qty == 0:
            strength_score = relative_strength_score(df, idx=i, lookback_bars=relative_strength_lookback)
            strength_snapshot = (
                RelativeStrengthSnapshot(score=strength_score, rank=1) if strength_score is not None else None
            )
            if passes_relative_strength_filter(
                strength_snapshot,
                max_rank=max_relative_strength_rank,
                min_return=min_relative_strength_return,
            ):
                pending = "BUY"
        elif signal == Signal.SELL and qty > 0:
            pending = "SELL"

        equity_curve.append(cash + qty * price)
        equity_index.append(timestamp)

    if qty > 0 and len(df) and entry_fill is not None:
        final_bar = df.iloc[-1]
        final_time = final_bar["timestamp"] if "timestamp" in final_bar else final_bar.name
        exit_fill = model.fill(-1, float(final_bar["close"]), qty, **_bar_context(final_bar))
        cash += qty * exit_fill.price
        trades.append(_trade_record(entry_time=entry_time, exit_time=final_time, entry_fill=entry_fill,
                                    exit_fill=exit_fill, qty=qty, hold_bars=max(len(df) - 1 - entry_index, 0),
                                    reason="END"))
        qty = 0
        if equity_curve:
            equity_curve[-1] = cash

    equity = pd.Series(equity_curve, index=equity_index)
    buy_and_hold_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1 if len(df) > 1 else 0.0
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
    execution: ExecutionModel | None = None,
) -> BacktestResult:
    """Multi-symbol backtest with the same next-open timing as
    :func:`backtest_sma`: decisions on bar ``t``'s close, fills at ``t+1``'s
    open, intrabar stops/targets with gap handling."""
    frames: dict[str, pd.DataFrame] = {}
    events: list[tuple[pd.Timestamp, str, int]] = []
    latest_prices: dict[str, float] = {}

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        require_columns(bars, ["open", "high", "low", "close"])
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
        events.extend((frame.iloc[idx]["timestamp"], symbol, idx) for idx in range(1, len(frame)))

    if not events:
        return _result_from_components(
            equity=pd.Series(dtype=float),
            trades_df=pd.DataFrame(),
            starting_cash=starting_cash,
            buy_and_hold_return=0.0,
            exposure_pct=0.0,
            bars_per_year=bars_per_year,
        )

    events.sort(key=lambda item: (item[0], item[1]))
    model = execution or ExecutionModel.from_legacy(commission_bps, slippage_bps)
    est_cost = (commission_bps + slippage_bps) / 10_000
    target_position_pct = _position_fraction(risk_per_trade_pct, stop_loss_pct, est_cost, max_position_pct)

    cash = starting_cash
    positions: dict[str, dict] = {}
    pending_sells: set[str] = set()
    pending_buys: list[str] = []  # ranked, decided on the previous bar
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_index: list[pd.Timestamp] = []
    exposure_samples: list[float] = []

    def portfolio_equity() -> float:
        return cash + sum(
            position["qty"] * latest_prices.get(symbol, position["entry_fill"].price)
            for symbol, position in positions.items()
        )

    def portfolio_gross_exposure() -> float:
        return sum(
            position["qty"] * latest_prices.get(symbol, position["entry_fill"].price)
            for symbol, position in positions.items()
        )

    event_groups: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for timestamp, symbol, idx in events:
        event_groups.setdefault(timestamp, []).append((symbol, idx))

    for timestamp in sorted(event_groups):
        symbol_events = event_groups[timestamp]
        idx_by_symbol = {symbol: idx for symbol, idx in symbol_events}

        # --- 1) Execute the previous bar's decisions at this bar's open ---
        for symbol in sorted(pending_sells):
            idx = idx_by_symbol.get(symbol)
            position = positions.get(symbol)
            if idx is None or position is None:
                continue
            bar = frames[symbol].iloc[idx]
            exit_fill = model.fill(-1, float(bar["open"]), position["qty"], **_bar_context(bar))
            cash += position["qty"] * exit_fill.price
            trades.append(_trade_record(symbol=symbol, entry_time=position["entry_time"], exit_time=timestamp,
                                        entry_fill=position["entry_fill"], exit_fill=exit_fill, qty=position["qty"],
                                        hold_bars=idx - position["entry_index"], reason="SIGNAL"))
            positions.pop(symbol)
        pending_sells = {s for s in pending_sells if s not in idx_by_symbol and s in positions}

        for symbol, idx in symbol_events:
            latest_prices[symbol] = float(frames[symbol].iloc[idx]["open"])

        carried_buys: list[str] = []
        for symbol in pending_buys:
            idx = idx_by_symbol.get(symbol)
            if idx is None:
                carried_buys.append(symbol)
                continue
            if symbol in positions or len(positions) >= max_open_positions:
                continue
            bar = frames[symbol].iloc[idx]
            open_price = float(bar["open"])
            equity = portfolio_equity()
            available_notional = max((equity * max_gross_exposure_pct) - portfolio_gross_exposure(), 0.0)
            desired_notional = min(equity * target_position_pct, available_notional, cash)
            qty = int(desired_notional // (open_price * (1 + est_cost)))
            if qty <= 0:
                continue
            entry_fill = model.fill(1, open_price, qty, **_bar_context(bar))
            cash -= qty * entry_fill.price
            positions[symbol] = {
                "qty": qty,
                "entry_fill": entry_fill,
                "stop": round(entry_fill.price * (1 - stop_loss_pct), 2),
                "target": round(entry_fill.price * (1 + take_profit_pct), 2),
                "entry_time": timestamp,
                "entry_index": idx,
            }
        pending_buys = carried_buys

        for symbol, idx in symbol_events:
            latest_prices[symbol] = float(frames[symbol].iloc[idx]["close"])

        # --- 2) Intrabar stops and targets ---
        for symbol, idx in symbol_events:
            position = positions.get(symbol)
            if position is None:
                continue
            bar = frames[symbol].iloc[idx]
            open_price, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
            reference = None
            reason = ""
            if low <= position["stop"]:
                reference, reason = stop_fill_reference(1, open_price, position["stop"], "stop"), "STOP"
            elif high >= position["target"]:
                reference, reason = stop_fill_reference(1, open_price, position["target"], "target"), "TARGET"
            if reference is None:
                continue
            exit_fill = model.fill(-1, reference, position["qty"], **_bar_context(bar))
            cash += position["qty"] * exit_fill.price
            trades.append(_trade_record(symbol=symbol, entry_time=position["entry_time"], exit_time=timestamp,
                                        entry_fill=position["entry_fill"], exit_fill=exit_fill, qty=position["qty"],
                                        hold_bars=idx - position["entry_index"], reason=reason))
            positions.pop(symbol, None)
            pending_sells.discard(symbol)

        # --- 3) Decide on this bar's close (executed at the next open) ---
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
            idx_by_symbol=idx_by_symbol,
            lookback_bars=relative_strength_lookback,
        )
        for symbol in idx_by_symbol:
            if signal_by_symbol[symbol] == Signal.SELL and symbol in positions:
                pending_sells.add(symbol)

        buy_candidates: list[tuple[float, str]] = []
        for symbol in idx_by_symbol:
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
            buy_candidates.append((score, symbol))
        buy_candidates.sort(key=lambda item: (-item[0], item[1]))
        pending_buys = [symbol for _, symbol in buy_candidates]

        equity = portfolio_equity()
        gross_exposure = portfolio_gross_exposure()
        equity_curve.append(equity)
        equity_index.append(timestamp)
        exposure_samples.append(gross_exposure / equity if equity > 0 else 0.0)

    for symbol, position in list(positions.items()):
        frame = frames[symbol]
        final_bar = frame.iloc[-1]
        exit_fill = model.fill(-1, float(final_bar["close"]), position["qty"], **_bar_context(final_bar))
        cash += position["qty"] * exit_fill.price
        trades.append(_trade_record(symbol=symbol, entry_time=position["entry_time"], exit_time=final_bar["timestamp"],
                                    entry_fill=position["entry_fill"], exit_fill=exit_fill, qty=position["qty"],
                                    hold_bars=max(len(frame) - 1 - position["entry_index"], 0), reason="END"))

    if equity_curve:
        equity_curve[-1] = cash

    buy_and_hold_return = float(
        np.mean([frame["close"].iloc[-1] / frame["close"].iloc[0] - 1 for frame in frames.values() if len(frame) > 1])
    )
    exposure_pct = float(np.mean(exposure_samples)) if exposure_samples else 0.0

    return _result_from_components(
        equity=pd.Series(equity_curve, index=equity_index),
        trades_df=pd.DataFrame(trades),
        starting_cash=starting_cash,
        buy_and_hold_return=buy_and_hold_return,
        exposure_pct=exposure_pct,
        bars_per_year=bars_per_year,
    )
