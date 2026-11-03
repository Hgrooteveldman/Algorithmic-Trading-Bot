"""Volatility-adjusted mean-reversion backtester and optimizer.

This module replaces the old, slow-reacting SMA research path with a more
active intraday engine that is built around three ideas:

1. Stretch:
   Mean reversion works best when price is genuinely stretched away from its
   local equilibrium. We model that stretch with Bollinger Bands and Keltner
   Channels and only trade when price pushes into the lower extreme.

2. Confirmation:
   A stretched market can stay stretched. We therefore require an oscillator
   confirmation (RSI) and a modest reversal cue before we buy the dip.

3. Volatility-aware risk:
   Static position sizing and flat percentage stops ignore the thing that
   matters most intraday: the market's current noise level. Position size,
   initial stop, trailing stop, and target are all driven by ATR so the
   strategy naturally adapts to calm versus violent regimes.

The implementation stays realistic:
    - fills happen at the close of the signal bar
    - slippage and commissions are preserved
    - if both stop and target are hit intrabar, the stop wins first
    - a portfolio-level circuit breaker halts new entries after a 25% peak to
      trough drawdown

This code is long by design. The comments explain the mathematical reasoning
behind each moving part so the research path is easy to audit and extend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import floor, isinf
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtest.execution_model import (
    ExecutionModel,
    decision_view,
    require_columns,
    sigma_proxy,
    stop_fill_reference,
)

DECISION_COLUMNS = (
    "atr",
    "bb_mid",
    "rsi",
    "zscore",
    "long_entry_setup",
    "long_exit_signal",
    "short_entry_setup",
    "short_exit_signal",
)
from loguru import logger


def bars_per_year(timeframe_minutes: int) -> float:
    """Approximate number of regular-session bars in a US trading year.

    The NYSE/Nasdaq regular session is 390 minutes long. Using the fractional
    6.5 bars/session for 60-minute data produces a better annualization factor
    than flooring to 6 bars/day.
    """

    return 252.0 * (390.0 / timeframe_minutes)


@dataclass(frozen=True, slots=True)
class MeanReversionConfig:
    """Research parameters for the mean-reversion engine.

    The defaults are intentionally conservative rather than curve-fit. They are
    meant to give the optimizer a stable center point, not to promise a
    particular return profile.
    """

    bollinger_window: int = 20
    bollinger_std_mult: float = 2.0
    keltner_window: int = 20
    keltner_atr_mult: float = 1.5

    rsi_length: int = 7
    rsi_entry_threshold: float = 30.0
    rsi_exit_threshold: float = 58.0

    regime_window: int = 200
    regime_buffer_pct: float = 0.02
    regime_slope_lookback: int = 10
    regime_max_down_slope_pct: float = 0.01
    require_asset_regime: bool = True
    asset_regime_buffer_pct: float = 0.02

    atr_length: int = 14
    atr_stop_mult: float = 1.8
    atr_trailing_stop_mult: float = 2.2
    atr_target_mult: float = 1.1
    mean_exit_atr_buffer: float = 0.15

    max_holding_bars: int = 18
    require_reversal_bar: bool = True
    entry_zscore_threshold: float = 1.0
    entry_reclaim_atr_buffer: float = 0.10

    risk_per_trade_pct: float = 0.003
    max_position_pct: float = 0.18
    max_gross_exposure_pct: float = 0.75
    max_open_positions: int = 4
    circuit_breaker_drawdown_pct: float = 0.25

    allow_short: bool = False

    def label(self) -> str:
        return (
            f"bb{self.bollinger_window}x{self.bollinger_std_mult:.2f}"
            f"_kc{self.keltner_window}x{self.keltner_atr_mult:.2f}"
            f"_rsi{self.rsi_length}-{self.rsi_entry_threshold:.0f}"
            f"_atr{self.atr_length}_stop{self.atr_stop_mult:.2f}"
            f"_trail{self.atr_trailing_stop_mult:.2f}"
            f"_target{self.atr_target_mult:.2f}"
            f"_z{self.entry_zscore_threshold:.1f}"
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


def _wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    """RSI using Wilder's smoothing.

    Wilder's EMA-like smoothing is widely used for trading systems because it
    reacts more smoothly than a simple rolling average of gains/losses while
    still remaining computationally cheap and vectorized.
    """

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Edge cases:
    # - if losses are zero, RSI should be 100
    # - if gains are zero, RSI should be 0
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(avg_gain > 0, 0.0)
    return rsi.fillna(50.0)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    ATR is the backbone of the risk model:
        - larger ATR => smaller size, wider stop, wider trailing band
        - smaller ATR => larger size, tighter stop, tighter trailing band
    """

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


def build_indicator_frame(
    bars: pd.DataFrame,
    proxy_bars: pd.DataFrame,
    config: MeanReversionConfig,
) -> pd.DataFrame:
    """Create a fully vectorized indicator frame for one symbol.

    We keep all expensive math in pandas/numpy here so the event loop only has
    to make discrete trading decisions.
    """

    df = bars.sort_index().copy()
    proxy = proxy_bars.sort_index()[["close"]].rename(columns={"close": "proxy_close"})
    df = df.join(proxy, how="left")
    df["proxy_close"] = df["proxy_close"].ffill().bfill()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    proxy_close = df["proxy_close"].astype(float)

    atr = _wilder_atr(high, low, close, config.atr_length)

    bb_mid = close.rolling(config.bollinger_window).mean()
    bb_std = close.rolling(config.bollinger_window).std(ddof=0)
    bb_upper = bb_mid + config.bollinger_std_mult * bb_std
    bb_lower = bb_mid - config.bollinger_std_mult * bb_std

    # Keltner Channels use an EMA centerline and ATR envelopes. Combining
    # Bollinger and Keltner extremes helps us distinguish a "real" stretch
    # (volatility and price both expanded) from a tiny move in a dead market.
    kc_mid = close.ewm(
        span=config.keltner_window,
        adjust=False,
        min_periods=config.keltner_window,
    ).mean()
    kc_upper = kc_mid + config.keltner_atr_mult * atr
    kc_lower = kc_mid - config.keltner_atr_mult * atr

    asset_regime_sma = close.rolling(config.regime_window).mean()
    proxy_regime_sma = proxy_close.rolling(config.regime_window).mean()
    proxy_regime_slope_pct = proxy_regime_sma.pct_change(config.regime_slope_lookback)

    proxy_regime_ok = (
        (proxy_close >= proxy_regime_sma * (1 - config.regime_buffer_pct))
        & (proxy_regime_slope_pct >= -config.regime_max_down_slope_pct)
    )
    asset_regime_ok = close >= asset_regime_sma * (1 - config.asset_regime_buffer_pct)
    if config.require_asset_regime:
        regime_ok = proxy_regime_ok & asset_regime_ok
    else:
        regime_ok = proxy_regime_ok

    lower_extreme = np.minimum(bb_lower, kc_lower)
    upper_extreme = np.maximum(bb_upper, kc_upper)
    zscore = (close - bb_mid) / bb_std.replace(0.0, np.nan)
    rsi = _wilder_rsi(close, config.rsi_length)

    reversal_bar = (close > close.shift(1)) | (close > df["open"].astype(float))
    rejection_bar = (close < close.shift(1)) | (close < df["open"].astype(float))
    stretched_low = low <= lower_extreme
    stretched_high = high >= upper_extreme
    close_reclaim_level = lower_extreme - (config.entry_reclaim_atr_buffer * atr)
    close_reject_level = upper_extreme + (config.entry_reclaim_atr_buffer * atr)
    close_reclaims = close >= close_reclaim_level
    close_rejects = close <= close_reject_level
    zscore_extreme = zscore <= -abs(config.entry_zscore_threshold)
    short_zscore_extreme = zscore >= abs(config.entry_zscore_threshold)
    if config.require_reversal_bar:
        long_entry_setup = (
            stretched_low
            & close_reclaims
            & zscore_extreme
            & (rsi <= config.rsi_entry_threshold)
            & regime_ok
            & reversal_bar
        )
        short_entry_setup = (
            stretched_high
            & close_rejects
            & short_zscore_extreme
            & (rsi >= (100.0 - config.rsi_entry_threshold))
            & proxy_regime_ok
            & rejection_bar
        )
    else:
        long_entry_setup = (
            stretched_low
            & close_reclaims
            & zscore_extreme
            & (rsi <= config.rsi_entry_threshold)
            & regime_ok
        )
        short_entry_setup = (
            stretched_high
            & close_rejects
            & short_zscore_extreme
            & (rsi >= (100.0 - config.rsi_entry_threshold))
            & proxy_regime_ok
        )

    long_exit_signal = (close >= bb_mid + (config.mean_exit_atr_buffer * atr)) | (
        rsi >= config.rsi_exit_threshold
    )
    short_exit_signal = (close <= bb_mid - (config.mean_exit_atr_buffer * atr)) | (
        rsi <= (100.0 - config.rsi_exit_threshold)
    )

    df["atr"] = atr
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["kc_mid"] = kc_mid
    df["kc_upper"] = kc_upper
    df["kc_lower"] = kc_lower
    df["rsi"] = rsi
    df["asset_regime_sma"] = asset_regime_sma
    df["proxy_regime_sma"] = proxy_regime_sma
    df["proxy_regime_slope_pct"] = proxy_regime_slope_pct
    df["proxy_regime_ok"] = proxy_regime_ok
    df["asset_regime_ok"] = asset_regime_ok
    df["regime_ok"] = regime_ok
    df["lower_extreme"] = lower_extreme
    df["upper_extreme"] = upper_extreme
    df["zscore"] = zscore
    df["close_reclaim_level"] = close_reclaim_level
    df["close_reject_level"] = close_reject_level
    df["long_entry_setup"] = long_entry_setup
    df["long_exit_signal"] = long_exit_signal
    df["short_entry_setup"] = short_entry_setup
    df["short_exit_signal"] = short_exit_signal
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

    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year_value))
    else:
        sharpe = 0.0

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


def backtest_mean_reversion_portfolio(
    bars_by_symbol: dict[str, pd.DataFrame],
    proxy_bars: pd.DataFrame,
    config: MeanReversionConfig,
    starting_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    bars_per_year_value: float = 252.0 * 6.5,
    execution: ExecutionModel | None = None,
) -> BacktestResult:
    """Run the portfolio backtest across all supplied symbols.

    Timing: entry setups, exit signals, z-scores and ATR are read from the
    previous bar (``prev_*`` columns) and filled at the current bar's open.
    Stops and targets fire inside the bar; gaps fill at the open.

    Architecture notes:
        - the engine supports long-only or long+short mode via `allow_short`
          while reusing the same state plumbing for both books.
        - all signal math is precomputed; the event loop only handles fills,
          cash, open positions, and risk limits.
    """

    frames: dict[str, pd.DataFrame] = {}
    events: list[tuple[pd.Timestamp, str, int]] = []
    latest_prices: dict[str, float] = {}

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        frame = build_indicator_frame(bars, proxy_bars=proxy_bars, config=config).reset_index(drop=False)
        if "timestamp" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        require_columns(frame, ["open", "high", "low", "close"])
        frame = decision_view(frame, DECISION_COLUMNS)
        frames[symbol] = frame
        latest_prices[symbol] = float(frame["close"].iloc[0])
        events.extend(
            (frame.iloc[idx]["timestamp"], symbol, idx)
            for idx in range(1, len(frame))
        )

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

    model = execution or ExecutionModel.from_legacy(commission_bps, slippage_bps)
    est_cost = (commission_bps + slippage_bps) / 10_000.0

    def bar_context(row: pd.Series, reference: float) -> dict:
        volume = row.get("volume")
        atr_value = row.get("atr")
        return {
            "high": float(row["high"]),
            "low": float(row["low"]),
            "bar_volume": float(volume) if volume is not None and pd.notna(volume) else None,
            "sigma_bar": sigma_proxy(float(atr_value) if atr_value is not None and pd.notna(atr_value) else None, reference),
        }

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
            latest_prices[symbol] = float(frames[symbol].iloc[idx]["open"])

        equity_before_entries = portfolio_equity()
        peak_equity = max(peak_equity, equity_before_entries)
        if peak_equity > 0 and equity_before_entries <= peak_equity * (1 - config.circuit_breaker_drawdown_pct):
            entries_halted = True
            circuit_breaker_triggered = True

        # 1) Manage existing positions with already-known thresholds.
        for symbol, idx in symbol_events:
            position = positions.get(symbol)
            if position is None:
                continue

            row = frames[symbol].iloc[idx]
            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            position.holding_bars += 1
            position.highest_high = max(position.highest_high, high)
            position.lowest_low = min(position.lowest_low, low)

            exit_trigger = None
            reported_exit_price = None
            reason = ""

            # Preserve the pessimistic assumption: if the stop and target are
            # both touched intrabar, the stop is assumed to fire first.
            # Decisions taken on the previous close (signal or time exit)
            # execute at this open, before anything inside the bar.
            if position.side == 1:
                if bool(row["prev_long_exit_signal"]):
                    exit_trigger, reason = open_price, "SIGNAL"
                elif position.holding_bars > config.max_holding_bars:
                    exit_trigger, reason = open_price, "TIME"
                elif low <= position.stop_price:
                    exit_trigger = stop_fill_reference(1, open_price, position.stop_price, "stop")
                    reason = "STOP"
                elif high >= position.target_price:
                    exit_trigger = stop_fill_reference(1, open_price, position.target_price, "target")
                    reason = "TARGET"
            else:
                if bool(row["prev_short_exit_signal"]):
                    exit_trigger, reason = open_price, "SIGNAL"
                elif position.holding_bars > config.max_holding_bars:
                    exit_trigger, reason = open_price, "TIME"
                elif high >= position.stop_price:
                    exit_trigger = stop_fill_reference(-1, open_price, position.stop_price, "stop")
                    reason = "STOP"
                elif low <= position.target_price:
                    exit_trigger = stop_fill_reference(-1, open_price, position.target_price, "target")
                    reason = "TARGET"

            if exit_trigger is None:
                current_atr = float(row["atr"])
                if np.isfinite(current_atr) and current_atr > 0:
                    if position.side == 1:
                        trailing_stop = position.highest_high - (
                            config.atr_trailing_stop_mult * current_atr
                        )
                        initial_stop = position.entry_price - (
                            config.atr_stop_mult * position.entry_atr
                        )
                        position.stop_price = max(position.stop_price, initial_stop, trailing_stop)

                        mean_target = float(row["bb_mid"]) + (
                            config.mean_exit_atr_buffer * current_atr
                        )
                        atr_target = position.entry_price + (
                            config.atr_target_mult * position.entry_atr
                        )
                        position.target_price = max(position.target_price, mean_target, atr_target)
                    else:
                        trailing_stop = position.lowest_low + (
                            config.atr_trailing_stop_mult * current_atr
                        )
                        initial_stop = position.entry_price + (
                            config.atr_stop_mult * position.entry_atr
                        )
                        position.stop_price = min(position.stop_price, initial_stop, trailing_stop)

                        mean_target = float(row["bb_mid"]) - (
                            config.mean_exit_atr_buffer * current_atr
                        )
                        atr_target = position.entry_price - (
                            config.atr_target_mult * position.entry_atr
                        )
                        position.target_price = min(position.target_price, mean_target, atr_target)
                continue

            exit_fill = model.fill(-position.side, exit_trigger, position.qty, **bar_context(row, exit_trigger))
            reported_exit_price = exit_fill.price
            if position.side == 1:
                proceeds = position.qty * exit_fill.price
                pnl = proceeds - (position.qty * position.entry_price)
                cash += proceeds
                pnl_pct = (exit_fill.price / position.entry_price) - 1
                logged_side = "LONG"
            else:
                cover_cost = position.qty * exit_fill.price
                pnl = (position.qty * position.entry_price) - cover_cost
                cash -= cover_cost
                pnl_pct = (position.entry_price - exit_fill.price) / position.entry_price
                logged_side = "SHORT"
            trades.append(
                {
                    "symbol": symbol,
                    "side": logged_side,
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

        # 2) Rank fresh entry candidates by stretch severity.
        entry_candidates: list[tuple[float, str, int, int]] = []
        if not entries_halted:
            for symbol, idx in symbol_events:
                if symbol in positions:
                    continue
                row = frames[symbol].iloc[idx]
                required_values = [
                    row.get("prev_atr"),
                    row.get("prev_bb_mid"),
                    row.get("prev_rsi"),
                    row.get("prev_zscore"),
                ]
                if any(pd.isna(value) for value in required_values):
                    continue
                zscore_value = float(row["prev_zscore"])
                if bool(row["prev_long_entry_setup"]):
                    entry_candidates.append((abs(zscore_value), symbol, idx, 1))
                if config.allow_short and bool(row["prev_short_entry_setup"]):
                    entry_candidates.append((abs(zscore_value), symbol, idx, -1))

        # Larger absolute z-score = more compelling mean-reversion setup.
        entry_candidates.sort(key=lambda item: (-item[0], item[1], -item[3]))

        for _, symbol, idx, side in entry_candidates:
            if symbol in positions or len(positions) >= config.max_open_positions:
                continue

            row = frames[symbol].iloc[idx]
            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row["prev_atr"])
            if not np.isfinite(atr) or atr <= 0 or open_price <= 0:
                continue

            equity = portfolio_equity()
            current_gross_exposure = gross_exposure()
            available_notional = max((equity * config.max_gross_exposure_pct) - current_gross_exposure, 0.0)
            est_price = open_price * (1 + est_cost) if side == 1 else open_price * (1 - est_cost)

            # ATR sizing:
            #   risk_budget = equity * risk_per_trade_pct
            #   per_share_risk ~= ATR_stop_multiple * ATR + round-trip friction
            #
            # This keeps the expected loss of each trade roughly stable even
            # when one symbol is placid and another is chaotic.
            risk_budget = equity * config.risk_per_trade_pct
            per_share_risk = (config.atr_stop_mult * atr) + (est_price * (2 * est_cost))
            qty_by_risk = floor(risk_budget / per_share_risk) if per_share_risk > 0 else 0
            qty_by_position_cap = floor((equity * config.max_position_pct) / est_price)
            if side == 1:
                qty_by_available_cash = floor(min(cash, available_notional) / est_price)
            else:
                qty_by_available_cash = floor(available_notional / max(open_price, 1e-9))
            qty = max(min(qty_by_risk, qty_by_position_cap, qty_by_available_cash), 0)
            if qty <= 0:
                continue
            entry_price = model.fill(side, open_price, qty, **bar_context(row, open_price)).price

            if side == 1:
                cash -= qty * entry_price
                initial_stop = entry_price - (config.atr_stop_mult * atr)
                atr_target = entry_price + (config.atr_target_mult * atr)
                mean_target = float(row["prev_bb_mid"]) + (config.mean_exit_atr_buffer * atr)
                target_price = max(atr_target, mean_target)
            else:
                cash += qty * entry_price
                initial_stop = entry_price + (config.atr_stop_mult * atr)
                atr_target = entry_price - (config.atr_target_mult * atr)
                mean_target = float(row["prev_bb_mid"]) - (config.mean_exit_atr_buffer * atr)
                target_price = min(atr_target, mean_target)
            positions[symbol] = OpenPosition(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=entry_price,
                entry_time=timestamp,
                entry_index=idx,
                entry_atr=atr,
                stop_price=initial_stop,
                target_price=target_price,
                highest_high=high,
                lowest_low=low,
            )

        for symbol, idx in symbol_events:
            latest_prices[symbol] = float(frames[symbol].iloc[idx]["close"])
        equity = portfolio_equity()
        equity_curve.append(equity)
        equity_index.append(timestamp)
        exposure_samples.append(gross_exposure() / equity if equity > 0 else 0.0)

    # Force close any remaining positions on the last available close.
    for symbol, position in list(positions.items()):
        frame = frames[symbol]
        final_bar = frame.iloc[-1]
        final_time = final_bar["timestamp"]
        final_close = float(final_bar["close"])
        final_effective_exit = model.fill(
            -position.side, final_close, position.qty, **bar_context(final_bar, final_close)
        ).price
        if position.side == 1:
            pnl = (position.qty * final_effective_exit) - (position.qty * position.entry_price)
            cash += position.qty * final_effective_exit
            pnl_pct = (final_effective_exit / position.entry_price) - 1
            logged_side = "LONG"
        else:
            pnl = (position.qty * position.entry_price) - (position.qty * final_effective_exit)
            cash -= position.qty * final_effective_exit
            pnl_pct = (position.entry_price - final_effective_exit) / position.entry_price
            logged_side = "SHORT"
        trades.append(
            {
                "symbol": symbol,
                "side": logged_side,
                "entry_time": position.entry_time,
                "exit_time": final_time,
                "entry": position.entry_price,
                "exit": final_effective_exit,
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


def backtest_mean_reversion_single(
    bars: pd.DataFrame,
    proxy_bars: pd.DataFrame | None = None,
    config: MeanReversionConfig | None = None,
    **kwargs,
) -> BacktestResult:
    """Convenience wrapper for a single symbol."""

    config = config or MeanReversionConfig()
    proxy_bars = proxy_bars if proxy_bars is not None else bars
    return backtest_mean_reversion_portfolio(
        {"ASSET": bars},
        proxy_bars=proxy_bars,
        config=config,
        **kwargs,
    )


def default_optimization_grid(
    timeframe_minutes: int,
    profile: str = "balanced",
) -> dict[str, list]:
    """Reasonable search space for 15m/60m intraday mean reversion."""

    profile = profile.strip().lower()
    if profile not in {"balanced", "aggressive", "active"}:
        raise ValueError("profile must be 'balanced', 'aggressive', or 'active'")

    if profile == "active":
        if timeframe_minutes <= 15:
            return {
                "bollinger_window": [10, 14, 18],
                "bollinger_std_mult": [1.0, 1.2, 1.4],
                "keltner_atr_mult": [0.8, 1.0, 1.2],
                "rsi_length": [3, 5, 7],
                "rsi_entry_threshold": [34.0, 38.0, 42.0],
                "rsi_exit_threshold": [50.0, 54.0, 58.0],
                "require_asset_regime": [False, True],
                "asset_regime_buffer_pct": [0.05, 0.08],
                "atr_stop_mult": [0.9, 1.2, 1.5],
                "atr_trailing_stop_mult": [1.2, 1.5, 1.8],
                "atr_target_mult": [0.6, 0.9, 1.2],
                "mean_exit_atr_buffer": [0.0, 0.05, 0.10],
                "max_holding_bars": [4, 8, 12],
                "require_reversal_bar": [False, True],
                "entry_zscore_threshold": [0.4, 0.6, 0.8],
                "entry_reclaim_atr_buffer": [0.0, 0.05, 0.10],
                "risk_per_trade_pct": [0.0025, 0.0040, 0.0060],
                "max_position_pct": [0.10, 0.15, 0.20],
                "max_gross_exposure_pct": [0.40, 0.60, 0.80],
                "max_open_positions": [4, 5],
            }
        return {
            "bollinger_window": [10, 14, 18],
            "bollinger_std_mult": [1.0, 1.2, 1.4],
            "keltner_atr_mult": [0.8, 1.0, 1.2],
            "rsi_length": [3, 5, 7],
            "rsi_entry_threshold": [34.0, 38.0, 42.0],
            "rsi_exit_threshold": [50.0, 54.0, 58.0],
            "require_asset_regime": [False, True],
            "asset_regime_buffer_pct": [0.05, 0.08],
            "atr_stop_mult": [1.0, 1.3, 1.6],
            "atr_trailing_stop_mult": [1.2, 1.6, 2.0],
            "atr_target_mult": [0.6, 0.9, 1.2],
            "mean_exit_atr_buffer": [0.0, 0.05, 0.10],
            "max_holding_bars": [4, 8, 12],
            "require_reversal_bar": [False, True],
            "entry_zscore_threshold": [0.4, 0.6, 0.8],
            "entry_reclaim_atr_buffer": [0.0, 0.05, 0.10],
            "risk_per_trade_pct": [0.0025, 0.0040, 0.0060],
            "max_position_pct": [0.10, 0.15, 0.20],
            "max_gross_exposure_pct": [0.40, 0.60, 0.80],
            "max_open_positions": [4, 5],
        }

    if profile == "aggressive":
        if timeframe_minutes <= 15:
            return {
                "bollinger_window": [14, 18, 20],
                "bollinger_std_mult": [1.2, 1.4, 1.6],
                "keltner_atr_mult": [1.0, 1.3, 1.5],
                "rsi_length": [3, 5, 7],
                "rsi_entry_threshold": [28.0, 32.0, 36.0],
                "rsi_exit_threshold": [58.0, 64.0, 70.0],
                "require_asset_regime": [False, True],
                "asset_regime_buffer_pct": [0.02, 0.05, 0.08],
                "atr_stop_mult": [1.0, 1.3, 1.6],
                "atr_trailing_stop_mult": [1.4, 1.8, 2.2],
                "atr_target_mult": [1.1, 1.5, 2.0],
                "mean_exit_atr_buffer": [0.0, 0.10, 0.20],
                "max_holding_bars": [12, 18, 24],
                "require_reversal_bar": [False, True],
                "entry_zscore_threshold": [0.6, 0.9, 1.2],
                "entry_reclaim_atr_buffer": [0.0, 0.10, 0.20],
                "risk_per_trade_pct": [0.0025, 0.0040, 0.0060],
                "max_position_pct": [0.10, 0.15, 0.20],
                "max_gross_exposure_pct": [0.40, 0.60, 0.80],
                "max_open_positions": [4, 5],
            }
        return {
            "bollinger_window": [14, 18, 20],
            "bollinger_std_mult": [1.2, 1.4, 1.6],
            "keltner_atr_mult": [1.0, 1.3, 1.5],
            "rsi_length": [3, 5, 7],
            "rsi_entry_threshold": [28.0, 32.0, 36.0],
            "rsi_exit_threshold": [58.0, 64.0, 70.0],
            "require_asset_regime": [False, True],
            "asset_regime_buffer_pct": [0.02, 0.05, 0.08],
            "atr_stop_mult": [1.0, 1.4, 1.8],
            "atr_trailing_stop_mult": [1.4, 1.8, 2.2],
            "atr_target_mult": [1.2, 1.6, 2.0],
            "mean_exit_atr_buffer": [0.0, 0.10, 0.20],
            "max_holding_bars": [12, 18, 24],
            "require_reversal_bar": [False, True],
            "entry_zscore_threshold": [0.6, 0.9, 1.2],
            "entry_reclaim_atr_buffer": [0.0, 0.10, 0.20],
            "risk_per_trade_pct": [0.0025, 0.0040, 0.0060],
            "max_position_pct": [0.10, 0.15, 0.20],
            "max_gross_exposure_pct": [0.40, 0.60, 0.80],
            "max_open_positions": [4, 5],
        }

    if timeframe_minutes <= 15:
        return {
            "bollinger_window": [18, 20, 24],
            "bollinger_std_mult": [1.8, 2.0, 2.2],
            "keltner_atr_mult": [1.2, 1.5, 1.8],
            "rsi_length": [5, 7, 9],
            "rsi_entry_threshold": [24.0, 28.0, 32.0],
            "rsi_exit_threshold": [52.0, 58.0, 64.0],
            "require_asset_regime": [False, True],
            "asset_regime_buffer_pct": [0.02, 0.05],
            "atr_stop_mult": [1.4, 1.8, 2.2],
            "atr_trailing_stop_mult": [1.8, 2.2, 2.6],
            "atr_target_mult": [0.8, 1.1, 1.4],
            "mean_exit_atr_buffer": [0.0, 0.15],
            "max_holding_bars": [8, 12, 16, 24],
            "require_reversal_bar": [False, True],
            "entry_zscore_threshold": [0.8, 1.1],
            "entry_reclaim_atr_buffer": [0.0, 0.10, 0.20],
        }

    return {
        "bollinger_window": [18, 20, 24],
        "bollinger_std_mult": [1.8, 2.0, 2.2],
        "keltner_atr_mult": [1.2, 1.5, 1.8],
        "rsi_length": [5, 7, 9],
        "rsi_entry_threshold": [24.0, 28.0, 32.0],
        "rsi_exit_threshold": [52.0, 58.0, 64.0],
        "require_asset_regime": [False, True],
        "asset_regime_buffer_pct": [0.02, 0.05],
        "atr_stop_mult": [1.6, 1.9, 2.3],
        "atr_trailing_stop_mult": [2.0, 2.4, 2.8],
        "atr_target_mult": [0.9, 1.2, 1.5],
        "mean_exit_atr_buffer": [0.0, 0.15],
        "max_holding_bars": [8, 12, 18, 24],
        "require_reversal_bar": [False, True],
        "entry_zscore_threshold": [0.8, 1.1],
        "entry_reclaim_atr_buffer": [0.0, 0.10, 0.20],
    }


def generate_candidate_configs(
    base_config: MeanReversionConfig,
    parameter_grid: dict[str, Iterable] | None = None,
) -> list[MeanReversionConfig]:
    grid = parameter_grid or default_optimization_grid(timeframe_minutes=60)
    keys = list(grid.keys())
    if not keys:
        return [base_config]

    configs: list[MeanReversionConfig] = []
    for values in product(*(grid[key] for key in keys)):
        config = base_config
        for key, value in zip(keys, values):
            config = replace(config, **{key: value})
        configs.append(config)
    return configs


def limit_candidate_configs(
    candidate_configs: list[MeanReversionConfig],
    max_configs: int | None,
) -> list[MeanReversionConfig]:
    """Deterministically sample a large grid down to a manageable subset.

    Taking the first N rows would bias the search toward the earliest parameter
    dimensions in the Cartesian product. Instead we sample evenly across the
    full grid so quick test runs still touch the broad search space.
    """

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
        return 500
    if timeframe_minutes <= 15:
        return 220
    if timeframe_minutes <= 30:
        return 140
    return 90


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
        (1 if meets_target else 0),
        result.annualized_return - penalty,
        result.sharpe,
        result.profit_factor,
        -abs(result.max_drawdown),
    )


def optimize_mean_reversion(
    bars_by_symbol: dict[str, pd.DataFrame],
    proxy_bars: pd.DataFrame,
    base_config: MeanReversionConfig,
    parameter_grid: dict[str, Iterable] | None = None,
    target_annual_return: float = 0.08,
    max_drawdown_limit: float = 0.25,
    starting_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    timeframe_minutes: int = 60,
    bars_per_year_value: float = 252.0 * 6.5,
    max_configs: int | None = None,
    progress_every: int = 100,
    evaluation_mode: str = "balanced",
    min_trade_count: int | None = None,
) -> tuple[pd.DataFrame, MeanReversionConfig, BacktestResult]:
    """Grid-search the mean-reversion parameters.

    `itertools.product` is deliberately used over scipy.optimize because the
    objective surface of a trading system is noisy, discontinuous, and full of
    local traps. A small deterministic grid is slower but much easier to audit.
    """

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
            "Optimizer grid contains {:,} combinations. Auto-limiting this run "
            "to {:,} evenly sampled configurations for practicality.",
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
    best_config: MeanReversionConfig | None = None
    best_score: tuple | None = None

    total_candidates = len(candidate_configs)
    progress_every = max(int(progress_every), 1)
    for index, config in enumerate(candidate_configs, start=1):
        result = backtest_mean_reversion_portfolio(
            bars_by_symbol=bars_by_symbol,
            proxy_bars=proxy_bars,
            config=config,
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

        if (
            index == 1
            or index == total_candidates
            or index % progress_every == 0
        ):
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
