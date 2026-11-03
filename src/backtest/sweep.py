"""Helpers for ranking and generating parameter-sweep configurations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import floor

import pandas as pd

from src.backtest.backtester import BacktestResult


@dataclass(frozen=True, slots=True)
class SweepConfig:
    timeframe_minutes: int
    short_window: int
    long_window: int
    trend_filter_window: int
    pullback_lookback: int
    trend_slope_lookback: int
    atr_window: int
    min_atr_pct: float
    max_atr_pct: float
    pullback_reclaim_tolerance_pct: float

    def label(self) -> str:
        min_atr = f"{self.min_atr_pct:.4f}".replace(".", "p")
        max_atr = f"{self.max_atr_pct:.4f}".replace(".", "p")
        reclaim = f"{self.pullback_reclaim_tolerance_pct:.4f}".replace(".", "p")
        return (
            f"{self.timeframe_minutes}m_s{self.short_window}_l{self.long_window}"
            f"_t{self.trend_filter_window}_pb{self.pullback_lookback}"
            f"_sl{self.trend_slope_lookback}_atr{self.atr_window}"
            f"_band{min_atr}-{max_atr}_rt{reclaim}"
        )


def sample_sweep_configs(
    configs: list[SweepConfig],
    max_configs: int,
) -> list[SweepConfig]:
    """Sample evenly across the config grid instead of taking only the prefix."""
    if max_configs <= 0 or len(configs) <= max_configs:
        return configs

    if max_configs == 1:
        return [configs[len(configs) // 2]]

    last_index = len(configs) - 1
    sampled_indices = []
    for sample_idx in range(max_configs):
        raw_index = floor(sample_idx * last_index / (max_configs - 1))
        if raw_index not in sampled_indices:
            sampled_indices.append(raw_index)

    while len(sampled_indices) < max_configs:
        sampled_indices.append(last_index)

    return [configs[index] for index in sampled_indices]


def bars_per_year_for_timeframe(timeframe_minutes: int) -> int:
    bars_per_session = max(1, 390 // timeframe_minutes)
    return 252 * bars_per_session


def generate_sweep_configs(
    timeframes: list[int],
    short_long_pairs: list[tuple[int, int]],
    trend_filters: list[int],
    pullback_lookbacks: list[int],
    trend_slope_lookbacks: list[int],
    atr_windows: list[int],
    atr_bands: list[tuple[float, float]],
    reclaim_tolerances: list[float],
) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for (
        timeframe,
        (short_window, long_window),
        trend_filter_window,
        pullback_lookback,
        trend_slope_lookback,
        atr_window,
        (min_atr_pct, max_atr_pct),
        reclaim_tolerance,
    ) in product(
        timeframes,
        short_long_pairs,
        trend_filters,
        pullback_lookbacks,
        trend_slope_lookbacks,
        atr_windows,
        atr_bands,
        reclaim_tolerances,
    ):
        if timeframe not in {1, 5, 15, 30, 60}:
            continue
        if short_window >= long_window:
            continue
        if trend_filter_window <= long_window:
            continue
        if min_atr_pct >= max_atr_pct:
            continue
        configs.append(
            SweepConfig(
                timeframe_minutes=timeframe,
                short_window=short_window,
                long_window=long_window,
                trend_filter_window=trend_filter_window,
                pullback_lookback=pullback_lookback,
                trend_slope_lookback=trend_slope_lookback,
                atr_window=atr_window,
                min_atr_pct=min_atr_pct,
                max_atr_pct=max_atr_pct,
                pullback_reclaim_tolerance_pct=reclaim_tolerance,
            )
        )
    return configs


def sweep_result_row(
    config: SweepConfig,
    result: BacktestResult,
    symbols: list[str],
    train_result: BacktestResult | None = None,
    test_result: BacktestResult | None = None,
    min_test_trades: int = 8,
    min_test_exposure: float = 0.02,
) -> dict:
    row = asdict(config)
    row.update(
        {
            "symbols": ",".join(symbols),
            "num_symbols": len(symbols),
            "total_return": result.total_return,
            "buy_and_hold_return": result.buy_and_hold_return,
            "outperformance_vs_buy_and_hold": result.outperformance_vs_buy_and_hold,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "average_win": result.average_win,
            "average_loss": result.average_loss,
            "exposure_pct": result.exposure_pct,
            "num_trades": result.num_trades,
        }
    )

    if train_result is not None:
        row.update(
            {
                "train_total_return": train_result.total_return,
                "train_buy_and_hold_return": train_result.buy_and_hold_return,
                "train_outperformance_vs_buy_and_hold": train_result.outperformance_vs_buy_and_hold,
                "train_sharpe": train_result.sharpe,
                "train_max_drawdown": train_result.max_drawdown,
                "train_win_rate": train_result.win_rate,
                "train_profit_factor": train_result.profit_factor,
                "train_exposure_pct": train_result.exposure_pct,
                "train_num_trades": train_result.num_trades,
            }
        )

    if test_result is not None:
        paper_candidate = (
            test_result.total_return > 0
            and test_result.sharpe > 0
            and test_result.profit_factor > 1.0
            and test_result.num_trades >= min_test_trades
            and test_result.exposure_pct >= min_test_exposure
        )
        paper_score = (
            (test_result.sharpe * 4.0)
            + (test_result.total_return * 100.0)
            + ((test_result.profit_factor - 1.0) * 3.0)
            + (test_result.win_rate * 2.0)
            + min(test_result.exposure_pct, 0.20)
            - (abs(test_result.max_drawdown) * 60.0)
            - (max(min_test_trades - test_result.num_trades, 0) * 0.10)
            - (max(min_test_exposure - test_result.exposure_pct, 0.0) * 25.0)
        )
        row.update(
            {
                "test_total_return": test_result.total_return,
                "test_buy_and_hold_return": test_result.buy_and_hold_return,
                "test_outperformance_vs_buy_and_hold": test_result.outperformance_vs_buy_and_hold,
                "test_sharpe": test_result.sharpe,
                "test_max_drawdown": test_result.max_drawdown,
                "test_win_rate": test_result.win_rate,
                "test_profit_factor": test_result.profit_factor,
                "test_exposure_pct": test_result.exposure_pct,
                "test_num_trades": test_result.num_trades,
                "paper_candidate": paper_candidate,
                "paper_score": paper_score,
            }
        )
    return row


def rank_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    if {"paper_candidate", "paper_score", "test_sharpe"}.issubset(results.columns):
        sort_columns = [
            "paper_candidate",
            "paper_score",
            "test_sharpe",
            "test_total_return",
            "test_profit_factor",
            "sharpe",
            "total_return",
            "profit_factor",
            "max_drawdown",
        ]
        ascending = [False] * len(sort_columns)
    else:
        sort_columns = ["sharpe", "total_return", "profit_factor", "max_drawdown"]
        ascending = [False, False, False, False]

    ranked = results.sort_values(
        by=sort_columns,
        ascending=ascending,
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
