"""Run a parameter sweep over the portfolio backtester.

Usage:
    python run_parameter_sweep.py
    python run_parameter_sweep.py --timeframes 30 60 --max-configs 24 --top 10
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from loguru import logger

from config.settings import PROJECT_ROOT, settings
from src.backtest.backtester import backtest_portfolio, backtest_sma
from src.backtest.sweep import (
    SweepConfig,
    bars_per_year_for_timeframe,
    generate_sweep_configs,
    rank_sweep_results,
    sample_sweep_configs,
    sweep_result_row,
)
from src.broker.alpaca_client import _to_alpaca_timeframe


DEFAULT_SHORT_LONG_PAIRS = [(30, 100), (20, 50), (10, 40)]
DEFAULT_TREND_FILTERS = [100, 200]
DEFAULT_PULLBACK_LOOKBACKS = [3, 5]
DEFAULT_TREND_SLOPE_LOOKBACKS = [5, 10]
DEFAULT_ATR_WINDOWS = [14]
DEFAULT_ATR_BANDS = [(0.002, 0.03), (0.003, 0.04)]
DEFAULT_RECLAIM_TOLERANCES = [0.001, 0.0025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter sweep for the algobot backtester")
    parser.add_argument("--days", type=int, default=365, help="Lookback period to fetch")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top-ranked configurations to print",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Optional cap on how many generated configurations to evaluate",
    )
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="*",
        default=[settings.bar_timeframe_minutes],
        help="Minute timeframes to evaluate, e.g. --timeframes 30 60",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of bars used for in-sample training before out-of-sample test",
    )
    parser.add_argument(
        "--min-test-trades",
        type=int,
        default=8,
        help="Minimum out-of-sample trades for a config to qualify as a paper candidate",
    )
    parser.add_argument(
        "--min-test-exposure",
        type=float,
        default=0.02,
        help="Minimum out-of-sample average exposure required for paper-candidate ranking",
    )
    return parser.parse_args()


def fetch_history(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe_minutes: int,
    days: int,
) -> pd.DataFrame:
    end = datetime.now(UTC) - timedelta(minutes=20)
    start = end - timedelta(days=days)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_to_alpaca_timeframe(timeframe_minutes),
        start=start,
        end=end,
    )
    df = client.get_stock_bars(request).df
    if df.empty:
        return df
    return df.xs(symbol, level=0) if hasattr(df.index, "levels") else df


def build_common_kwargs(config: SweepConfig) -> dict:
    return {
        "short_window": config.short_window,
        "long_window": config.long_window,
        "trend_filter_window": config.trend_filter_window,
        "pullback_lookback": config.pullback_lookback,
        "trend_slope_lookback": config.trend_slope_lookback,
        "atr_window": config.atr_window,
        "min_atr_pct": config.min_atr_pct,
        "max_atr_pct": config.max_atr_pct,
        "pullback_reclaim_tolerance_pct": config.pullback_reclaim_tolerance_pct,
        "relative_strength_lookback": settings.relative_strength_lookback_bars,
        "max_relative_strength_rank": settings.max_relative_strength_rank,
        "min_relative_strength_return": settings.min_relative_strength_return,
        "max_position_pct": settings.max_position_pct,
        "risk_per_trade_pct": settings.risk_per_trade_pct,
        "stop_loss_pct": settings.stop_loss_pct,
        "take_profit_pct": settings.take_profit_pct,
        "bars_per_year": bars_per_year_for_timeframe(config.timeframe_minutes),
    }


def config_env_overrides(config: SweepConfig) -> list[str]:
    return [
        f"BAR_TIMEFRAME_MINUTES={config.timeframe_minutes}",
        f"SHORT_SMA_WINDOW={config.short_window}",
        f"LONG_SMA_WINDOW={config.long_window}",
        f"REGIME_FILTER_WINDOW={config.trend_filter_window}",
        f"PULLBACK_LOOKBACK_BARS={config.pullback_lookback}",
        f"TREND_SLOPE_LOOKBACK_BARS={config.trend_slope_lookback}",
        f"ATR_WINDOW={config.atr_window}",
        f"MIN_ATR_PCT={config.min_atr_pct}",
        f"MAX_ATR_PCT={config.max_atr_pct}",
        f"PULLBACK_RECLAIM_TOLERANCE_PCT={config.pullback_reclaim_tolerance_pct}",
        f"RELATIVE_STRENGTH_LOOKBACK_BARS={settings.relative_strength_lookback_bars}",
        f"MAX_RELATIVE_STRENGTH_RANK={settings.max_relative_strength_rank}",
        f"MIN_RELATIVE_STRENGTH_RETURN={settings.min_relative_strength_return}",
    ]


def split_history(
    bars: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bars.empty:
        return bars, bars

    split_index = int(len(bars) * train_ratio)
    split_index = max(2, min(split_index, len(bars) - 2))
    return bars.iloc[:split_index].copy(), bars.iloc[split_index:].copy()


def evaluate_config(
    bars_by_symbol: dict[str, pd.DataFrame],
    common_kwargs: dict,
):
    if len(bars_by_symbol) == 1:
        _, bars = next(iter(bars_by_symbol.items()))
        return backtest_sma(bars, **common_kwargs)
    return backtest_portfolio(
        bars_by_symbol,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_open_positions=settings.max_open_positions,
        **common_kwargs,
    )


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    settings.validate()

    configs = generate_sweep_configs(
        timeframes=args.timeframes,
        short_long_pairs=DEFAULT_SHORT_LONG_PAIRS,
        trend_filters=DEFAULT_TREND_FILTERS,
        pullback_lookbacks=DEFAULT_PULLBACK_LOOKBACKS,
        trend_slope_lookbacks=DEFAULT_TREND_SLOPE_LOOKBACKS,
        atr_windows=DEFAULT_ATR_WINDOWS,
        atr_bands=DEFAULT_ATR_BANDS,
        reclaim_tolerances=DEFAULT_RECLAIM_TOLERANCES,
    )
    if not (0.50 <= args.train_ratio <= 0.90):
        raise ValueError("--train-ratio must be between 0.50 and 0.90")
    if args.min_test_trades < 1:
        raise ValueError("--min-test-trades must be >= 1")
    if args.min_test_exposure < 0:
        raise ValueError("--min-test-exposure must be >= 0")

    if args.max_configs and args.max_configs > 0:
        configs = sample_sweep_configs(configs, args.max_configs)
    if not configs:
        print("No valid sweep configurations were generated.")
        return

    timeframes = sorted({config.timeframe_minutes for config in configs})
    print(
        f"Preparing sweep for {len(configs)} configuration(s), "
        f"{len(settings.symbols)} symbol(s), timeframes={timeframes}"
    )

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    bars_cache: dict[int, dict[str, pd.DataFrame]] = {}
    for timeframe in timeframes:
        print(f"\nFetching data for timeframe {timeframe}m...")
        timeframe_bars = {}
        for symbol in settings.symbols:
            bars = fetch_history(client, symbol, timeframe, args.days)
            if bars.empty:
                print(f"{symbol}: no data for {timeframe}m; skipping.")
                continue
            timeframe_bars[symbol] = bars
            print(
                f"{symbol}: {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}."
            )
        bars_cache[timeframe] = timeframe_bars

    results_rows: list[dict] = []
    results_by_label: dict[str, tuple[SweepConfig, object, object, object]] = {}

    for index, config in enumerate(configs, start=1):
        bars_by_symbol = bars_cache.get(config.timeframe_minutes, {})
        if not bars_by_symbol:
            continue
        common_kwargs = build_common_kwargs(config)
        result = evaluate_config(bars_by_symbol, common_kwargs)
        train_bars = {}
        test_bars = {}
        for symbol, bars in bars_by_symbol.items():
            train_split, test_split = split_history(bars, args.train_ratio)
            if not train_split.empty:
                train_bars[symbol] = train_split
            if not test_split.empty:
                test_bars[symbol] = test_split
        train_result = evaluate_config(train_bars, common_kwargs) if train_bars else None
        test_result = evaluate_config(test_bars, common_kwargs) if test_bars else None

        results_rows.append(
            sweep_result_row(
                config=config,
                result=result,
                symbols=list(bars_by_symbol.keys()),
                train_result=train_result,
                test_result=test_result,
                min_test_trades=args.min_test_trades,
                min_test_exposure=args.min_test_exposure,
            )
        )
        results_by_label[config.label()] = (config, result, train_result, test_result)

        test_summary = ""
        if test_result is not None:
            test_summary = (
                f" | test_ret={test_result.total_return:+.2%}"
                f" | test_sharpe={test_result.sharpe:.2f}"
            )

        print(
            f"[{index}/{len(configs)}] {config.label()} | "
            f"ret={result.total_return:+.2%} | sharpe={result.sharpe:.2f} | "
            f"dd={result.max_drawdown:.2%}{test_summary}"
        )

    if not results_rows:
        print("Sweep completed, but no results were produced.")
        return

    ranked = rank_sweep_results(pd.DataFrame(results_rows))
    best_row = ranked.iloc[0]
    best_config, best_result, best_train_result, best_test_result = results_by_label[
        SweepConfig(
            timeframe_minutes=int(best_row["timeframe_minutes"]),
            short_window=int(best_row["short_window"]),
            long_window=int(best_row["long_window"]),
            trend_filter_window=int(best_row["trend_filter_window"]),
            pullback_lookback=int(best_row["pullback_lookback"]),
            trend_slope_lookback=int(best_row["trend_slope_lookback"]),
            atr_window=int(best_row["atr_window"]),
            min_atr_pct=float(best_row["min_atr_pct"]),
            max_atr_pct=float(best_row["max_atr_pct"]),
            pullback_reclaim_tolerance_pct=float(best_row["pullback_reclaim_tolerance_pct"]),
        ).label()
    ]
    output_dir = PROJECT_ROOT / "outputs" / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"sweep_summary_{timestamp}.csv"
    ranked.to_csv(summary_path, index=False)

    best_trade_log_path = output_dir / f"best_{best_config.label()}_{timestamp}_trades.csv"
    best_result.trades.to_csv(best_trade_log_path, index=False)
    best_test_trade_log_path = None
    if best_test_result is not None:
        best_test_trade_log_path = output_dir / f"best_{best_config.label()}_{timestamp}_test_trades.csv"
        best_test_result.trades.to_csv(best_test_trade_log_path, index=False)
    best_overrides_path = output_dir / f"best_{best_config.label()}_{timestamp}.env"
    best_overrides = config_env_overrides(best_config)
    best_overrides_path.write_text("\n".join(best_overrides) + "\n", encoding="utf-8")

    print("\n=== Top Configurations ===")
    columns = [
        "rank",
        "timeframe_minutes",
        "short_window",
        "long_window",
        "trend_filter_window",
        "pullback_lookback",
        "trend_slope_lookback",
        "min_atr_pct",
        "max_atr_pct",
        "pullback_reclaim_tolerance_pct",
        "total_return",
        "sharpe",
        "test_total_return",
        "test_sharpe",
        "paper_candidate",
        "paper_score",
        "max_drawdown",
        "profit_factor",
        "num_trades",
    ]
    print(ranked.loc[: args.top - 1, columns].to_string(index=False))
    print(f"\nSummary CSV: {summary_path}")
    print(f"Best trade log: {best_trade_log_path}")
    if best_test_trade_log_path is not None:
        print(f"Best test trade log: {best_test_trade_log_path}")
    print(f"Best env overrides: {best_overrides_path}")
    print("\nSuggested strategy overrides:")
    for line in best_overrides:
        print(line)


if __name__ == "__main__":
    main()
