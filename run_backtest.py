"""Run the upgraded research backtester.

Examples:
    ./.venv/bin/python run_backtest.py
    ./.venv/bin/python run_backtest.py --timeframe 15 --days 540
    ./.venv/bin/python run_backtest.py --optimize --timeframe 60 --top 10
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from loguru import logger

from config.settings import PROJECT_ROOT, settings
from src.backtest.evidence_momentum import (
    EvidenceMomentumConfig,
    backtest_evidence_momentum_portfolio,
    default_optimization_grid as default_evidence_optimization_grid,
    optimize_evidence_momentum,
)
from src.backtest.volatility_mean_reversion import (
    MeanReversionConfig,
    bars_per_year,
    default_optimization_grid,
    backtest_mean_reversion_portfolio,
    optimize_mean_reversion,
)
from src.analytics.metrics import summarize
from src.analytics.report import ValidationSummary, write_report
from src.broker.alpaca_client import _to_alpaca_timeframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research backtester for the bot's strategy families on Alpaca equities."
    )
    parser.add_argument("--days", type=int, default=365, help="Historical lookback to fetch.")
    parser.add_argument(
        "--timeframe",
        type=int,
        default=settings.bar_timeframe_minutes,
        choices=[1, 5, 15, 30, 60],
        help="Bar size in minutes. 15 or 60 is the intended research path.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=settings.symbols,
        help="Symbols to backtest. Defaults to the configured universe.",
    )
    parser.add_argument(
        "--market-proxy",
        default="SPY",
        help="Proxy used for the macro regime filter.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run a parameter grid search instead of a single backtest.",
    )
    parser.add_argument(
        "--strategy",
        default="evidence_momentum",
        choices=["evidence_momentum", "mean_reversion"],
        help=(
            "evidence_momentum follows the PDF's journal-backed direction. "
            "mean_reversion keeps the prior retail-style research path for comparison."
        ),
    )
    parser.add_argument(
        "--optimization-profile",
        default="balanced",
        choices=["balanced", "aggressive", "active"],
        help=(
            "Balanced keeps the search tighter and safer. Aggressive searches "
            "looser entries and larger capped exposure in pursuit of higher returns. "
            "Active explicitly searches faster-turnover configurations."
        ),
    )
    parser.add_argument(
        "--evaluation-mode",
        default="auto",
        choices=["auto", "balanced", "active"],
        help=(
            "How the optimizer ranks strategies. Active mode rewards configs "
            "that meet the return/drawdown goals while also trading often enough."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many optimizer rows to print when --optimize is used.",
    )
    parser.add_argument(
        "--starting-cash",
        type=float,
        default=100_000.0,
        help="Initial capital for the simulation.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Enable mirrored short trades in the research engine where supported.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help=(
            "Optional cap on optimizer configurations. Useful for a faster "
            "sanity-check run before the full grid search."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print optimizer progress every N configurations.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=0,
        help=(
            "Optional minimum trade count for optimizer ranking. In active "
            "evaluation mode, leaving this at 0 uses a timeframe-based default."
        ),
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


def build_default_mean_reversion_config() -> MeanReversionConfig:
    return MeanReversionConfig(
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_open_positions=settings.max_open_positions,
        circuit_breaker_drawdown_pct=0.25,
        allow_short=settings.mr_allow_short,
    )


def build_default_evidence_momentum_config() -> EvidenceMomentumConfig:
    return EvidenceMomentumConfig(
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_open_positions=settings.max_open_positions,
        circuit_breaker_drawdown_pct=0.25,
        allow_short=False,
    )


def fetch_universe(
    symbols: list[str],
    market_proxy: str,
    timeframe_minutes: int,
    days: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    requested_symbols = list(dict.fromkeys([market_proxy, *symbols]))

    print(f"Fetching {len(requested_symbols)} symbol(s) at {timeframe_minutes}-min...")
    proxy_bars = pd.DataFrame()
    for symbol in requested_symbols:
        bars = fetch_history(client, symbol, timeframe_minutes=timeframe_minutes, days=days)
        if bars.empty:
            print(f"{symbol}: no data returned; skipping.")
            continue
        print(f"{symbol}: {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}.")
        if symbol == market_proxy:
            proxy_bars = bars
        if symbol in symbols:
            bars_by_symbol[symbol] = bars

    return bars_by_symbol, proxy_bars


def save_backtest_artifacts(
    result,
    config,
    symbols: list[str],
    timeframe_minutes: int,
    strategy_name: str,
    periods_per_year: float | None = None,
) -> None:
    output_dir = PROJECT_ROOT / "outputs" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_name = f"{strategy_name}_{len(symbols)}symbols_{timeframe_minutes}m"
    trades_path = output_dir / f"{output_name}_{timestamp}_trades.csv"
    result.trades.to_csv(trades_path, index=False)

    print("\n=== Backtest result ===")
    print(result.summary())
    print("\nConfig:")
    for key, value in asdict(config).items():
        print(f"- {key}: {value}")
    print(f"\nTrade log: {trades_path}")
    if periods_per_year and len(result.equity_curve) > 1:
        summary = summarize(result.equity_curve, periods_per_year)
        _, report_html = write_report(output_dir, f"{output_name}_{timestamp}_report",
                                      f"{strategy_name} backtest ({timeframe_minutes}m)", result.equity_curve, summary)
        print(f"Report: {report_html}")


def save_optimization_artifacts(
    results_df: pd.DataFrame,
    best_config,
    best_result,
    symbols: list[str],
    timeframe_minutes: int,
    top: int,
    strategy_name: str,
    periods_per_year: float | None = None,
) -> None:
    output_dir = PROJECT_ROOT / "outputs" / "optimizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"{strategy_name}_summary_{timeframe_minutes}m_{timestamp}.csv"
    best_trades_path = output_dir / f"{strategy_name}_best_{timeframe_minutes}m_{timestamp}_trades.csv"

    results_df.to_csv(summary_path, index=False)
    best_result.trades.to_csv(best_trades_path, index=False)

    print("\n=== Optimization results ===")
    if results_df.empty:
        print("No optimization rows were produced.")
    else:
        printable_columns = [
            "rank",
            "meets_target",
            "meets_trade_goal",
            "annualized_return",
            "total_return",
            "sharpe",
            "max_drawdown",
            "profit_factor",
            "num_trades",
            "exposure_pct",
            "bollinger_window",
            "bollinger_std_mult",
            "keltner_atr_mult",
            "rsi_length",
            "rsi_entry_threshold",
            "rsi_exit_threshold",
            "require_asset_regime",
            "asset_regime_buffer_pct",
            "atr_stop_mult",
            "atr_trailing_stop_mult",
            "atr_target_mult",
            "mean_exit_atr_buffer",
            "max_holding_bars",
            "require_reversal_bar",
            "entry_zscore_threshold",
            "entry_reclaim_atr_buffer",
            "risk_per_trade_pct",
            "max_position_pct",
            "max_gross_exposure_pct",
            "max_open_positions",
            "allow_short",
            "short_lookback_sessions",
            "medium_lookback_sessions",
            "long_lookback_sessions",
            "skip_recent_sessions",
            "min_high_ratio",
            "rebalance_interval_sessions",
            "top_n",
        ]
        printable_columns = [column for column in printable_columns if column in results_df.columns]
        print(results_df[printable_columns].head(top).to_string(index=False))

    print("\n=== Best configuration ===")
    for key, value in asdict(best_config).items():
        print(f"- {key}: {value}")
    print("\n=== Best result ===")
    print(best_result.summary())
    print(f"\nSummary CSV: {summary_path}")
    print(f"Best trade log: {best_trades_path}")
    if periods_per_year and len(best_result.equity_curve) > 1 and "sharpe" in results_df.columns:
        # Deflate the winner's Sharpe by every configuration that was tried.
        trial_sharpes = (results_df["sharpe"].astype(float) / periods_per_year**0.5).tolist()
        summary = summarize(best_result.equity_curve, periods_per_year,
                            n_trials=len(trial_sharpes), trial_sharpes_per_period=trial_sharpes)
        _, report_html = write_report(
            output_dir, f"{strategy_name}_best_{timeframe_minutes}m_{timestamp}_report",
            f"{strategy_name} optimisation winner ({timeframe_minutes}m)", best_result.equity_curve, summary,
            ValidationSummary(n_trials=len(trial_sharpes)),
        )
        print(f"Deflated Sharpe of the winner: {summary.dsr:.1%} over {len(trial_sharpes)} configurations")
        print(f"Report: {report_html}")


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    settings.validate()

    symbols = [symbol.upper() for symbol in args.symbols]
    market_proxy = args.market_proxy.upper()
    if args.strategy == "evidence_momentum":
        config = build_default_evidence_momentum_config()
    else:
        config = build_default_mean_reversion_config()
    if args.allow_short:
        config = replace(config, allow_short=True)
    bars_by_symbol, proxy_bars = fetch_universe(
        symbols=symbols,
        market_proxy=market_proxy,
        timeframe_minutes=args.timeframe,
        days=args.days,
    )
    if not bars_by_symbol:
        print("No symbol data returned. Check credentials, symbols, or market-data access.")
        return
    if proxy_bars.empty:
        print(f"Market proxy {market_proxy} returned no data. The regime filter needs that series.")
        return

    bar_factor = bars_per_year(args.timeframe)
    evaluation_mode = (
        "active"
        if args.evaluation_mode == "auto" and args.optimization_profile == "active"
        else ("balanced" if args.evaluation_mode == "auto" else args.evaluation_mode)
    )
    min_trade_count = args.min_trades if args.min_trades > 0 else None

    if args.optimize:
        if args.strategy == "evidence_momentum":
            parameter_grid = default_evidence_optimization_grid(
                args.timeframe,
                profile=args.optimization_profile,
            )
            results_df, best_config, best_result = optimize_evidence_momentum(
                bars_by_symbol=bars_by_symbol,
                proxy_bars=proxy_bars,
                base_config=config,
                timeframe_minutes=args.timeframe,
                parameter_grid=parameter_grid,
                target_annual_return=0.08,
                max_drawdown_limit=0.25,
                starting_cash=args.starting_cash,
                commission_bps=1.0,
                slippage_bps=2.0,
                bars_per_year_value=bar_factor,
                max_configs=(args.max_configs if args.max_configs > 0 else None),
                progress_every=args.progress_every,
                evaluation_mode=evaluation_mode,
                min_trade_count=min_trade_count,
            )
        else:
            parameter_grid = default_optimization_grid(
                args.timeframe,
                profile=args.optimization_profile,
            )
            results_df, best_config, best_result = optimize_mean_reversion(
                bars_by_symbol=bars_by_symbol,
                proxy_bars=proxy_bars,
                base_config=config,
                parameter_grid=parameter_grid,
                target_annual_return=0.08,
                max_drawdown_limit=0.25,
                starting_cash=args.starting_cash,
                commission_bps=1.0,
                slippage_bps=2.0,
                timeframe_minutes=args.timeframe,
                bars_per_year_value=bar_factor,
                max_configs=(args.max_configs if args.max_configs > 0 else None),
                progress_every=args.progress_every,
                evaluation_mode=evaluation_mode,
                min_trade_count=min_trade_count,
            )
        save_optimization_artifacts(
            results_df=results_df,
            best_config=best_config,
            best_result=best_result,
            symbols=symbols,
            timeframe_minutes=args.timeframe,
            top=args.top,
            strategy_name=args.strategy,
            periods_per_year=bar_factor,
        )
    else:
        if args.strategy == "evidence_momentum":
            result = backtest_evidence_momentum_portfolio(
                bars_by_symbol=bars_by_symbol,
                proxy_bars=proxy_bars,
                config=config,
                timeframe_minutes=args.timeframe,
                starting_cash=args.starting_cash,
                commission_bps=1.0,
                slippage_bps=2.0,
                bars_per_year_value=bar_factor,
            )
        else:
            result = backtest_mean_reversion_portfolio(
                bars_by_symbol=bars_by_symbol,
                proxy_bars=proxy_bars,
                config=config,
                starting_cash=args.starting_cash,
                commission_bps=1.0,
                slippage_bps=2.0,
                bars_per_year_value=bar_factor,
            )
        save_backtest_artifacts(
            result=result,
            config=config,
            symbols=symbols,
            timeframe_minutes=args.timeframe,
            strategy_name=args.strategy,
            periods_per_year=bar_factor,
        )

    print(
        "\nReminder: an optimized backtest is not a guarantee of live edge. "
        "Validate out-of-sample and paper-trade before risking capital."
    )


if __name__ == "__main__":
    main()
