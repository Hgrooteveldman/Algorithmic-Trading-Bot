import numpy as np
import pandas as pd

from src.backtest.volatility_mean_reversion import (
    MeanReversionConfig,
    backtest_mean_reversion_portfolio,
    build_indicator_frame,
    default_optimization_grid,
    generate_candidate_configs,
    limit_candidate_configs,
    optimize_mean_reversion,
)


def _make_mean_reverting_bars(periods: int = 420, base_price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    trend = np.linspace(0, 6, periods)
    wave = 3.5 * np.sin(np.linspace(0, 42, periods))
    noise = 0.35 * np.sin(np.linspace(0, 140, periods))
    close = base_price + trend + wave + noise
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.9,
            "low": close - 0.9,
            "close": close,
            "volume": np.full(periods, 1_000_000),
        },
        index=index,
    )


def test_indicator_frame_contains_mean_reversion_columns():
    bars = _make_mean_reverting_bars()
    proxy = _make_mean_reverting_bars(base_price=105.0)
    config = MeanReversionConfig()

    frame = build_indicator_frame(bars, proxy_bars=proxy, config=config)

    expected_columns = {
        "atr",
        "bb_mid",
        "bb_lower",
        "kc_lower",
        "rsi",
        "regime_ok",
        "zscore",
        "long_entry_setup",
        "long_exit_signal",
        "short_entry_setup",
        "short_exit_signal",
    }
    assert expected_columns.issubset(frame.columns)
    assert frame["long_entry_setup"].dtype == bool
    assert frame["long_exit_signal"].dtype == bool
    assert frame["short_entry_setup"].dtype == bool
    assert frame["short_exit_signal"].dtype == bool


def test_backtest_mean_reversion_portfolio_runs_on_synthetic_data():
    bars_by_symbol = {
        "QQQ": _make_mean_reverting_bars(base_price=100.0),
        "IWM": _make_mean_reverting_bars(base_price=90.0),
    }
    proxy = _make_mean_reverting_bars(base_price=105.0)
    config = MeanReversionConfig(
        bollinger_window=18,
        rsi_length=5,
        rsi_entry_threshold=35.0,
        rsi_exit_threshold=55.0,
        atr_stop_mult=1.5,
        atr_trailing_stop_mult=2.0,
        atr_target_mult=1.0,
        max_holding_bars=12,
    )

    result = backtest_mean_reversion_portfolio(
        bars_by_symbol=bars_by_symbol,
        proxy_bars=proxy,
        config=config,
        bars_per_year_value=252 * 6.5,
    )

    assert isinstance(result.trades, pd.DataFrame)
    assert "Annualized return" in result.summary()
    assert 0.0 <= result.exposure_pct <= 1.0
    assert isinstance(result.circuit_breaker_triggered, bool)


def test_backtest_mean_reversion_portfolio_can_take_short_trades():
    bars_by_symbol = {
        "QQQ": _make_mean_reverting_bars(base_price=100.0),
        "IWM": _make_mean_reverting_bars(base_price=95.0),
    }
    proxy = _make_mean_reverting_bars(base_price=105.0)
    config = MeanReversionConfig(
        allow_short=True,
        require_asset_regime=False,
        require_reversal_bar=False,
        entry_zscore_threshold=0.6,
        bollinger_window=18,
        rsi_length=5,
        rsi_entry_threshold=35.0,
        rsi_exit_threshold=55.0,
        atr_stop_mult=1.5,
        atr_trailing_stop_mult=2.0,
        atr_target_mult=1.0,
        max_holding_bars=12,
    )

    result = backtest_mean_reversion_portfolio(
        bars_by_symbol=bars_by_symbol,
        proxy_bars=proxy,
        config=config,
        bars_per_year_value=252 * 6.5,
    )

    assert isinstance(result.trades, pd.DataFrame)
    assert not result.trades.empty
    assert "SHORT" in set(result.trades["side"])


def test_optimizer_returns_ranked_dataframe():
    bars_by_symbol = {
        "QQQ": _make_mean_reverting_bars(base_price=100.0),
        "IWM": _make_mean_reverting_bars(base_price=95.0),
    }
    proxy = _make_mean_reverting_bars(base_price=105.0)
    base_config = MeanReversionConfig()
    parameter_grid = {
        "bollinger_window": [18],
        "bollinger_std_mult": [1.8, 2.0],
        "rsi_length": [5],
        "rsi_entry_threshold": [28.0, 32.0],
        "rsi_exit_threshold": [55.0],
        "atr_stop_mult": [1.5],
        "atr_trailing_stop_mult": [2.0],
        "atr_target_mult": [1.0],
        "max_holding_bars": [12],
    }

    results_df, best_config, best_result = optimize_mean_reversion(
        bars_by_symbol=bars_by_symbol,
        proxy_bars=proxy,
        base_config=base_config,
        parameter_grid=parameter_grid,
        target_annual_return=0.08,
        max_drawdown_limit=0.25,
        timeframe_minutes=60,
        bars_per_year_value=252 * 6.5,
        evaluation_mode="active",
        min_trade_count=1,
    )

    assert not results_df.empty
    assert "rank" in results_df.columns
    assert "meets_trade_goal" in results_df.columns
    assert isinstance(best_config, MeanReversionConfig)
    assert isinstance(best_result.trades, pd.DataFrame)


def test_limit_candidate_configs_spreads_across_grid():
    base_config = MeanReversionConfig()
    parameter_grid = {
        "bollinger_window": [18, 20, 24],
        "rsi_length": [5, 7, 9],
        "atr_stop_mult": [1.5, 2.0, 2.5],
    }
    candidates = generate_candidate_configs(
        base_config=base_config,
        parameter_grid=parameter_grid,
    )

    limited = limit_candidate_configs(candidates, max_configs=5)

    assert len(candidates) == 27
    assert len(limited) == 5
    assert limited[0] == candidates[0]
    assert limited[-1] == candidates[-1]


def test_aggressive_optimization_grid_exposes_new_entry_and_risk_knobs():
    grid = default_optimization_grid(60, profile="aggressive")

    assert "entry_zscore_threshold" in grid
    assert "entry_reclaim_atr_buffer" in grid
    assert "risk_per_trade_pct" in grid
    assert "max_position_pct" in grid
    assert "max_gross_exposure_pct" in grid


def test_active_optimization_grid_pushes_trade_frequency_knobs():
    grid = default_optimization_grid(60, profile="active")

    assert "max_open_positions" in grid
    assert grid["max_holding_bars"] == [4, 8, 12]
