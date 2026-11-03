import numpy as np
import pandas as pd

from src.backtest.evidence_momentum import (
    EvidenceMomentumConfig,
    backtest_evidence_momentum_portfolio,
    build_indicator_frame,
    default_optimization_grid,
    optimize_evidence_momentum,
)


def _make_trending_bars(periods: int = 2400, base_price: float = 100.0, slope: float = 18.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    trend = np.linspace(0, slope, periods)
    wave = 2.0 * np.sin(np.linspace(0, 30, periods))
    close = base_price + trend + wave
    return pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(periods, 1_000_000),
        },
        index=index,
    )


def test_indicator_frame_contains_evidence_momentum_columns():
    bars = _make_trending_bars()
    proxy = _make_trending_bars(base_price=105.0, slope=20.0)
    config = EvidenceMomentumConfig()

    frame = build_indicator_frame(
        bars=bars,
        proxy_bars=proxy,
        config=config,
        timeframe_minutes=60,
    )

    expected_columns = {
        "atr",
        "realized_vol",
        "mom_short",
        "mom_medium",
        "mom_long",
        "high_ratio",
        "score",
        "long_entry_setup",
        "long_exit_signal",
        "short_entry_setup",
        "short_exit_signal",
    }
    assert expected_columns.issubset(frame.columns)
    assert frame["long_entry_setup"].dtype == bool
    assert frame["short_entry_setup"].dtype == bool


def test_backtest_evidence_momentum_portfolio_runs_on_synthetic_data():
    bars_by_symbol = {
        "QQQ": _make_trending_bars(base_price=100.0, slope=25.0),
        "IWM": _make_trending_bars(base_price=90.0, slope=12.0),
        "TLT": _make_trending_bars(base_price=95.0, slope=5.0),
    }
    proxy = _make_trending_bars(base_price=105.0, slope=18.0)
    config = EvidenceMomentumConfig(
        short_lookback_sessions=10,
        medium_lookback_sessions=30,
        long_lookback_sessions=60,
        long_sma_sessions=80,
        high_lookback_sessions=120,
        rebalance_interval_sessions=1,
        top_n=2,
    )

    result = backtest_evidence_momentum_portfolio(
        bars_by_symbol=bars_by_symbol,
        proxy_bars=proxy,
        config=config,
        timeframe_minutes=60,
        bars_per_year_value=252 * 6.5,
    )

    assert isinstance(result.trades, pd.DataFrame)
    assert "Annualized return" in result.summary()
    assert 0.0 <= result.exposure_pct <= 1.0


def test_optimizer_returns_ranked_dataframe_for_evidence_momentum():
    bars_by_symbol = {
        "QQQ": _make_trending_bars(base_price=100.0, slope=25.0),
        "IWM": _make_trending_bars(base_price=95.0, slope=14.0),
    }
    proxy = _make_trending_bars(base_price=105.0, slope=20.0)
    base_config = EvidenceMomentumConfig()
    parameter_grid = {
        "short_lookback_sessions": [10],
        "medium_lookback_sessions": [30, 45],
        "long_lookback_sessions": [60],
        "skip_recent_sessions": [0],
        "min_high_ratio": [0.80],
        "rebalance_interval_sessions": [1],
        "top_n": [1, 2],
        "atr_stop_mult": [1.5],
        "atr_trailing_stop_mult": [2.5],
        "risk_per_trade_pct": [0.003],
        "max_position_pct": [0.10],
        "max_gross_exposure_pct": [0.40],
        "allow_short": [False],
    }

    results_df, best_config, best_result = optimize_evidence_momentum(
        bars_by_symbol=bars_by_symbol,
        proxy_bars=proxy,
        base_config=base_config,
        timeframe_minutes=60,
        parameter_grid=parameter_grid,
        bars_per_year_value=252 * 6.5,
        evaluation_mode="active",
        min_trade_count=1,
    )

    assert not results_df.empty
    assert "rank" in results_df.columns
    assert "meets_trade_goal" in results_df.columns
    assert isinstance(best_config, EvidenceMomentumConfig)
    assert isinstance(best_result.trades, pd.DataFrame)


def test_aggressive_grid_includes_allow_short():
    grid = default_optimization_grid(60, profile="aggressive")
    assert "allow_short" in grid


def test_active_grid_pushes_frequency_controls():
    grid = default_optimization_grid(60, profile="active")
    assert "max_open_positions" in grid
    assert grid["rebalance_interval_sessions"] == [1, 2, 3]
