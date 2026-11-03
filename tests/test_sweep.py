import pandas as pd

from src.backtest.sweep import (
    SweepConfig,
    generate_sweep_configs,
    rank_sweep_results,
    sample_sweep_configs,
)


def test_generate_sweep_configs_filters_invalid_combinations():
    configs = generate_sweep_configs(
        timeframes=[30, 60, 7],
        short_long_pairs=[(10, 40), (50, 20)],
        trend_filters=[80, 200],
        pullback_lookbacks=[3],
        trend_slope_lookbacks=[5],
        atr_windows=[14],
        atr_bands=[(0.002, 0.03), (0.04, 0.03)],
        reclaim_tolerances=[0.001],
    )

    assert all(config.timeframe_minutes in {30, 60} for config in configs)
    assert all(config.short_window < config.long_window < config.trend_filter_window for config in configs)
    assert all(config.min_atr_pct < config.max_atr_pct for config in configs)


def test_rank_sweep_results_sorts_best_first():
    df = pd.DataFrame(
        [
            {
                "timeframe_minutes": 60,
                "short_window": 20,
                "long_window": 50,
                "trend_filter_window": 200,
                "pullback_lookback": 5,
                "trend_slope_lookback": 10,
                "atr_window": 14,
                "min_atr_pct": 0.003,
                "max_atr_pct": 0.04,
                "pullback_reclaim_tolerance_pct": 0.0025,
                "total_return": 0.01,
                "sharpe": 0.5,
                "profit_factor": 1.1,
                "max_drawdown": -0.05,
                "test_total_return": -0.01,
                "test_sharpe": -0.2,
                "test_profit_factor": 0.9,
                "paper_candidate": False,
                "paper_score": -1.0,
            },
            {
                "timeframe_minutes": 30,
                "short_window": 10,
                "long_window": 40,
                "trend_filter_window": 100,
                "pullback_lookback": 3,
                "trend_slope_lookback": 5,
                "atr_window": 14,
                "min_atr_pct": 0.002,
                "max_atr_pct": 0.03,
                "pullback_reclaim_tolerance_pct": 0.001,
                "total_return": 0.03,
                "sharpe": 0.8,
                "profit_factor": 1.2,
                "max_drawdown": -0.04,
                "test_total_return": 0.02,
                "test_sharpe": 0.4,
                "test_profit_factor": 1.1,
                "paper_candidate": True,
                "paper_score": 2.5,
            },
        ]
    )

    ranked = rank_sweep_results(df)

    assert list(ranked["rank"]) == [1, 2]
    assert bool(ranked.iloc[0]["paper_candidate"]) is True
    assert ranked.iloc[0]["test_sharpe"] == 0.4


def test_sample_sweep_configs_spreads_across_grid():
    configs = [
        SweepConfig(
            timeframe_minutes=60,
            short_window=short_window,
            long_window=100,
            trend_filter_window=200,
            pullback_lookback=3,
            trend_slope_lookback=5,
            atr_window=14,
            min_atr_pct=0.003,
            max_atr_pct=0.04,
            pullback_reclaim_tolerance_pct=0.001,
        )
        for short_window in [10, 20, 30, 40, 50]
    ]

    sampled = sample_sweep_configs(configs, 3)

    assert [config.short_window for config in sampled] == [10, 30, 50]


def test_overfitting_diagnostics_on_synthetic_sweep() -> None:
    import numpy as np
    import pandas as pd

    from src.backtest.sweep import overfitting_diagnostics

    rng = np.random.default_rng(11)
    index = pd.date_range("2025-01-02 14:30", periods=1400, freq="h")
    curves = {
        f"cfg{i}": pd.Series(100_000 * np.cumprod(1 + rng.normal(0, 0.002, len(index))), index=index)
        for i in range(12)
    }
    best = max(curves, key=lambda k: curves[k].iloc[-1])
    result = overfitting_diagnostics(curves, best_label=best, bars_per_year=252 * 7, pbo_blocks=6)
    assert result.n_trials == 12
    assert 0.0 <= result.deflated_sharpe <= 1.0
    assert result.deflated_sharpe < 0.95  # noise should not look like skill
    assert result.cpcv["n_paths"] == 5
    assert result.pbo is not None and 0.0 <= result.pbo <= 1.0
