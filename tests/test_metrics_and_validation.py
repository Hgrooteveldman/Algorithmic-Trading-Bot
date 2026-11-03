"""Metrics, CPCV, PBO and the report generator."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdown_stats,
    expected_max_sharpe,
    omega_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)
from src.analytics.report import ValidationSummary, render_html, render_markdown, write_report
from src.validation.cpcv import (
    CombinatorialPurgedKFold,
    cpcv_select_and_evaluate,
    probability_of_backtest_overfitting,
    returns_matrix,
    validate_purge_inputs,
)


def _equity(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2025-01-01", periods=len(values), freq="D"), dtype=float)


# --- metrics --------------------------------------------------------------------

def test_drawdown_depth_duration_and_recovery() -> None:
    equity = _equity([100, 110, 99, 88, 95, 110, 120, 115, 121])
    dd = drawdown_stats(equity)
    assert dd.max_drawdown == pytest.approx(88 / 110 - 1)
    assert dd.peak == equity.index[1]
    assert dd.trough == equity.index[3]
    assert dd.recovery == equity.index[5]       # back at 110
    assert dd.recovery_periods == 2
    assert dd.max_duration_periods == 3         # 99, 88, 95 below 110


def test_unrecovered_drawdown() -> None:
    dd = drawdown_stats(_equity([100, 90, 80, 85]))
    assert dd.recovery is None and dd.recovery_periods is None


def test_sortino_ignores_upside_volatility() -> None:
    calm = pd.Series([0.01, -0.01] * 50)
    spiky_up = pd.Series([0.05, -0.01] * 50)
    assert sortino_ratio(spiky_up, 252) > sortino_ratio(calm, 252)


def test_omega_known_value() -> None:
    assert omega_ratio(pd.Series([0.02, -0.01, 0.01, -0.01])) == pytest.approx(1.5)


def test_calmar_positive_for_rising_curve_with_dip() -> None:
    assert calmar_ratio(_equity(np.linspace(100, 150, 300).tolist() + [140, 160]), 252) > 0


def test_sharpe_zero_for_constant_returns() -> None:
    assert sharpe_ratio(pd.Series([0.001] * 10), 252) == 0.0


def test_psr_higher_for_longer_track_record() -> None:
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0.001, 0.01, 60))
    long = pd.Series(np.tile(short.to_numpy(), 10))
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01) > 0
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_dsr_penalises_many_trials() -> None:
    rng = np.random.default_rng(1)
    winner = pd.Series(rng.normal(0.0008, 0.01, 750))
    trial_sharpes = rng.normal(0.0, 0.03, 500)
    few = deflated_sharpe_ratio(winner, n_trials=2, trial_sharpes_per_period=trial_sharpes[:2])
    many = deflated_sharpe_ratio(winner, n_trials=500, trial_sharpes_per_period=trial_sharpes)
    assert many < few


def test_pure_noise_winner_is_not_significant_after_deflation() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (500, 200))
    sharpes = noise.mean(axis=0) / noise.std(axis=0, ddof=1)
    best = int(np.argmax(sharpes))
    dsr = deflated_sharpe_ratio(pd.Series(noise[:, best]), n_trials=200, trial_sharpes_per_period=sharpes)
    assert dsr < 0.95


def test_summarize_bundles_everything() -> None:
    summary = summarize(_equity(np.linspace(100, 130, 200)), 252, n_trials=10,
                        trial_sharpes_per_period=[0.01, 0.02, 0.05, 0.0])
    assert summary.total_return == pytest.approx(0.3)
    assert summary.dsr is not None
    assert summary.n_periods == 199


# --- CPCV ---------------------------------------------------------------------

def test_cpcv_counts_and_coverage() -> None:
    cv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2)
    splits = list(cv.split(600))
    assert len(splits) == cv.n_splits == 15
    assert cv.n_paths == 5
    for split in splits:
        assert not set(split.train) & set(split.test)


def test_cpcv_purge_and_embargo_remove_neighbours() -> None:
    cv = CombinatorialPurgedKFold(n_groups=5, n_test_groups=1, purge_periods=3, embargo_periods=2)
    split = next(s for s in cv.split(100) if s.test_groups == (2,))  # test rows 40..59
    train = set(split.train)
    assert not {37, 38, 39} & train      # purged before the test group
    assert not {60, 61} & train          # embargoed after it
    assert {36, 62} <= train


def test_cpcv_paths_cover_every_period_once() -> None:
    idx = pd.date_range("2025-01-01", periods=300, freq="D")
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), index=idx, columns=list("abcd"))
    result = cpcv_select_and_evaluate(returns, CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, purge_periods=2))
    assert len(result.path_returns) == 5
    for path in result.path_returns:
        assert len(path) == 300 and path.notna().all()
    assert sum(result.selection_counts.values()) == 15


def test_cpcv_finds_a_real_edge() -> None:
    rng = np.random.default_rng(4)
    idx = pd.date_range("2025-01-01", periods=1000, freq="D")
    noise = rng.normal(0, 0.01, (1000, 5))
    noise[:, 2] += 0.002  # one configuration has genuine skill
    returns = pd.DataFrame(noise, index=idx, columns=[f"cfg{i}" for i in range(5)])
    result = cpcv_select_and_evaluate(returns, CombinatorialPurgedKFold(n_groups=6, n_test_groups=2))
    assert result.summary(252)["most_selected"] == "cfg2"
    assert result.summary(252)["share_paths_positive"] == 1.0


def test_pbo_high_for_noise_low_for_real_edge() -> None:
    # Pure noise: the in-sample winner is a coin flip out of sample, PBO ~ 0.5.
    noise_pbos = [
        probability_of_backtest_overfitting(pd.DataFrame(np.random.default_rng(s).normal(0, 0.01, (800, 30))), n_blocks=8).pbo
        for s in range(12)
    ]
    assert 0.35 < float(np.mean(noise_pbos)) < 0.65
    noise = pd.DataFrame(np.random.default_rng(5).normal(0, 0.01, (800, 30)))
    skilled = noise.copy()
    skilled[7] += 0.003
    assert probability_of_backtest_overfitting(skilled, n_blocks=8).pbo < 0.1


@pytest.mark.parametrize("kwargs", [{"n_groups": 1}, {"n_test_groups": 0}, {"n_groups": 4, "n_test_groups": 4}, {"purge_periods": -1}])
def test_invalid_cv_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(**kwargs)


def test_purge_must_cover_holding_period() -> None:
    with pytest.raises(ValueError):
        validate_purge_inputs(CombinatorialPurgedKFold(purge_periods=2), 600, max_holding=10)
    validate_purge_inputs(CombinatorialPurgedKFold(purge_periods=10), 600, max_holding=[3, 10])


def test_returns_matrix_aligns_and_fills_flat() -> None:
    a = _equity([100, 101, 102])
    b = pd.Series([100.0, 99.0], index=a.index[[0, 2]])
    m = returns_matrix({"a": a, "b": b})
    assert list(m.columns) == ["a", "b"]
    assert m.loc[a.index[1], "b"] == 0.0


# --- report ---------------------------------------------------------------------

def test_report_renders_and_flags_overfitting(tmp_path) -> None:
    equity = _equity(np.linspace(100, 120, 120))
    summary = summarize(equity, 252, n_trials=50, trial_sharpes_per_period=np.linspace(-0.05, 0.05, 50))
    validation = ValidationSummary(n_trials=50, cpcv={"n_paths": 5, "oos_sharpe_mean": -0.1, "oos_sharpe_median": -0.2,
                                                     "oos_sharpe_p10": -1.0, "oos_sharpe_min": -1.5,
                                                     "share_paths_positive": 0.2, "most_selected": "x"}, pbo=0.7)
    md = render_markdown("Test", summary, validation)
    assert "Deflated Sharpe" in md and "Treat with caution" in md
    page = render_html("Test <b>", equity, summary, validation)
    assert "<svg" in page and "Test &lt;b&gt;" in page
    md_path, html_path = write_report(tmp_path, "r", "Test", equity, summary, validation)
    assert md_path.exists() and html_path.exists()
    assert not math.isnan(summary.sortino)
