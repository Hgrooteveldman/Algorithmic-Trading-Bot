"""Execution model and no-look-ahead guarantees for every backtest engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import backtest_portfolio
from src.backtest.evidence_momentum import EvidenceMomentumConfig, backtest_evidence_momentum_portfolio
from src.backtest.execution_model import (
    ExecutionModel,
    FeeSchedule,
    FeeTier,
    ImpactModel,
    decision_view,
    stop_fill_reference,
)
from src.backtest.volatility_mean_reversion import MeanReversionConfig, backtest_mean_reversion_portfolio


# --- friction -----------------------------------------------------------------

def test_buy_pays_up_and_sell_receives_less() -> None:
    model = ExecutionModel(FeeSchedule((FeeTier(0.0, 2.0, 0.0),)), ImpactModel(min_half_spread_bps=1.0, eta=0.0))
    buy = model.fill(1, 100.0, 10)
    sell = model.fill(-1, 100.0, 10)
    assert buy.price == pytest.approx(100.0 * (1 + 3 / 10_000))
    assert sell.price == pytest.approx(100.0 * (1 - 3 / 10_000))


def test_impact_grows_linearly_with_participation() -> None:
    impact = ImpactModel(min_half_spread_bps=0.0, eta=1.0, max_participation=1.0)
    small = impact.impact_bps(qty=1_000, bar_volume=100_000, sigma_bar=0.01)
    large = impact.impact_bps(qty=2_000, bar_volume=100_000, sigma_bar=0.01)
    assert small == pytest.approx(1.0)  # 1.0 * 1% * 1% * 10_000
    assert large == pytest.approx(2 * small)


def test_impact_participation_is_capped() -> None:
    impact = ImpactModel(eta=1.0, max_participation=0.05)
    assert impact.impact_bps(qty=10**9, bar_volume=1_000, sigma_bar=0.01) == pytest.approx(0.05 * 0.01 * 10_000)


def test_missing_volume_means_no_impact() -> None:
    assert ImpactModel().impact_bps(qty=100, bar_volume=None, sigma_bar=0.01) == 0.0


def test_spread_uses_range_proxy_above_floor() -> None:
    impact = ImpactModel(min_half_spread_bps=1.0, range_spread_fraction=0.1)
    # 2% range -> 200 bps * 0.1 = 20 bps, above the 1 bp floor
    assert impact.half_spread_bps(100.0, high=101.0, low=99.0) == pytest.approx(20.0)
    assert impact.half_spread_bps(100.0, high=None, low=None) == 1.0


def test_fee_tiers_switch_on_cumulative_notional() -> None:
    fees = FeeSchedule((FeeTier(0.0, 5.0, 2.0), FeeTier(1_000_000.0, 2.0, 0.0)))
    model = ExecutionModel(fees, ImpactModel(min_half_spread_bps=0.0, eta=0.0))
    first = model.fill(1, 100.0, 10_000)  # 1M notional traded at tier 0
    second = model.fill(1, 100.0, 1)
    assert first.fee_bps == 5.0
    assert second.fee_bps == 2.0
    assert fees.fee_bps(0.0, maker=True) == 2.0


def test_maker_fills_pay_no_spread() -> None:
    model = ExecutionModel(FeeSchedule((FeeTier(0.0, 3.0, 1.0),)), ImpactModel(min_half_spread_bps=5.0, eta=0.0))
    fill = model.fill(1, 50.0, 1, maker=True)
    assert fill.half_spread_bps == 0.0
    assert fill.fee_bps == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiers": ()},
        {"tiers": (FeeTier(10.0, 1.0, 1.0),)},
        {"tiers": (FeeTier(0.0, 1.0, 1.0), FeeTier(5.0, 1.0, 1.0), FeeTier(1.0, 1.0, 1.0))},
    ],
)
def test_invalid_fee_schedules_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        FeeSchedule(**kwargs)


def test_invalid_fill_inputs_rejected() -> None:
    model = ExecutionModel()
    with pytest.raises(ValueError):
        model.fill(0, 100.0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        model.fill(1, 0.0, 1)


# --- gaps and timing ------------------------------------------------------------

def test_long_stop_gapped_through_fills_at_open() -> None:
    assert stop_fill_reference(1, open_price=95.0, level=98.0, kind="stop") == 95.0
    assert stop_fill_reference(1, open_price=100.0, level=98.0, kind="stop") == 98.0


def test_short_stop_gapped_through_fills_at_open() -> None:
    assert stop_fill_reference(-1, open_price=105.0, level=102.0, kind="stop") == 105.0
    assert stop_fill_reference(-1, open_price=100.0, level=102.0, kind="target") == 100.0


def test_decision_view_shifts_one_bar() -> None:
    frame = pd.DataFrame({"sig": [False, True, False], "score": [1.0, 2.0, 3.0]})
    view = decision_view(frame, ["sig", "score"])
    assert view["prev_sig"].tolist() == [False, False, True]
    assert view["prev_score"].iloc[0] != view["prev_score"].iloc[0]  # NaN
    assert view["prev_score"].iloc[2] == 2.0


def _assert_entries_reference_entry_bar_open(result, bars_by_symbol, max_cost_bps: float = 60.0) -> None:
    """Every entry must be priced off the open of its own entry bar, never
    off the close of the bar that produced the signal."""
    trades = result.trades
    assert not trades.empty, "fixture should produce trades"
    for trade in trades.itertuples():
        bars = bars_by_symbol[trade.symbol]
        entry_bar = bars.loc[pd.Timestamp(trade.entry_time)]
        open_price = float(entry_bar["open"])
        side = getattr(trade, "side", "LONG")
        if side == "LONG":
            assert open_price <= trade.entry <= open_price * (1 + max_cost_bps / 10_000)
        else:
            assert open_price * (1 - max_cost_bps / 10_000) <= trade.entry <= open_price
        assert trade.entry_time != trade.exit_time or trade.reason in {"STOP", "TARGET"}


def test_trend_engine_fills_at_next_open(synthetic_universe) -> None:
    result = backtest_portfolio(synthetic_universe, bars_per_year=252 * 7)
    _assert_entries_reference_entry_bar_open(result, synthetic_universe)


def test_momentum_engine_fills_at_next_open(synthetic_universe, synthetic_proxy) -> None:
    result = backtest_evidence_momentum_portfolio(
        synthetic_universe, synthetic_proxy, EvidenceMomentumConfig(), timeframe_minutes=60
    )
    _assert_entries_reference_entry_bar_open(result, synthetic_universe)


def test_mean_reversion_engine_fills_at_next_open(synthetic_universe, synthetic_proxy) -> None:
    result = backtest_mean_reversion_portfolio(synthetic_universe, synthetic_proxy, MeanReversionConfig())
    _assert_entries_reference_entry_bar_open(result, synthetic_universe)


def test_future_bars_cannot_change_past_decisions(synthetic_universe, synthetic_proxy) -> None:
    """Truncation test: trades closed before the cut must be identical
    whether or not later bars exist. Any look-ahead breaks this."""
    full = backtest_mean_reversion_portfolio(synthetic_universe, synthetic_proxy, MeanReversionConfig())
    cut = synthetic_universe["AAA"].index[len(synthetic_universe["AAA"]) // 2]
    truncated_universe = {s: b.loc[:cut] for s, b in synthetic_universe.items()}
    truncated = backtest_mean_reversion_portfolio(truncated_universe, synthetic_proxy.loc[:cut], MeanReversionConfig())

    def closed_before(trades: pd.DataFrame) -> pd.DataFrame:
        done = trades[(pd.to_datetime(trades["exit_time"]) < cut) & (trades["reason"] != "END")]
        return done[["symbol", "entry_time", "exit_time", "qty"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(closed_before(full.trades), closed_before(truncated.trades))


def test_higher_impact_never_improves_results(synthetic_universe) -> None:
    cheap = backtest_portfolio(synthetic_universe, bars_per_year=252 * 7,
                               execution=ExecutionModel(impact=ImpactModel(eta=0.0)))
    costly = backtest_portfolio(synthetic_universe, bars_per_year=252 * 7,
                                execution=ExecutionModel(impact=ImpactModel(eta=50.0, max_participation=1.0)))
    assert costly.total_return <= cheap.total_return
    assert np.isfinite(costly.sharpe)
