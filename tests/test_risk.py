"""Tests for the RiskManager. Run with: pytest tests/

Why focus tests here first?
    - Risk gates are the last line of defense before bad orders go out.
    - Any change to RiskManager should be validated immediately.
"""
import pytest

from config.settings import settings
from src.risk.risk_manager import RiskManager


@pytest.fixture
def rm():
    return RiskManager(starting_equity=100_000)


def test_can_trade_initial_state(rm):
    allowed, _ = rm.can_trade(current_equity=100_000, open_positions=0)
    assert allowed


def test_daily_loss_limit_halts(rm):
    # Drop equity below the daily loss limit
    bad_equity = 100_000 * (1 - settings.max_daily_loss_pct - 0.001)
    allowed, reason = rm.can_trade(current_equity=bad_equity, open_positions=0)
    assert not allowed
    assert "Daily loss" in reason
    # Once halted, stays halted even if equity recovers
    allowed2, _ = rm.can_trade(current_equity=100_000, open_positions=0)
    assert not allowed2


def test_max_open_positions(rm):
    allowed, reason = rm.can_trade(
        current_equity=100_000,
        open_positions=settings.max_open_positions,
    )
    assert not allowed
    assert "Max open positions" in reason


def test_cooldown_after_consecutive_losses(rm):
    for _ in range(settings.cooldown_after_consecutive_losses):
        rm.record_trade_outcome(pnl=-100)
    allowed, reason = rm.can_trade(current_equity=100_000, open_positions=0)
    assert not allowed
    assert "Cooldown" in reason


def test_position_sizing_respects_risk_per_trade(rm):
    qty = rm.calculate_position_size(
        equity=100_000,
        entry_price=100.0,
        stop_loss_price=98.0,
    )
    stop_distance_pct = (100.0 - 98.0) / 100.0
    effective_stop_pct = stop_distance_pct + settings.round_trip_friction_pct
    expected_qty = settings.shares_for_entry(
        equity=100_000,
        entry_price=100.0,
        stop_distance_pct=effective_stop_pct,
    )
    assert qty == expected_qty


def test_position_sizing_zero_when_no_stop_distance(rm):
    qty = rm.calculate_position_size(
        equity=100_000,
        entry_price=100.0,
        stop_loss_price=100.0,
    )
    assert qty == 0


def test_kill_switch_file(rm, tmp_path, monkeypatch):
    kill = tmp_path / "STOP"
    monkeypatch.setattr(settings, "kill_switch_path", kill)
    kill.write_text("halt")
    allowed, reason = rm.can_trade(current_equity=100_000, open_positions=0)
    assert not allowed
    assert "Kill-switch" in reason


def test_weekly_loss_limit_halts(rm):
    rm.state.starting_equity_today = 95_000
    rm.state.starting_equity_week = 100_000
    bad_equity = 100_000 * (1 - settings.max_weekly_loss_pct - 0.001)
    allowed, reason = rm.can_trade(current_equity=bad_equity, open_positions=0)
    assert not allowed
    assert "Weekly loss" in reason


def test_gross_exposure_cap_rejects_oversized_entry(rm):
    exceeds, reason = rm.would_breach_gross_exposure(
        equity=100_000,
        existing_position_notionals=[20_000, 15_000],
        proposed_notional=10_000,
    )
    assert exceeds
    assert "Gross exposure cap exceeded" in reason


def test_circuit_breaker_halts_entries_until_manual_reset(rm):
    rm.enforce_circuit_breaker(current_equity=100_000, max_drawdown_pct=0.25)
    triggered = rm.enforce_circuit_breaker(current_equity=74_999, max_drawdown_pct=0.25)
    assert triggered

    allowed, reason = rm.can_trade(current_equity=90_000, open_positions=0)
    assert not allowed
    assert "circuit breaker" in reason.lower()
