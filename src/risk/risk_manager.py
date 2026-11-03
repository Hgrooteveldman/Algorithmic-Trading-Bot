"""Pre-trade risk gates and position sizing.

This module is the most important safety layer in the system. Every order
flows through `can_trade()` and `calculate_position_size()`. If a check
fails, the order is rejected — *capital preservation > returns*.

The gates are deliberately layered so multiple independent failures must
align to allow a bad trade through:

    1. Manual halt or kill-switch file
    2. Daily loss limit
    3. Cooldown after consecutive losses
    4. Max open positions
    5. Order rate limit
    6. No-trade time window (open / close volatility)
    7. Position-size cap (% of equity)
    8. Risk-per-trade cap (% of equity at risk)

When in doubt, REJECT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Iterable, Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import settings
from src.market.session import is_flatten_window, is_opening_no_trade_window


@dataclass
class RiskState:
    starting_equity_today: float
    starting_equity_week: float
    current_week_key: tuple[int, int]
    peak_equity: float
    consecutive_losses: int = 0
    last_loss_time: Optional[datetime] = None
    orders_this_minute: int = 0
    minute_window_start: Optional[datetime] = None
    halted: bool = False
    circuit_breaker_halted: bool = False


class RiskManager:
    def __init__(self, starting_equity: float):
        week_key = self._week_key(self._utc_now().date())
        self.state = RiskState(
            starting_equity_today=starting_equity,
            starting_equity_week=starting_equity,
            current_week_key=week_key,
            peak_equity=starting_equity,
        )

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def reset_daily(self, equity: float) -> None:
        """Call at session rollover."""
        self.state.starting_equity_today = equity
        self.state.consecutive_losses = 0
        self.state.last_loss_time = None
        self.state.halted = False if not self.state.circuit_breaker_halted else self.state.halted
        logger.info(f"Risk state reset. Starting equity: ${equity:,.2f}")

    def reset_weekly(self, equity: float) -> None:
        self.state.starting_equity_week = equity
        self.state.current_week_key = self._week_key(self._utc_now().date())
        logger.info(f"Weekly risk state reset. Starting equity: ${equity:,.2f}")

    def record_trade_outcome(self, pnl: float) -> None:
        """Call when a position closes so cooldown logic works."""
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_time = self._utc_now()
            logger.warning(
                f"Loss recorded: ${pnl:.2f}. "
                f"Consecutive: {self.state.consecutive_losses}"
            )
        else:
            self.state.consecutive_losses = 0

    def enforce_circuit_breaker(
        self,
        current_equity: float,
        max_drawdown_pct: float,
    ) -> bool:
        """Halt fresh entries after a portfolio-level drawdown from the peak.

        This is a persistent kill switch for the strategy path: once the drawdown
        threshold is breached, the bot stays halted until the runtime state is
        intentionally reset.
        """

        if self.state.circuit_breaker_halted:
            return True
        if current_equity <= 0 or max_drawdown_pct <= 0:
            return False

        self.state.peak_equity = max(self.state.peak_equity, current_equity)
        threshold = self.state.peak_equity * (1 - max_drawdown_pct)
        if current_equity <= threshold:
            self.state.circuit_breaker_halted = True
            logger.error(
                "PORTFOLIO CIRCUIT BREAKER HIT: equity ${:.2f} <= ${:.2f} "
                "({:.2%} drawdown from peak ${:.2f}). Halting new entries.",
                current_equity,
                threshold,
                max_drawdown_pct,
                self.state.peak_equity,
            )
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "starting_equity_today": self.state.starting_equity_today,
            "starting_equity_week": self.state.starting_equity_week,
            "current_week_key": list(self.state.current_week_key),
            "peak_equity": self.state.peak_equity,
            "consecutive_losses": self.state.consecutive_losses,
            "last_loss_time": (
                self.state.last_loss_time.isoformat()
                if self.state.last_loss_time is not None
                else None
            ),
            "orders_this_minute": self.state.orders_this_minute,
            "minute_window_start": (
                self.state.minute_window_start.isoformat()
                if self.state.minute_window_start is not None
                else None
            ),
            "halted": self.state.halted,
            "circuit_breaker_halted": self.state.circuit_breaker_halted,
        }

    def restore(self, snapshot: dict | None) -> None:
        if not snapshot:
            return
        week_key = snapshot.get("current_week_key") or list(self.state.current_week_key)
        self.state.starting_equity_today = float(
            snapshot.get("starting_equity_today", self.state.starting_equity_today)
        )
        self.state.starting_equity_week = float(
            snapshot.get("starting_equity_week", self.state.starting_equity_week)
        )
        self.state.current_week_key = (int(week_key[0]), int(week_key[1]))
        self.state.peak_equity = float(
            snapshot.get("peak_equity", max(self.state.peak_equity, self.state.starting_equity_week))
        )
        self.state.consecutive_losses = int(
            snapshot.get("consecutive_losses", self.state.consecutive_losses)
        )
        self.state.last_loss_time = self._parse_datetime(snapshot.get("last_loss_time"))
        self.state.orders_this_minute = int(
            snapshot.get("orders_this_minute", self.state.orders_this_minute)
        )
        self.state.minute_window_start = self._parse_datetime(
            snapshot.get("minute_window_start")
        )
        self.state.halted = bool(snapshot.get("halted", self.state.halted))
        self.state.circuit_breaker_halted = bool(
            snapshot.get("circuit_breaker_halted", self.state.circuit_breaker_halted)
        )

    def register_order(self) -> None:
        """Call after submitting any order, for rate-limit accounting."""
        self.state.orders_this_minute += 1

    # ------------------------------------------------------------------
    # Pre-trade gates
    # ------------------------------------------------------------------
    def can_trade(
        self,
        current_equity: float,
        open_positions: int,
        open_pnl: float = 0.0,
    ) -> Tuple[bool, str]:
        """Returns (allowed, human-readable reason)."""
        self._roll_session_state_if_needed(current_equity)

        if self.state.circuit_breaker_halted:
            return False, "Portfolio circuit breaker active"
        if self.state.halted:
            return False, "Bot manually halted"

        # 1. Operator kill-switch — touch logs/STOP to halt.
        if settings.kill_switch_path.exists():
            self.state.halted = True
            return False, "Kill-switch file present"

        # 2. Daily loss limit
        equity_for_limits = (
            current_equity
            if settings.include_open_pnl_in_loss_limits
            else current_equity - open_pnl
        )
        if self.state.starting_equity_today > 0:
            daily_pnl_pct = (
                (equity_for_limits - self.state.starting_equity_today)
                / self.state.starting_equity_today
            )
            if daily_pnl_pct <= -settings.max_daily_loss_pct:
                self.state.halted = True
                logger.error(
                    f"DAILY LOSS LIMIT HIT: {daily_pnl_pct:.2%}. Halting for the day."
                )
                return False, f"Daily loss limit reached ({daily_pnl_pct:.2%})"

        if self.state.starting_equity_week > 0:
            weekly_pnl_pct = (
                (equity_for_limits - self.state.starting_equity_week)
                / self.state.starting_equity_week
            )
            if weekly_pnl_pct <= -settings.max_weekly_loss_pct:
                self.state.halted = True
                logger.error(
                    f"WEEKLY LOSS LIMIT HIT: {weekly_pnl_pct:.2%}. Halting for the week."
                )
                return False, f"Weekly loss limit reached ({weekly_pnl_pct:.2%})"

        # 3. Cooldown after consecutive losses
        if (
            self.state.consecutive_losses >= settings.cooldown_after_consecutive_losses
            and self.state.last_loss_time is not None
        ):
            elapsed = self._utc_now() - self.state.last_loss_time
            if elapsed < timedelta(minutes=settings.cooldown_minutes):
                remaining = settings.cooldown_minutes - elapsed.total_seconds() / 60
                return False, f"Cooldown: {remaining:.1f} min remaining"
            # Cooldown expired — reset counter and allow trading again.
            self.state.consecutive_losses = 0

        # 4. Max open positions
        if open_positions >= settings.max_open_positions:
            return False, f"Max open positions reached ({open_positions})"

        # 5. Order rate limit (rolling 1-minute window)
        now = self._utc_now()
        if (
            self.state.minute_window_start is None
            or (now - self.state.minute_window_start) > timedelta(minutes=1)
        ):
            self.state.minute_window_start = now
            self.state.orders_this_minute = 0
        if self.state.orders_this_minute >= settings.max_orders_per_minute:
            return False, "Order rate limit"

        return True, "OK"

    def would_breach_gross_exposure(
        self,
        equity: float,
        existing_position_notionals: Iterable[float],
        proposed_notional: float,
    ) -> Tuple[bool, str]:
        if equity <= 0:
            return True, "Equity must be positive"
        gross_exposure = sum(max(float(notional), 0.0) for notional in existing_position_notionals)
        gross_exposure += max(proposed_notional, 0.0)
        gross_exposure_pct = gross_exposure / equity
        if gross_exposure_pct > settings.max_gross_exposure_pct:
            return (
                True,
                "Gross exposure cap exceeded "
                f"({gross_exposure_pct:.2%} > {settings.max_gross_exposure_pct:.2%})",
            )
        return False, "OK"

    def would_breach_correlation_cap(
        self,
        candidate_symbol: str,
        candidate_returns: pd.Series,
        existing_returns: dict[str, pd.Series],
    ) -> Tuple[bool, str]:
        candidate = candidate_returns.dropna().tail(settings.correlation_lookback_bars)
        if candidate.empty:
            return False, "OK"

        minimum_overlap = max(10, settings.correlation_lookback_bars // 2)
        high_correlation_matches = []
        for symbol, returns in existing_returns.items():
            aligned = pd.concat(
                [
                    candidate.rename(candidate_symbol),
                    returns.dropna().tail(settings.correlation_lookback_bars).rename(symbol),
                ],
                axis=1,
            ).dropna()
            if len(aligned) < minimum_overlap:
                continue

            correlation = float(aligned.corr().iloc[0, 1])
            if correlation >= settings.max_pairwise_correlation:
                high_correlation_matches.append(f"{symbol} ({correlation:.2f})")

        if len(high_correlation_matches) >= settings.max_high_correlation_positions:
            return (
                True,
                "Correlation cap exceeded against "
                + ", ".join(high_correlation_matches),
            )
        return False, "OK"

    def in_no_trade_window(self) -> bool:
        """Avoid the first and last few minutes of the regular session.

        Spreads are wide and volatility is high, which inflates slippage.
        Uses DST-aware New York session time instead of hard-coded UTC.
        """
        now = self._utc_now()
        if is_opening_no_trade_window(now, minutes=5):
            return True
        close_window_minutes = (
            settings.flatten_before_close_minutes
            if settings.flatten_at_eod
            else 5
        )
        if is_flatten_window(now, flatten_before_close_minutes=close_window_minutes):
            return True
        return False

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        """Compute integer share quantity using two independent caps:

        1. Risk-based target notional from the configured loss assumptions
        2. Position-cap target notional from the single-position ceiling

        We take the MINIMUM, so whichever is more conservative wins.
        Returns 0 if any input is invalid.
        """
        if equity <= 0 or entry_price <= 0:
            return 0
        per_share_risk = abs(entry_price - stop_loss_price)
        if per_share_risk <= 0:
            return 0

        stop_distance_pct = per_share_risk / entry_price
        effective_stop_pct = stop_distance_pct + settings.round_trip_friction_pct

        # Risk-based size, aligned with configured friction and overnight gap assumptions.
        target_notional = settings.target_position_notional(
            equity=equity,
            stop_distance_pct=(
                effective_stop_pct
                if settings.flatten_at_eod
                else effective_stop_pct * settings.gap_risk_multiplier_overnight
            ),
        )
        qty_by_risk = math.floor(target_notional / entry_price)

        # Position-cap size remains an independent hard ceiling.
        qty_by_size = settings.shares_for_entry(
            equity=equity,
            entry_price=entry_price,
            stop_distance_pct=effective_stop_pct,
        )

        return max(0, min(qty_by_risk, qty_by_size))

    def _roll_session_state_if_needed(self, equity: float) -> None:
        today = self._utc_now().date()
        if self._week_key(today) != self.state.current_week_key:
            self.reset_weekly(equity)

    @staticmethod
    def _week_key(current_date: date) -> tuple[int, int]:
        iso = current_date.isocalendar()
        return (iso.year, iso.week)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)
