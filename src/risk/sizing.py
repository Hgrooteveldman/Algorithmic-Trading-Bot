"""Dynamic position sizing: volatility targeting under hard caps.

Order of constraints (the smallest quantity wins, and the binding one is
reported for the audit log):

1. **Volatility target** — risk a constant fraction of equity per trade:
   ``qty = equity * risk_per_trade * regime_multiplier / per_share_risk``,
   where ``per_share_risk = max(stop distance, atr_stop_mult * ATR)`` plus
   round-trip friction. High ATR -> fewer shares, so a trade in a wild name
   risks the same euros as one in a calm name.
2. **Fractional Kelly ceiling** — from the bot's own closed trades,
   ``f* = p / a - (1 - p) / b`` (p win rate, a average loss, b average win,
   as fractions of the position), scaled by 0.25-0.5. Used only as a cap:
   with a few dozen trades the edge estimate is too noisy to size *up* on.
   Below ``min_trades`` there is no Kelly cap; a non-positive Kelly (no
   measured edge) blocks new entries.
3. **Per-asset cap** — ``max_position_pct`` of equity.
4. **Per-sector cap** — total notional in one sector at most
   ``max_sector_pct`` of equity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class KellyEstimate:
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    full_kelly: float


def kelly_from_trade_returns(trade_returns: Sequence[float]) -> KellyEstimate | None:
    """Kelly fraction for bets that win ``avg_win`` or lose ``avg_loss`` of
    the stake. Returns None when there are no wins or no losses yet."""
    returns = [float(r) for r in trade_returns if math.isfinite(r)]
    wins = [r for r in returns if r > 0]
    losses = [-r for r in returns if r < 0]
    if not wins or not losses:
        return None
    p = len(wins) / len(returns)
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    full = p / avg_loss - (1 - p) / avg_win
    return KellyEstimate(len(returns), p, avg_win, avg_loss, full)


@dataclass(frozen=True)
class SizingDecision:
    qty: int
    binding: str                    # which constraint set the quantity
    components: dict[str, float]    # every candidate quantity, for the audit trail
    kelly: KellyEstimate | None = None

    @property
    def rejected(self) -> bool:
        return self.qty <= 0


@dataclass
class DynamicSizer:
    risk_per_trade_pct: float = 0.0015
    atr_stop_mult: float = 1.8
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.25
    kelly_fraction: float = 0.25
    min_trades_for_kelly: int = 30
    round_trip_friction_pct: float = 0.0006
    sector_by_symbol: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.25 <= self.kelly_fraction <= 0.5:
            raise ValueError("kelly_fraction must be between 0.25 and 0.5 (conservative fractional Kelly)")
        if not 0 < self.risk_per_trade_pct <= 0.02:
            raise ValueError("risk_per_trade_pct must be in (0, 2%]")
        if not 0 < self.max_position_pct <= 0.25:
            raise ValueError("max_position_pct must be in (0, 25%]")
        if not 0 < self.max_sector_pct <= 1:
            raise ValueError("max_sector_pct must be in (0, 100%]")

    def sector(self, symbol: str) -> str:
        return self.sector_by_symbol.get(symbol, symbol)  # unknown -> its own sector

    def size(
        self,
        *,
        symbol: str,
        equity: float,
        price: float,
        stop_price: float | None,
        atr: float | None,
        regime_multiplier: float = 1.0,
        trade_returns: Sequence[float] = (),
        notional_by_symbol: dict[str, float] | None = None,
    ) -> SizingDecision:
        if equity <= 0 or price <= 0:
            return SizingDecision(0, "invalid_input", {})
        if regime_multiplier <= 0:
            return SizingDecision(0, "regime", {"regime_multiplier": regime_multiplier})

        stop_distance = abs(price - stop_price) if stop_price else 0.0
        atr_distance = self.atr_stop_mult * atr if atr and math.isfinite(atr) and atr > 0 else 0.0
        per_share_risk = max(stop_distance, atr_distance) + price * self.round_trip_friction_pct
        if per_share_risk <= price * self.round_trip_friction_pct:
            return SizingDecision(0, "no_risk_estimate", {})

        components: dict[str, float] = {}
        risk_budget = equity * self.risk_per_trade_pct * min(regime_multiplier, 1.0)
        components["vol_target"] = math.floor(risk_budget / per_share_risk)
        components["asset_cap"] = math.floor(equity * self.max_position_pct / price)

        exposures = notional_by_symbol or {}
        sector = self.sector(symbol)
        sector_used = sum(abs(n) for s, n in exposures.items() if self.sector(s) == sector)
        components["sector_cap"] = math.floor(max(equity * self.max_sector_pct - sector_used, 0.0) / price)

        kelly = None
        if len(trade_returns) >= self.min_trades_for_kelly:
            kelly = kelly_from_trade_returns(trade_returns)
            if kelly is not None:
                kelly_notional = max(kelly.full_kelly, 0.0) * self.kelly_fraction * equity
                components["kelly_cap"] = math.floor(kelly_notional / price)

        binding = min(components, key=lambda k: components[k])
        return SizingDecision(int(max(components[binding], 0)), binding, components, kelly)
