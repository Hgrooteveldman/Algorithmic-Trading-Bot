"""Execution and friction model shared by every backtest engine.

Why this module exists
----------------------
The original engines filled orders at the *close of the bar whose close
produced the signal*, with one flat cost per side. That is look-ahead: the
signal needs the close, so no order based on it can be filled at that same
close. It also ignores that larger orders in thinner bars cost more.

This module fixes both in one place, so all engines share the same rules:

1. **Timing** — a decision taken on bar ``t`` (using data up to and including
   ``t``'s close) is filled at the **open of bar t+1**. Engines implement this
   by reading decision columns through :func:`decision_view` (signals shifted
   one bar) and pricing fills off the current bar's ``open``.
2. **Gaps** — a stop or target that the market gaps through is filled at the
   open, not at the stop level (:func:`stop_fill_reference`).
3. **Friction** — cost per fill = fee tier + half spread + linear market
   impact (a simplified, linear-temporary-impact form of Almgren-Chriss)::

       impact_bps = eta * sigma_bar * (qty / bar_volume) * 10_000

   where ``sigma_bar`` is the bar's volatility as a fraction of price
   (ATR / price is used as a proxy) and participation is capped.

Funding rates are deliberately not modelled: the bot trades US equities via
Alpaca, not crypto perpetuals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import pandas as pd

Side = Literal[1, -1]
"""+1 buys (opening a long or covering a short), -1 sells."""

BPS = 10_000.0


@dataclass(frozen=True)
class FeeTier:
    """Fee in basis points that applies once cumulative traded notional
    reaches ``min_notional``."""

    min_notional: float
    taker_bps: float
    maker_bps: float


@dataclass(frozen=True)
class FeeSchedule:
    """Tiered maker/taker fees. Tiers are matched on cumulative notional
    traded so far in the backtest (a proxy for monthly volume tiers).

    Market and stop orders pay the taker fee; resting limit orders pay the
    maker fee.
    """

    tiers: tuple[FeeTier, ...] = (FeeTier(0.0, 1.0, 1.0),)

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("FeeSchedule needs at least one tier")
        thresholds = [tier.min_notional for tier in self.tiers]
        if thresholds != sorted(thresholds) or thresholds[0] != 0.0:
            raise ValueError("fee tiers must be sorted by min_notional and start at 0")
        for tier in self.tiers:
            if tier.taker_bps < 0 or tier.maker_bps < -BPS:
                raise ValueError("fees must be non-negative (maker rebates allowed)")

    def fee_bps(self, cumulative_notional: float, *, maker: bool = False) -> float:
        tier = self.tiers[0]
        for candidate in self.tiers:
            if cumulative_notional >= candidate.min_notional:
                tier = candidate
        return tier.maker_bps if maker else tier.taker_bps


@dataclass(frozen=True)
class ImpactModel:
    """Half spread plus linear temporary impact.

    Attributes:
        min_half_spread_bps: floor for the half spread when no quote data is
            available (bars carry no bid/ask).
        range_spread_fraction: fraction of the bar's high-low range, relative
            to price, used as a spread proxy when it exceeds the floor.
        eta: impact coefficient (dimensionless). ``eta * sigma *
            participation`` is the price concession as a fraction of price.
        max_participation: participation above this is capped (and the
            engine should not size orders that large in the first place).
    """

    min_half_spread_bps: float = 1.0
    range_spread_fraction: float = 0.05
    eta: float = 1.0
    max_participation: float = 0.10

    def __post_init__(self) -> None:
        if self.min_half_spread_bps < 0 or self.range_spread_fraction < 0:
            raise ValueError("spread parameters must be non-negative")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")

    def half_spread_bps(self, price: float, high: float | None, low: float | None) -> float:
        floor = self.min_half_spread_bps
        if high is None or low is None or price <= 0 or not (high >= low):
            return floor
        range_bps = (high - low) / price * BPS
        return max(floor, self.range_spread_fraction * range_bps)

    def impact_bps(self, qty: float, bar_volume: float | None, sigma_bar: float | None) -> float:
        if qty <= 0 or not bar_volume or bar_volume <= 0 or not sigma_bar or sigma_bar <= 0:
            return 0.0
        participation = min(qty / bar_volume, self.max_participation)
        return self.eta * sigma_bar * participation * BPS


@dataclass(frozen=True)
class FillResult:
    """One simulated fill, with its cost broken down for the audit trail."""

    price: float
    reference_price: float
    fee_bps: float
    half_spread_bps: float
    impact_bps: float

    @property
    def total_cost_bps(self) -> float:
        return self.fee_bps + self.half_spread_bps + self.impact_bps


@dataclass
class ExecutionModel:
    """Prices fills and tracks traded notional for fee tiers."""

    fees: FeeSchedule = field(default_factory=FeeSchedule)
    impact: ImpactModel = field(default_factory=ImpactModel)
    traded_notional: float = 0.0

    @classmethod
    def from_legacy(cls, commission_bps: float, slippage_bps: float, eta: float = 1.0) -> "ExecutionModel":
        """Map the old flat ``commission_bps`` / ``slippage_bps`` arguments:
        commission becomes a single fee tier and slippage the spread floor;
        size-dependent impact is added on top."""
        return cls(
            fees=FeeSchedule((FeeTier(0.0, commission_bps, commission_bps),)),
            impact=ImpactModel(min_half_spread_bps=slippage_bps, eta=eta),
        )

    def fill(
        self,
        side: Side,
        reference_price: float,
        qty: float,
        *,
        high: float | None = None,
        low: float | None = None,
        bar_volume: float | None = None,
        sigma_bar: float | None = None,
        maker: bool = False,
    ) -> FillResult:
        """Price a fill of ``qty`` shares at ``reference_price`` (normally
        the bar's open). Buys pay up, sells receive less."""
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        fee = self.fees.fee_bps(self.traded_notional, maker=maker)
        spread = 0.0 if maker else self.impact.half_spread_bps(reference_price, high, low)
        impact = self.impact.impact_bps(qty, bar_volume, sigma_bar)
        cost = (fee + spread + impact) / BPS
        price = reference_price * (1 + cost) if side == 1 else reference_price * (1 - cost)
        self.traded_notional += abs(qty) * reference_price
        return FillResult(price, reference_price, fee, spread, impact)

    def estimated_cost_fraction(self, price: float, qty: float = 0.0, **bar: float | None) -> float:
        """Expected one-way cost as a fraction of price, for sizing before an
        order exists (no state change)."""
        fee = self.fees.fee_bps(self.traded_notional)
        spread = self.impact.half_spread_bps(price, bar.get("high"), bar.get("low"))
        impact = self.impact.impact_bps(qty, bar.get("bar_volume"), bar.get("sigma_bar"))
        return (fee + spread + impact) / BPS


def stop_fill_reference(side_of_position: Side, open_price: float, level: float, kind: Literal["stop", "target"]) -> float:
    """Reference price for a stop or target hit during a bar.

    A long's stop that the bar *opens* below fills at the open (gap), not at
    the stop; a long's target that the bar opens above fills at the open too
    (a gap in your favour is still only the open). Shorts mirror this.
    """
    if side_of_position == 1:
        if kind == "stop":
            return min(open_price, level)
        return max(open_price, level)
    if kind == "stop":
        return max(open_price, level)
    return min(open_price, level)


def sigma_proxy(atr: float | None, price: float) -> float | None:
    """Bar volatility as a fraction of price, from ATR."""
    if atr is None or price <= 0 or not math.isfinite(atr) or atr <= 0:
        return None
    return atr / price


def decision_view(frame: pd.DataFrame, columns: Iterable[str], prefix: str = "prev_") -> pd.DataFrame:
    """Add ``prev_<col>`` = ``<col>`` shifted one bar.

    Engines read every *decision* (entry setup, exit signal, score, ATR used
    for stops) from these columns while pricing the fill off the current
    bar's open. That makes "decide on t's close, fill at t+1's open" explicit
    and greppable. Boolean columns are filled with False, others left NaN.
    """
    out = frame.copy()
    for column in columns:
        shifted = out[column].shift(1)
        if out[column].dtype == bool:
            shifted = shifted.fillna(False).astype(bool)
        out[f"{prefix}{column}"] = shifted
    return out


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"bars are missing required columns: {missing}")
