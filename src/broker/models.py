"""Typed models for everything that crosses the broker boundary.

Pydantic v2 validates data once, where it enters the bot, so the rest of the
code can trust it. ``from_attributes=True`` lets a model read the Alpaca SDK
objects directly (no intermediate dict copy), and ``frozen=True`` makes the
validated values immutable.

Bars are validated column-wise with :func:`validate_bars` instead of one
model per row: a year of hourly bars is thousands of rows, and a vectorised
check is both faster and gives one clear log line per problem type.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True, extra="ignore")


class Quote(_Frozen):
    symbol: str
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    timestamp: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> "Quote":
        if self.bid <= 0 and self.ask <= 0:
            raise ValueError("quote has neither bid nor ask")
        if self.bid > 0 and self.ask > 0 and self.ask < self.bid:
            raise ValueError(f"crossed quote: bid {self.bid} > ask {self.ask}")
        return self

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ask or self.bid

    @property
    def spread_bps(self) -> float | None:
        if self.bid > 0 and self.ask > 0:
            return (self.ask - self.bid) / self.mid * 10_000
        return None

    def as_dict(self) -> dict[str, float]:
        """The shape the executor has always consumed."""
        return {"bid": self.bid, "ask": self.ask, "mid": self.mid}

    @classmethod
    def from_alpaca(cls, symbol: str, raw: object) -> "Quote":
        return cls(
            symbol=symbol,
            bid=float(getattr(raw, "bid_price", 0) or 0),
            ask=float(getattr(raw, "ask_price", 0) or 0),
            timestamp=getattr(raw, "timestamp", None),
        )


class Bar(_Frozen):
    symbol: str
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def _envelope(self) -> "Bar":
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close) or self.low > self.high:
            raise ValueError("bar high/low do not envelope open/close")
        return self


class OrderAck(_Frozen):
    """What we keep from an order submission."""

    id: str
    client_order_id: str | None = None
    symbol: str
    side: Side
    qty: float = Field(gt=0)
    status: str
    submitted_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: object) -> object:
        if not isinstance(data, dict):
            data = {name: getattr(data, name, None) for name in cls.model_fields}
        data = dict(data)
        for key in ("id", "client_order_id"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        for key in ("side", "status"):
            value = data.get(key)
            if value is not None and hasattr(value, "value"):
                data[key] = value.value
            if isinstance(data.get(key), str):
                data[key] = data[key].lower()
        return data


class Fill(_Frozen):
    order_id: str
    symbol: str
    side: Side
    qty: float = Field(gt=0)
    price: float = Field(gt=0)
    filled_at: datetime

    @classmethod
    def from_order(cls, order: object) -> "Fill":
        raw_side = getattr(order, "side", None)
        side_text = str(getattr(raw_side, "value", raw_side)).lower()
        return cls(
            order_id=str(getattr(order, "id")),
            symbol=str(getattr(order, "symbol")),
            side=Side(side_text),
            qty=float(getattr(order, "filled_qty", 0) or 0),
            price=float(getattr(order, "filled_avg_price", 0) or 0),
            filled_at=getattr(order, "filled_at"),
        )


BAR_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_bars(bars: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Drop rows that cannot be real bars, sort by time, remove duplicate
    timestamps (keep the last), and log what was dropped and why."""
    if bars.empty:
        return bars
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"[{symbol}] bars missing columns {missing}")
    frame = bars.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    values = frame[list(BAR_COLUMNS)].astype(float)
    checks = {
        "non_finite": ~np.isfinite(values.to_numpy()).all(axis=1),
        "non_positive_price": (values[["open", "high", "low", "close"]] <= 0).any(axis=1).to_numpy(),
        "negative_volume": (values["volume"] < 0).to_numpy(),
        "bad_envelope": (
            (values["high"] < values[["open", "close"]].max(axis=1))
            | (values["low"] > values[["open", "close"]].min(axis=1))
        ).to_numpy(),
    }
    bad = np.zeros(len(frame), dtype=bool)
    for reason, mask in checks.items():
        if mask.any():
            logger.bind(event="bad_bars", symbol=symbol, reason=reason, count=int(mask.sum())).warning(
                f"[{symbol}] dropping {int(mask.sum())} bar(s): {reason}"
            )
        bad |= mask
    return frame[~bad]
