"""Thin wrapper around the alpaca-py SDK.

Why wrap it?
    - The rest of the codebase shouldn't know whether we use Alpaca, IBKR,
      or a mock. Swapping brokers should mean reimplementing this one class.
    - Centralized error handling + retries belong here, not scattered in
      strategy code.
    - Tests can substitute a fake AlpacaClient without monkey-patching the SDK.

NOTE: alpaca-py changes its API occasionally. If something here breaks,
check https://alpaca.markets/sdks/python/ for the current signatures.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Optional, List

import pandas as pd
from loguru import logger

from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from config.settings import settings
from src.market.session import attach_market_timezone


def _to_alpaca_timeframe(minutes: int) -> TimeFrame:
    """Map a minute count to Alpaca's TimeFrame. The Minute unit only accepts
    1–59; 60 must be expressed as 1 Hour."""
    if minutes == 60:
        return TimeFrame(amount=1, unit=TimeFrameUnit.Hour)
    if minutes >= 1440 and minutes % 1440 == 0:
        return TimeFrame(amount=minutes // 1440, unit=TimeFrameUnit.Day)
    return TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute)


class AlpacaClient:
    """Unified gateway to Alpaca's trading + market-data APIs."""

    def __init__(self) -> None:
        settings.validate()
        # `paper=True` routes to the paper endpoint automatically.
        self.trading = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.paper_trading,
        )
        # Market data endpoints are the same for paper and live.
        self.data = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        mode = "PAPER" if settings.paper_trading else "LIVE"
        logger.info(f"AlpacaClient initialised in {mode} mode")

    # ----- Account ---------------------------------------------------
    def get_account(self):
        return self.trading.get_account()

    def get_equity(self) -> float:
        return float(self.get_account().equity)

    def get_buying_power(self) -> float:
        return float(self.get_account().buying_power)

    def is_market_open(self) -> bool:
        try:
            return self.trading.get_clock().is_open
        except Exception as e:
            logger.warning(f"Clock check failed: {e}; assuming closed.")
            return False

    def get_clock(self):
        return self.trading.get_clock()

    def get_closed_orders(self, after: datetime, limit: int = 200):
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=after,
            limit=limit,
            direction=Sort.ASC,
            nested=True,
        )
        return self.trading.get_orders(filter=request)

    def get_calendar_day(self, trading_day: date):
        request = GetCalendarRequest(start=trading_day, end=trading_day)
        days = self.trading.get_calendar(filters=request)
        if not days:
            return None
        return days[0]

    # ----- Positions -------------------------------------------------
    def get_positions(self) -> List:
        return self.trading.get_all_positions()

    def get_position(self, symbol: str):
        """Returns the open position for `symbol`, or None if flat."""
        try:
            return self.trading.get_open_position(symbol)
        except Exception:
            # alpaca-py raises when no position exists; treat as flat.
            return None

    # ----- Market data ----------------------------------------------
    def get_bars(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch recent OHLCV bars for the configured timeframe.

        The SDK returns a multi-index DataFrame; we flatten for convenience.
        """
        end = datetime.now(UTC) - timedelta(minutes=20)
        start = end - timedelta(days=lookback_days)
        timeframe = _to_alpaca_timeframe(settings.bar_timeframe_minutes)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as e:
            logger.exception(f"Bar fetch failed for {symbol}: {e}")
            return pd.DataFrame()
        if bars is None or bars.empty:
            return pd.DataFrame()
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)
        return bars

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Mid price from the latest quote, or None if unavailable."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data.get_stock_latest_quote(req)[symbol]
            ask, bid = float(quote.ask_price or 0), float(quote.bid_price or 0)
            if ask > 0 and bid > 0:
                return (ask + bid) / 2
            return ask or bid or None
        except Exception as e:
            logger.warning(f"Quote fetch failed for {symbol}: {e}")
            return None

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Return the latest quote in a simple, test-friendly shape."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data.get_stock_latest_quote(req)[symbol]
            bid = float(quote.bid_price or 0)
            ask = float(quote.ask_price or 0)
            mid = (ask + bid) / 2 if ask > 0 and bid > 0 else (ask or bid or 0.0)
            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
            }
        except Exception as e:
            logger.warning(f"Quote fetch failed for {symbol}: {e}")
            return None

    # ----- Orders ----------------------------------------------------
    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        stop_loss_price: float,
        take_profit_price: float,
    ):
        """Market entry with stop-loss + take-profit attached.

        A bracket order means Alpaca manages the exits server-side, so even if
        our bot dies, the broker still closes the trade at SL or TP. This is
        a key safeguard — never run unattended without bracket protection.
        """
        stop_loss = StopLossRequest(stop_price=stop_loss_price)
        if settings.stop_order_type == "stop_limit":
            stop_loss = StopLossRequest(
                stop_price=stop_loss_price,
                limit_price=round(stop_loss_price * 0.995, 2),
            )
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=stop_loss,
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
        )
        return self.trading.submit_order(order)

    def submit_entry_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        stop_loss_price: float,
        take_profit_price: float,
        use_bracket_order: bool | None = None,
    ):
        if use_bracket_order is None:
            use_bracket_order = settings.use_bracket_orders
        if use_bracket_order:
            return self.submit_bracket_order(
                symbol=symbol,
                qty=qty,
                side=side,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            extended_hours=settings.allow_extended_hours,
        )
        return self.trading.submit_order(order)

    def close_position(self, symbol: str):
        return self.trading.close_position(symbol)

    def cancel_all_orders(self):
        return self.trading.cancel_orders()

    def close_all_positions(self):
        """Emergency flatten — cancels open orders and closes every position."""
        return self.trading.close_all_positions(cancel_orders=True)

    def session_close_for_timestamp(self, timestamp: datetime) -> datetime | None:
        calendar_day = self.get_calendar_day(attach_market_timezone(timestamp).date())
        if calendar_day is None:
            return None
        return attach_market_timezone(calendar_day.close).astimezone(UTC)
