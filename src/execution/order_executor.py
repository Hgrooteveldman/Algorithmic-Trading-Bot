"""Translates trading signals into actual broker orders.

Flow:
    Signal -> risk gates -> position sizing -> bracket order -> log

The executor never makes risk decisions itself. It only translates an
intent into a broker call after the RiskManager has approved it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger
from alpaca.trading.enums import OrderSide

from src.broker.alpaca_client import AlpacaClient
from src.data.market_data import close_returns_for_correlation, correlation_lookback_days
from src.risk.risk_manager import RiskManager
from src.strategy.base import Signal
from config.settings import settings


@dataclass(frozen=True, slots=True)
class EntrySubmission:
    symbol: str
    qty: int
    entry_price: float
    stop_price: float
    take_profit_price: float


class OrderExecutor:
    def __init__(self, broker: AlpacaClient, risk: RiskManager):
        self.broker = broker
        self.risk = risk

    def execute(
        self,
        symbol: str,
        signal: Signal,
        bars: pd.DataFrame | None = None,
        positions=None,
        correlation_returns_cache: dict[str, pd.Series] | None = None,
    ) -> None:
        if signal == Signal.HOLD:
            return

        positions = positions if positions is not None else self.broker.get_positions()
        position_for_symbol = self.broker.get_position(symbol)

        if signal == Signal.BUY:
            if position_for_symbol is not None:
                logger.info(f"[{symbol}] Already long; ignoring BUY.")
                return
            equity = self.broker.get_equity()
            open_pnl = sum(
                float(getattr(position, "unrealized_pl", 0) or 0)
                for position in positions
            )
            allowed, reason = self.risk.can_trade(
                equity,
                len(positions),
                open_pnl=open_pnl,
            )
            if not allowed:
                logger.warning(f"[{symbol}] Trade rejected: {reason}")
                return
            if self.risk.in_no_trade_window():
                logger.info(f"[{symbol}] In no-trade window; skipping {signal.value}.")
                return
            self._open_long(
                symbol,
                equity,
                positions,
                bars,
                correlation_returns_cache=correlation_returns_cache,
            )
        elif signal == Signal.SELL:
            if position_for_symbol is None:
                logger.info(f"[{symbol}] No position to close.")
                return
            self._close_position(symbol)

    def close_position(self, symbol: str) -> bool:
        return self._close_position(symbol)

    def open_long_with_levels(
        self,
        symbol: str,
        stop_loss_price: float,
        take_profit_price: float,
        bars: pd.DataFrame | None = None,
        positions=None,
        correlation_returns_cache: dict[str, pd.Series] | None = None,
        use_bracket_order: bool | None = None,
    ) -> EntrySubmission | None:
        equity = self.broker.get_equity()
        positions = positions if positions is not None else self.broker.get_positions()
        open_pnl = sum(
            float(getattr(position, "unrealized_pl", 0) or 0)
            for position in positions
        )

        allowed, reason = self.risk.can_trade(equity, len(positions), open_pnl=open_pnl)
        if not allowed:
            logger.warning(f"[{symbol}] Trade rejected: {reason}")
            return None
        if self.risk.in_no_trade_window():
            logger.info(f"[{symbol}] In no-trade window; skipping BUY.")
            return None
        if self.broker.get_position(symbol) is not None:
            logger.info(f"[{symbol}] Already long; ignoring BUY.")
            return None

        quote = self._validated_entry_quote(symbol)
        if quote is None:
            return None
        entry_price, _, _ = quote
        return self._submit_long_order(
            symbol=symbol,
            equity=equity,
            positions=positions,
            bars=bars,
            correlation_returns_cache=correlation_returns_cache,
            entry_price=entry_price,
            stop_price=stop_loss_price,
            take_price=take_profit_price,
            use_bracket_order=use_bracket_order,
        )

    # ------------------------------------------------------------------
    def _open_long(
        self,
        symbol: str,
        equity: float,
        positions,
        bars: pd.DataFrame | None,
        correlation_returns_cache: dict[str, pd.Series] | None,
    ) -> EntrySubmission | None:
        quote = self._validated_entry_quote(symbol)
        if quote is None:
            return None
        entry_price, _, _ = quote
        stop_price = entry_price * (1 - settings.stop_loss_pct)
        take_price = entry_price * (1 + settings.take_profit_pct)
        return self._submit_long_order(
            symbol=symbol,
            equity=equity,
            positions=positions,
            bars=bars,
            correlation_returns_cache=correlation_returns_cache,
            entry_price=entry_price,
            stop_price=stop_price,
            take_price=take_price,
            use_bracket_order=None,
        )

    def _validated_entry_quote(self, symbol: str) -> tuple[float, float, float] | None:
        quote = self.broker.get_latest_quote(symbol)
        if quote is None:
            logger.warning(f"[{symbol}] No valid quote; skipping entry.")
            return None

        entry_price = float(quote["mid"])
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        if entry_price <= 0:
            logger.warning(f"[{symbol}] No valid mid price; skipping entry.")
            return None
        if entry_price < settings.min_price:
            logger.info(
                f"[{symbol}] Price ${entry_price:.2f} below minimum ${settings.min_price:.2f}."
            )
            return None
        if bid > 0 and ask > 0:
            spread_bps = ((ask - bid) / entry_price) * 10_000
            if spread_bps > settings.max_bid_ask_spread_bps:
                logger.info(
                    f"[{symbol}] Spread too wide ({spread_bps:.1f} bps); skipping."
                )
                return None
        return entry_price, bid, ask

    def _submit_long_order(
        self,
        symbol: str,
        equity: float,
        positions,
        bars: pd.DataFrame | None,
        correlation_returns_cache: dict[str, pd.Series] | None,
        entry_price: float,
        stop_price: float,
        take_price: float,
        use_bracket_order: bool | None,
    ) -> EntrySubmission | None:
        stop_price = round(float(stop_price), 2)
        take_price = round(float(take_price), 2)
        if stop_price <= 0 or stop_price >= entry_price:
            logger.warning(
                f"[{symbol}] Invalid stop ${stop_price:.2f} for entry ${entry_price:.2f}; skipping."
            )
            return None
        if take_price <= entry_price:
            logger.warning(
                f"[{symbol}] Invalid target ${take_price:.2f} for entry ${entry_price:.2f}; skipping."
            )
            return None

        qty = self.risk.calculate_position_size(equity, entry_price, stop_price)
        if qty <= 0:
            logger.warning(f"[{symbol}] Position size = 0; skipping.")
            return None

        proposed_notional = qty * entry_price
        current_notionals = []
        for position in positions:
            market_value = getattr(position, "market_value", None)
            current_price = getattr(position, "current_price", None)
            position_qty = getattr(position, "qty", None)
            if market_value is not None:
                current_notionals.append(abs(float(market_value)))
            elif current_price is not None and position_qty is not None:
                current_notionals.append(abs(float(current_price) * float(position_qty)))
        exceeds_gross, exposure_reason = self.risk.would_breach_gross_exposure(
            equity=equity,
            existing_position_notionals=current_notionals,
            proposed_notional=proposed_notional,
        )
        if exceeds_gross:
            logger.warning(f"[{symbol}] Trade rejected: {exposure_reason}")
            return None

        if bars is not None and not bars.empty:
            candidate_returns = close_returns_for_correlation(bars)
            existing_returns = {}
            for position in positions:
                existing_symbol = getattr(position, "symbol", None)
                if not existing_symbol:
                    continue
                cached_returns = None
                if correlation_returns_cache is not None:
                    cached_returns = correlation_returns_cache.get(existing_symbol)
                if cached_returns is None:
                    position_bars = self.broker.get_bars(
                        existing_symbol,
                        lookback_days=correlation_lookback_days(),
                    )
                    cached_returns = close_returns_for_correlation(position_bars)
                    if correlation_returns_cache is not None and not cached_returns.empty:
                        correlation_returns_cache[existing_symbol] = cached_returns
                if cached_returns is not None and not cached_returns.empty:
                    existing_returns[existing_symbol] = cached_returns

            exceeds_corr, corr_reason = self.risk.would_breach_correlation_cap(
                candidate_symbol=symbol,
                candidate_returns=candidate_returns,
                existing_returns=existing_returns,
            )
            if exceeds_corr:
                logger.warning(f"[{symbol}] Trade rejected: {corr_reason}")
                return None

        # Pre-flight: ensure we are not implicitly using leverage.
        if not settings.use_leverage:
            cost = proposed_notional
            buying_power = self.broker.get_buying_power()
            if cost > equity:
                logger.warning(
                    f"[{symbol}] Order cost ${cost:,.2f} > equity ${equity:,.2f}. "
                    f"Leverage disabled; reducing qty."
                )
                qty = int(equity // entry_price)
                if qty <= 0:
                    return None
                proposed_notional = qty * entry_price
                cost = proposed_notional
            if cost > buying_power:
                logger.warning(f"[{symbol}] Insufficient buying power.")
                return None

        logger.info(
            f"[{symbol}] LONG {qty} @ ~${entry_price:.2f} "
            f"(SL=${stop_price:.2f}, TP=${take_price:.2f})"
        )
        try:
            order = self.broker.submit_entry_order(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                stop_loss_price=stop_price,
                take_profit_price=take_price,
                use_bracket_order=use_bracket_order,
            )
            self.risk.register_order()
            logger.info(f"[{symbol}] Order submitted, id={order.id}")
            return EntrySubmission(
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_price,
            )
        except Exception as e:
            logger.exception(f"[{symbol}] Order submission failed: {e}")
            return None

    def _close_position(self, symbol: str) -> bool:
        try:
            self.broker.close_position(symbol)
            self.risk.register_order()
            logger.info(f"[{symbol}] Close submitted.")
            return True
        except Exception as e:
            logger.exception(f"[{symbol}] Close failed: {e}")
            return False
