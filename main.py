"""Live / paper trading orchestrator for the configured strategy path.

USAGE:
    ./.venv/bin/python main.py

SAFETY:
    - Defaults to paper trading. Set PAPER_TRADING=false in .env for live.
    - Touch logs/STOP to halt the bot at the next loop iteration.
    - Review logs daily. Never run unattended without alerts in place.
"""
from __future__ import annotations

import time
from datetime import datetime

from loguru import logger

from config.settings import settings
from src.broker.alpaca_client import AlpacaClient
from src.data.bar_stream import BarStream
from src.data.market_data import close_returns_for_correlation, get_recent_bars
from src.execution.order_executor import OrderExecutor
from src.execution.trade_reconciliation import (
    reconcile_closed_trades,
    snapshot_open_positions,
)
from src.market.session import is_flatten_window, market_session_date
from src.monitoring.logger import setup_logger
from src.regime.detector import regime_multipliers
from src.risk.risk_manager import RiskManager
from src.runtime.state_store import ManagedPositionSnapshot, RuntimeState, RuntimeStateStore
from src.strategy.evidence_momentum import (
    advance_managed_position as advance_evidence_position,
    build_live_config as build_evidence_live_config,
    build_snapshot as build_evidence_snapshot,
    evidence_momentum_lookback_days,
    seed_managed_position as seed_evidence_position,
)
from src.strategy.mean_reversion import (
    advance_managed_position as advance_mean_reversion_position,
    build_live_config as build_mean_reversion_config,
    build_snapshot as build_mean_reversion_snapshot,
    mean_reversion_lookback_days,
    seed_managed_position as seed_mean_reversion_position,
)


def _save_runtime_state(
    state_store: RuntimeStateStore,
    risk: RiskManager,
    tracked_positions,
    managed_positions: dict[str, ManagedPositionSnapshot],
    processed_exit_order_ids: list[str],
    last_reconciled_fill_at: datetime | None,
    flattened_session_date,
    last_bar_timestamps: dict[str, str],
) -> None:
    state_store.save(
        RuntimeState(
            risk=risk.snapshot(),
            tracked_positions=tracked_positions,
            managed_positions=managed_positions,
            processed_exit_order_ids=processed_exit_order_ids,
            last_reconciled_fill_at=(
                last_reconciled_fill_at.isoformat()
                if last_reconciled_fill_at is not None
                else None
            ),
            flattened_session_date=flattened_session_date,
            last_bar_timestamps=last_bar_timestamps,
        )
    )


def main() -> None:
    setup_logger()
    settings.validate()

    mode = "PAPER" if settings.paper_trading else "LIVE"
    logger.info(f"=== Starting bot in {mode} mode ===")
    if not settings.paper_trading:
        logger.warning(
            "*** LIVE TRADING ENABLED. Real money at risk. ***  "
            "Aborting in 5s if Ctrl-C..."
        )
        time.sleep(5)

    strategy_mode = settings.strategy_mode
    regime_kind = "momentum" if strategy_mode == "evidence_momentum" else "mean_reversion"
    if strategy_mode == "evidence_momentum":
        strategy_config = build_evidence_live_config()
        lookback_days = evidence_momentum_lookback_days(strategy_config)
        strategy_label = "evidence-backed momentum"
        proxy_symbol = settings.evidence_market_proxy
        build_snapshot_fn = build_evidence_snapshot
        advance_position_fn = advance_evidence_position
        seed_position_fn = seed_evidence_position
    else:
        strategy_config = build_mean_reversion_config()
        lookback_days = mean_reversion_lookback_days(strategy_config)
        strategy_label = "volatility-adjusted mean reversion"
        proxy_symbol = settings.mean_reversion_market_proxy
        build_snapshot_fn = build_mean_reversion_snapshot
        advance_position_fn = advance_mean_reversion_position
        seed_position_fn = seed_mean_reversion_position

    logger.info(
        "Live strategy: {} | timeframe={}m | "
        "proxy={} | lookback_days={}",
        strategy_label,
        settings.bar_timeframe_minutes,
        proxy_symbol,
        lookback_days,
    )

    broker = AlpacaClient()
    account = broker.get_account()
    logger.info(
        f"Account equity=${float(account.equity):,.2f}, "
        f"buying_power=${float(account.buying_power):,.2f}, "
        f"status={account.status}"
    )

    if not settings.use_leverage and float(account.buying_power) > float(account.equity) * 1.05:
        logger.warning(
            "Buying power > equity (margin available). "
            "Bot will size positions against equity to avoid leverage."
        )

    state_store = RuntimeStateStore(settings.runtime_state_path)
    runtime_state = state_store.load()

    risk = RiskManager(
        starting_equity=float(account.equity),
        atr_stop_mult=strategy_config.atr_stop_mult,
    )
    risk.restore(runtime_state.risk)
    executor = OrderExecutor(broker=broker, risk=risk)

    clock = broker.get_clock()
    last_session_date = market_session_date(clock.timestamp)
    tracked_positions = runtime_state.tracked_positions or snapshot_open_positions(
        broker.get_positions()
    )
    managed_positions = dict(runtime_state.managed_positions)
    processed_exit_order_ids = runtime_state.processed_exit_order_ids
    last_reconciled_fill_at = (
        datetime.fromisoformat(runtime_state.last_reconciled_fill_at)
        if runtime_state.last_reconciled_fill_at
        else None
    )
    flattened_session_date = runtime_state.flattened_session_date
    last_bar_timestamps = dict(runtime_state.last_bar_timestamps)
    configured_symbols = list(dict.fromkeys(settings.symbols))

    # Wake on new bars when the stream is up; otherwise this is a plain sleep.
    bar_stream = BarStream(configured_symbols).start() if settings.use_websocket_market_data else None

    def wait_for_next_cycle(seconds: float) -> None:
        if bar_stream is not None:
            bar_stream.wait_for_bar(seconds)
        else:
            time.sleep(seconds)

    while True:
        try:
            clock = broker.get_clock()
            now = clock.timestamp
            today = market_session_date(now)

            reconciliation = reconcile_closed_trades(
                broker=broker,
                tracked_positions=tracked_positions,
                processed_exit_order_ids=processed_exit_order_ids,
                last_reconciled_fill_at=last_reconciled_fill_at,
                entry_estimates=executor.entry_estimates,
            )
            for index, pnl in enumerate(reconciliation.realized_pnls):
                pnl_pct = (
                    reconciliation.realized_returns[index]
                    if index < len(reconciliation.realized_returns)
                    else None
                )
                risk.record_trade_outcome(pnl, pnl_pct)
            tracked_positions = reconciliation.tracked_positions
            processed_exit_order_ids = reconciliation.processed_exit_order_ids
            last_reconciled_fill_at = reconciliation.last_reconciled_fill_at

            equity = broker.get_equity()
            risk.enforce_circuit_breaker(
                current_equity=equity,
                max_drawdown_pct=strategy_config.circuit_breaker_drawdown_pct,
            )

            calendar_day = broker.get_calendar_day(today)
            if calendar_day is not None and today != last_session_date:
                risk.reset_daily(equity)
                last_session_date = today
                flattened_session_date = None
                logger.info(f"--- New session: {today} ---")

            if not clock.is_open:
                logger.info("Market closed. Sleeping.")
                _save_runtime_state(
                    state_store=state_store,
                    risk=risk,
                    tracked_positions=tracked_positions,
                    managed_positions=managed_positions,
                    processed_exit_order_ids=processed_exit_order_ids,
                    last_reconciled_fill_at=last_reconciled_fill_at,
                    flattened_session_date=flattened_session_date,
                    last_bar_timestamps=last_bar_timestamps,
                )
                time.sleep(settings.poll_interval_seconds * 5)
                continue

            session_close = broker.session_close_for_timestamp(now)
            if settings.flatten_at_eod and session_close is not None and is_flatten_window(
                now,
                flatten_before_close_minutes=settings.flatten_before_close_minutes,
                session_close=session_close,
            ):
                if flattened_session_date != today:
                    positions = broker.get_positions()
                    if positions:
                        logger.warning(
                            "Flatten window reached. Closing all positions before the bell."
                        )
                        broker.close_all_positions()
                    flattened_session_date = today
                logger.info("Flatten window active. Skipping new entries.")
                tracked_positions = snapshot_open_positions(broker.get_positions())
                _save_runtime_state(
                    state_store=state_store,
                    risk=risk,
                    tracked_positions=tracked_positions,
                    managed_positions=managed_positions,
                    processed_exit_order_ids=processed_exit_order_ids,
                    last_reconciled_fill_at=last_reconciled_fill_at,
                    flattened_session_date=flattened_session_date,
                    last_bar_timestamps=last_bar_timestamps,
                )
                time.sleep(settings.poll_interval_seconds)
                continue

            proxy_bars = get_recent_bars(
                broker,
                proxy_symbol,
                lookback_days=lookback_days,
            )
            if proxy_bars.empty:
                logger.warning(
                    "[{}] No proxy bars; skipping this cycle.",
                    proxy_symbol,
                )
                _save_runtime_state(
                    state_store=state_store,
                    risk=risk,
                    tracked_positions=tracked_positions,
                    managed_positions=managed_positions,
                    processed_exit_order_ids=processed_exit_order_ids,
                    last_reconciled_fill_at=last_reconciled_fill_at,
                    flattened_session_date=flattened_session_date,
                    last_bar_timestamps=last_bar_timestamps,
                )
                time.sleep(settings.poll_interval_seconds)
                continue

            live_positions = broker.get_positions()
            live_positions_by_symbol = {
                getattr(position, "symbol", None): position
                for position in live_positions
                if getattr(position, "symbol", None)
            }
            managed_positions = {
                symbol: position
                for symbol, position in managed_positions.items()
                if symbol in live_positions_by_symbol
            }

            bars_cache = {}
            correlation_returns_cache = {}
            snapshots_by_symbol = {}
            for symbol in configured_symbols:
                bars = get_recent_bars(broker, symbol, lookback_days=lookback_days)
                bars_cache[symbol] = bars
                if bars.empty:
                    logger.warning(f"[{symbol}] No bars / illiquid; skipping.")
                    continue
                correlation_returns_cache[symbol] = close_returns_for_correlation(bars)
                snapshot = build_snapshot_fn(
                    bars=bars,
                    proxy_bars=proxy_bars,
                    config=strategy_config,
                )
                if snapshot is None:
                    logger.debug(f"[{symbol}] Insufficient indicator history.")
                    continue
                snapshots_by_symbol[symbol] = snapshot

            for symbol, broker_position in live_positions_by_symbol.items():
                if symbol not in configured_symbols or symbol in managed_positions:
                    continue
                snapshot = snapshots_by_symbol.get(symbol)
                if snapshot is None:
                    continue
                qty = abs(float(getattr(broker_position, "qty", 0) or 0))
                avg_entry_price = float(getattr(broker_position, "avg_entry_price", 0) or 0)
                if qty <= 0 or avg_entry_price <= 0:
                    continue
                managed_positions[symbol] = seed_position_fn(
                    qty=qty,
                    entry_price=avg_entry_price,
                    snapshot=snapshot,
                    config=strategy_config,
                )
                logger.warning(
                    "[{}] Reconstructed {} position state from broker position.",
                    symbol,
                    strategy_label,
                )

            new_bar_snapshots = {
                symbol: snapshot
                for symbol, snapshot in snapshots_by_symbol.items()
                if last_bar_timestamps.get(symbol) != snapshot.timestamp.isoformat()
            }

            closing_symbols: set[str] = set()
            for symbol in configured_symbols:
                if symbol not in live_positions_by_symbol:
                    continue
                snapshot = new_bar_snapshots.get(symbol)
                managed_position = managed_positions.get(symbol)
                if snapshot is None or managed_position is None:
                    continue

                updated_position, exit_reason = advance_position_fn(
                    managed_position,
                    snapshot=snapshot,
                    config=strategy_config,
                )
                managed_positions[symbol] = updated_position

                if exit_reason is None:
                    continue

                logger.info(
                    "[{}] EXIT {} | close=${:.2f} | stop=${:.2f} | target=${:.2f} | "
                    "hold_bars={}",
                    symbol,
                    exit_reason,
                    snapshot.close,
                    updated_position.stop_price,
                    updated_position.target_price,
                    updated_position.holding_bars,
                )
                if executor.close_position(symbol):
                    managed_positions.pop(symbol, None)
                    closing_symbols.add(symbol)

            desired_longs: set[str] = set()
            rebalance_due = False
            if strategy_mode == "evidence_momentum":
                rebalance_due = any(
                    bool(snapshot.rebalance_due)
                    for snapshot in new_bar_snapshots.values()
                )
                if rebalance_due:
                    ranked_longs = [
                        (float(snapshot.score), symbol)
                        for symbol, snapshot in new_bar_snapshots.items()
                        if snapshot.long_entry_setup
                    ]
                    ranked_longs.sort(key=lambda item: (-item[0], item[1]))
                    desired_longs = {
                        symbol
                        for _, symbol in ranked_longs[: strategy_config.top_n]
                    }
                    for symbol in configured_symbols:
                        if symbol in closing_symbols or symbol not in live_positions_by_symbol:
                            continue
                        if symbol in desired_longs:
                            continue
                        snapshot = new_bar_snapshots.get(symbol)
                        managed_position = managed_positions.get(symbol)
                        if snapshot is None or managed_position is None:
                            continue
                        logger.info(
                            "[{}] EXIT RANK | score={:.4f} | close=${:.2f} | stop=${:.2f} | "
                            "target=${:.2f}",
                            symbol,
                            snapshot.score,
                            snapshot.close,
                            managed_position.stop_price,
                            managed_position.target_price,
                        )
                        if executor.close_position(symbol):
                            managed_positions.pop(symbol, None)
                            closing_symbols.add(symbol)

            buy_candidates: list[tuple[float, str]] = []
            for symbol in configured_symbols:
                if symbol in live_positions_by_symbol or symbol in closing_symbols:
                    continue
                snapshot = new_bar_snapshots.get(symbol)
                if snapshot is None:
                    continue

                if strategy_mode == "evidence_momentum":
                    if rebalance_due and symbol in desired_longs and snapshot.long_entry_setup:
                        buy_candidates.append((snapshot.score, symbol))
                elif snapshot.long_entry_setup:
                    buy_candidates.append((snapshot.zscore, symbol))

            if strategy_mode == "evidence_momentum":
                buy_candidates.sort(key=lambda item: (-item[0], item[1]))
            else:
                buy_candidates.sort(key=lambda item: (item[0], item[1]))

            opened_symbols: set[str] = set()
            for _, symbol in buy_candidates:
                if risk.state.circuit_breaker_halted:
                    break
                if symbol in opened_symbols:
                    continue
                snapshot = snapshots_by_symbol.get(symbol)
                bars = bars_cache.get(symbol)
                if snapshot is None or bars is None or bars.empty:
                    continue

                quote = broker.get_latest_quote(symbol)
                if quote is None or float(quote.get("mid", 0) or 0) <= 0:
                    logger.warning(f"[{symbol}] No valid quote; skipping entry.")
                    continue

                entry_reference_price = float(quote["mid"])
                seeded_position = seed_position_fn(
                    qty=1,
                    entry_price=entry_reference_price,
                    snapshot=snapshot,
                    config=strategy_config,
                )
                live_positions = broker.get_positions()
                regime_multiplier = 1.0
                if settings.regime_filter:
                    regime_multiplier = float(
                        regime_multipliers(bars, kind=regime_kind).iloc[-1]
                    )
                    if regime_multiplier <= 0:
                        logger.info("[{}] Regime filter: no entry in this regime.", symbol)
                        continue
                submission = executor.open_long_with_levels(
                    symbol=symbol,
                    stop_loss_price=seeded_position.stop_price,
                    take_profit_price=seeded_position.target_price,
                    bars=bars,
                    positions=live_positions,
                    correlation_returns_cache=correlation_returns_cache,
                    regime_multiplier=regime_multiplier,
                )
                if submission is None:
                    continue

                managed_position = seed_position_fn(
                    qty=submission.qty,
                    entry_price=submission.entry_price,
                    snapshot=snapshot,
                    config=strategy_config,
                )
                managed_positions[symbol] = ManagedPositionSnapshot(
                    qty=managed_position.qty,
                    entry_price=managed_position.entry_price,
                    entry_time=managed_position.entry_time,
                    entry_atr=managed_position.entry_atr,
                    stop_price=submission.stop_price,
                    target_price=submission.take_profit_price,
                    highest_high=managed_position.highest_high,
                    lowest_low=managed_position.lowest_low,
                    holding_bars=managed_position.holding_bars,
                )
                opened_symbols.add(symbol)

            for symbol, snapshot in new_bar_snapshots.items():
                last_bar_timestamps[symbol] = snapshot.timestamp.isoformat()

            tracked_positions = snapshot_open_positions(broker.get_positions())
            equity = broker.get_equity()
            if risk.state.starting_equity_today > 0:
                day_pnl_pct = (
                    (equity - risk.state.starting_equity_today)
                    / risk.state.starting_equity_today
                )
            else:
                day_pnl_pct = 0.0
            logger.info(
                "HEARTBEAT | equity=${:,.2f} | day_pnl={:+.2%} | halted={} | "
                "circuit_breaker={} | managed_positions={}",
                equity,
                day_pnl_pct,
                risk.state.halted,
                risk.state.circuit_breaker_halted,
                len(managed_positions),
            )
            _save_runtime_state(
                state_store=state_store,
                risk=risk,
                tracked_positions=tracked_positions,
                managed_positions=managed_positions,
                processed_exit_order_ids=processed_exit_order_ids,
                last_reconciled_fill_at=last_reconciled_fill_at,
                flattened_session_date=flattened_session_date,
                last_bar_timestamps=last_bar_timestamps,
            )

            wait_for_next_cycle(settings.poll_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Manual interrupt. Shutting down.")
            if bar_stream is not None:
                bar_stream.stop()
            break
        except Exception as e:
            logger.exception(f"Unhandled error in main loop: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
