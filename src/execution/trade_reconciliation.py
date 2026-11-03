"""Reconcile closed trades into realized P&L for live risk controls."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from alpaca.trading.enums import OrderSide, OrderStatus

from src.runtime.state_store import PositionSnapshot


@dataclass(slots=True)
class ReconciliationResult:
    realized_pnls: list[float]
    tracked_positions: dict[str, PositionSnapshot]
    processed_exit_order_ids: list[str]
    last_reconciled_fill_at: datetime | None


def snapshot_open_positions(positions: Iterable) -> dict[str, PositionSnapshot]:
    snapshots = {}
    for position in positions:
        symbol = getattr(position, "symbol", None)
        qty = getattr(position, "qty", None)
        avg_entry_price = getattr(position, "avg_entry_price", None)
        if symbol is None or qty is None or avg_entry_price is None:
            continue
        snapshots[symbol] = PositionSnapshot(
            qty=abs(float(qty)),
            avg_entry_price=float(avg_entry_price),
        )
    return snapshots


def default_reconciliation_start() -> datetime:
    return datetime.now(UTC) - timedelta(days=7)


def reconcile_closed_trades(
    broker,
    tracked_positions: dict[str, PositionSnapshot],
    processed_exit_order_ids: list[str],
    last_reconciled_fill_at: datetime | None,
) -> ReconciliationResult:
    processed_ids = set(processed_exit_order_ids)
    updated_positions = dict(tracked_positions)
    realized_pnls: list[float] = []
    latest_fill_at = last_reconciled_fill_at

    after = last_reconciled_fill_at or default_reconciliation_start()
    closed_orders = broker.get_closed_orders(after=after)

    for order in closed_orders:
        order_id = str(getattr(order, "id", ""))
        if not order_id or order_id in processed_ids:
            continue
        filled_at = getattr(order, "filled_at", None)
        if filled_at is None:
            continue
        side = getattr(order, "side", None)
        status = getattr(order, "status", None)
        if side != OrderSide.SELL or status != OrderStatus.FILLED:
            continue

        symbol = getattr(order, "symbol", None)
        tracked = updated_positions.get(symbol or "")
        filled_qty = abs(float(getattr(order, "filled_qty", 0) or 0))
        fill_price = float(getattr(order, "filled_avg_price", 0) or 0)

        if tracked is not None and filled_qty > 0 and fill_price > 0:
            qty_closed = min(filled_qty, tracked.qty)
            realized_pnls.append(qty_closed * (fill_price - tracked.avg_entry_price))
            remaining_qty = tracked.qty - qty_closed
            if remaining_qty > 1e-9:
                updated_positions[order.symbol] = PositionSnapshot(
                    qty=remaining_qty,
                    avg_entry_price=tracked.avg_entry_price,
                )
            else:
                updated_positions.pop(order.symbol, None)

        processed_ids.add(order_id)
        if latest_fill_at is None or filled_at > latest_fill_at:
            latest_fill_at = filled_at

    return ReconciliationResult(
        realized_pnls=realized_pnls,
        tracked_positions=updated_positions,
        processed_exit_order_ids=sorted(processed_ids),
        last_reconciled_fill_at=latest_fill_at,
    )
