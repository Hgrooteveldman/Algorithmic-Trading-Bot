"""Persistence for runtime state that must survive restarts."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path


@dataclass(slots=True)
class PositionSnapshot:
    qty: float
    avg_entry_price: float


@dataclass(slots=True)
class ManagedPositionSnapshot:
    qty: float
    entry_price: float
    entry_time: str
    entry_atr: float
    stop_price: float
    target_price: float
    highest_high: float
    lowest_low: float
    holding_bars: int = 0


@dataclass(slots=True)
class RuntimeState:
    risk: dict = field(default_factory=dict)
    tracked_positions: dict[str, PositionSnapshot] = field(default_factory=dict)
    managed_positions: dict[str, ManagedPositionSnapshot] = field(default_factory=dict)
    processed_exit_order_ids: list[str] = field(default_factory=list)
    last_reconciled_fill_at: str | None = None
    flattened_session_date: date | None = None
    last_bar_timestamps: dict[str, str] = field(default_factory=dict)


class RuntimeStateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> RuntimeState:
        if not self.path.exists():
            return RuntimeState()
        payload = json.loads(self.path.read_text())
        tracked_positions = {
            symbol: PositionSnapshot(**data)
            for symbol, data in payload.get("tracked_positions", {}).items()
        }
        managed_positions = {
            symbol: ManagedPositionSnapshot(**data)
            for symbol, data in payload.get("managed_positions", {}).items()
        }
        return RuntimeState(
            risk=payload.get("risk", {}),
            tracked_positions=tracked_positions,
            managed_positions=managed_positions,
            processed_exit_order_ids=payload.get("processed_exit_order_ids", []),
            last_reconciled_fill_at=payload.get("last_reconciled_fill_at"),
            flattened_session_date=self._parse_date(payload.get("flattened_session_date")),
            last_bar_timestamps=payload.get("last_bar_timestamps", {}),
        )

    def save(self, state: RuntimeState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "risk": state.risk,
            "tracked_positions": {
                symbol: asdict(snapshot)
                for symbol, snapshot in state.tracked_positions.items()
            },
            "managed_positions": {
                symbol: asdict(snapshot)
                for symbol, snapshot in state.managed_positions.items()
            },
            "processed_exit_order_ids": state.processed_exit_order_ids[-500:],
            "last_reconciled_fill_at": state.last_reconciled_fill_at,
            "flattened_session_date": (
                state.flattened_session_date.isoformat()
                if state.flattened_session_date is not None
                else None
            ),
            "last_bar_timestamps": state.last_bar_timestamps,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        return date.fromisoformat(value)
