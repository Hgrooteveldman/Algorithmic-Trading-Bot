"""Kill-switch on structural performance degradation.

The daily loss limit and the drawdown circuit breaker catch sudden damage.
They do not catch a strategy that has quietly stopped working: many small
losers, no single bad day. This monitor tracks a rolling window of closed
trade returns and trips when *both* the rolling Sharpe and Sortino (per
trade, not annualised) stay below their floors for ``confirmations``
consecutive checks, once at least ``min_trades`` are in the window.

Tripping writes the operator kill-switch file (``logs/STOP``), the same one a
human would touch, so the halt survives restarts and must be cleared by a
person on purpose.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.monitoring.alerts import AlertSink, alerts as default_alerts


def _rolling_stats(returns: Iterable[float]) -> tuple[float, float]:
    values = [float(r) for r in returns]
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((r - mean) ** 2 for r in values) / (n - 1)
    std = math.sqrt(var)
    downside = math.sqrt(sum(min(r, 0.0) ** 2 for r in values) / n)
    sharpe = mean / std if std > 1e-12 else 0.0
    sortino = mean / downside if downside > 1e-12 else (math.inf if mean > 0 else 0.0)
    return sharpe, sortino


@dataclass
class PerformanceMonitor:
    window: int = 40
    min_trades: int = 25
    sharpe_floor: float = -0.10       # per-trade Sharpe
    sortino_floor: float = -0.15      # per-trade Sortino
    confirmations: int = 3
    kill_switch_path: Path | None = None
    alert_sink: AlertSink = field(default_factory=lambda: default_alerts)
    _returns: deque = field(init=False)
    _breaches: int = field(default=0, init=False)
    tripped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.min_trades > self.window:
            raise ValueError("min_trades cannot exceed window")
        self._returns = deque(maxlen=self.window)

    def record(self, trade_return: float) -> bool:
        """Add one closed trade's return; returns True if this call trips."""
        if math.isfinite(trade_return):
            self._returns.append(float(trade_return))
        return self.check()

    def stats(self) -> dict[str, float | int]:
        sharpe, sortino = _rolling_stats(self._returns)
        return {"n": len(self._returns), "rolling_sharpe": sharpe, "rolling_sortino": sortino,
                "consecutive_breaches": self._breaches}

    def check(self) -> bool:
        if self.tripped or len(self._returns) < self.min_trades:
            return False
        sharpe, sortino = _rolling_stats(self._returns)
        if sharpe < self.sharpe_floor and sortino < self.sortino_floor:
            self._breaches += 1
        else:
            self._breaches = 0
        if self._breaches >= self.confirmations:
            self.tripped = True
            if self.kill_switch_path is not None:
                self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
                self.kill_switch_path.write_text(
                    f"Performance kill-switch: rolling Sharpe {sharpe:.3f}, Sortino {sortino:.3f} "
                    f"over {len(self._returns)} trades. Delete this file to resume.\n",
                    encoding="utf-8",
                )
            self.alert_sink.send(
                "performance_kill_switch",
                f"Strategy degraded: rolling Sharpe {sharpe:.3f} and Sortino {sortino:.3f} "
                f"below floors for {self._breaches} checks. New entries halted.",
                rolling_sharpe=sharpe, rolling_sortino=sortino, n_trades=len(self._returns),
            )
            return True
        return False

    # persistence -----------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"returns": list(self._returns), "breaches": self._breaches, "tripped": self.tripped}

    def restore(self, snapshot: dict | None) -> None:
        if not snapshot:
            return
        self._returns = deque((float(r) for r in snapshot.get("returns", [])), maxlen=self.window)
        self._breaches = int(snapshot.get("breaches", 0))
        self.tripped = bool(snapshot.get("tripped", False))

    @property
    def trade_returns(self) -> list[float]:
        return list(self._returns)
