from datetime import date

from src.runtime.state_store import (
    ManagedPositionSnapshot,
    PositionSnapshot,
    RuntimeState,
    RuntimeStateStore,
)


def test_runtime_state_store_round_trips_session_date(tmp_path):
    path = tmp_path / "runtime_state.json"
    store = RuntimeStateStore(path)
    state = RuntimeState(
        risk={"halted": False},
        tracked_positions={"QQQ": PositionSnapshot(qty=5.0, avg_entry_price=100.0)},
        managed_positions={
            "QQQ": ManagedPositionSnapshot(
                qty=5.0,
                entry_price=100.0,
                entry_time="2026-04-28T19:00:00+00:00",
                entry_atr=1.5,
                stop_price=97.3,
                target_price=102.2,
                highest_high=101.0,
                lowest_low=99.4,
                holding_bars=3,
            )
        },
        processed_exit_order_ids=["order-1"],
        last_reconciled_fill_at="2026-04-28T19:00:00+00:00",
        flattened_session_date=date(2026, 4, 28),
        last_bar_timestamps={"QQQ": "2026-04-28T19:00:00+00:00"},
    )

    store.save(state)
    loaded = store.load()

    assert loaded.flattened_session_date == date(2026, 4, 28)
    assert loaded.tracked_positions["QQQ"].qty == 5.0
    assert loaded.managed_positions["QQQ"].stop_price == 97.3
    assert loaded.processed_exit_order_ids == ["order-1"]
    assert loaded.last_bar_timestamps["QQQ"] == "2026-04-28T19:00:00+00:00"
