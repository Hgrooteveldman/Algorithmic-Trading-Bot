"""Bar-event stream with automatic reconnects.

The bot trades hourly bars, so a full asyncio rewrite of the trading loop
would add complexity without a speed benefit. What the stream does add is
*promptness*: instead of sleeping a fixed poll interval, the loop wakes as
soon as a new bar arrives for any configured symbol, and falls back to
polling whenever the stream is down.

The websocket runs in a daemon thread (alpaca-py's ``StockDataStream`` owns
its own asyncio loop). If it drops, it is restarted with exponential backoff
and jitter; a healthy run resets the backoff.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Protocol

from loguru import logger

from src.broker.models import Bar


class StreamLike(Protocol):
    def subscribe_bars(self, handler: Callable[[Any], Any], *symbols: str) -> None: ...
    def run(self) -> None: ...
    def stop(self) -> None: ...


StreamFactory = Callable[[], StreamLike]


def alpaca_stream_factory() -> StreamLike:  # pragma: no cover - needs credentials and network
    from alpaca.data.live import StockDataStream

    from config.settings import settings

    return StockDataStream(settings.alpaca_api_key, settings.alpaca_secret_key)


@dataclass
class BarStream:
    symbols: Iterable[str]
    factory: StreamFactory = alpaca_stream_factory
    base_backoff: float = 1.0
    max_backoff: float = 60.0
    healthy_after_s: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    last_bars: dict[str, Bar] = field(default_factory=dict)
    reconnects: int = 0
    _event: threading.Event = field(default_factory=threading.Event, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stream: StreamLike | None = field(default=None, init=False)
    connected: bool = field(default=False, init=False)

    async def _on_bar(self, raw: Any) -> None:
        self.handle_bar(raw)

    def handle_bar(self, raw: Any) -> Bar | None:
        """Validate one incoming bar; wake the trading loop if it is new."""
        try:
            bar = Bar.model_validate(raw)
        except Exception as exc:
            logger.bind(event="bad_stream_bar").warning(f"Dropping invalid stream bar: {exc}")
            return None
        previous = self.last_bars.get(bar.symbol)
        if previous is None or bar.timestamp > previous.timestamp:
            self.last_bars[bar.symbol] = bar
            self._event.set()
        return bar

    def run_forever(self) -> None:
        """Connect, run, reconnect with backoff until :meth:`stop`."""
        attempt = 0
        symbols = list(dict.fromkeys(s.upper() for s in self.symbols))
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._stream = self.factory()
                self._stream.subscribe_bars(self._on_bar, *symbols)
                self.connected = True
                logger.bind(event="stream_connected", symbols=symbols).info("Bar stream connected")
                self._stream.run()  # blocks until the socket closes
            except Exception as exc:
                logger.bind(event="stream_error").warning(f"Bar stream error: {exc!r}")
            finally:
                self.connected = False
            if self._stop.is_set():
                break
            if time.monotonic() - started >= self.healthy_after_s:
                attempt = 0
            wait = random.uniform(0, min(self.max_backoff, self.base_backoff * 2**attempt))
            attempt += 1
            self.reconnects += 1
            logger.bind(event="stream_reconnect", attempt=attempt, wait_s=wait).warning(
                f"Bar stream down; reconnecting in {wait:.1f}s (attempt {attempt})"
            )
            self.sleep(wait)

    def start(self) -> "BarStream":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self.run_forever, name="bar-stream", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass
        self._event.set()

    def wait_for_bar(self, timeout: float) -> bool:
        """Sleep until a new bar arrives or ``timeout`` passes. Returns True
        if woken by a bar. Use in place of ``time.sleep(poll_interval)``."""
        woke = self._event.wait(timeout)
        self._event.clear()
        return woke

    def latest_bar_time(self, symbol: str) -> datetime | None:
        bar = self.last_bars.get(symbol.upper())
        return bar.timestamp if bar else None
