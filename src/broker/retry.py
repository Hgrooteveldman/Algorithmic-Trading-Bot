"""Exponential backoff with full jitter for broker calls.

Only *idempotent* calls are retried (reads: account, clock, bars, quotes,
positions, orders). Order submission is never retried blindly: a timeout
does not tell you whether the order reached the exchange, and a second
submit can double the position. Entries carry a ``client_order_id`` instead,
so a manual check can tell whether a lost submission actually landed.
"""
from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import Callable, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
T = TypeVar("T")


def _default_retryable(exc: BaseException) -> bool:
    """Network trouble and server-side errors are retryable; client errors
    (bad request, unauthorised, not found) are not."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    retryable: Callable[[BaseException], bool] = _default_retryable

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full jitter: uniform in [0, min(max_delay, base * 2**attempt)]."""
        cap = min(self.max_delay, self.base_delay * (2**attempt))
        return (rng or random).uniform(0, cap)


def with_retry(
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    policy = policy or RetryPolicy()

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(policy.attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last = attempt == policy.attempts - 1
                    if last or not policy.retryable(exc):
                        raise
                    wait = policy.delay(attempt)
                    logger.bind(event="broker_retry", call=fn.__name__, attempt=attempt + 1, wait_s=wait).warning(
                        f"{fn.__name__} failed ({exc!r}); retry {attempt + 1}/{policy.attempts - 1} in {wait:.2f}s"
                    )
                    sleep(wait)
            raise RuntimeError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
