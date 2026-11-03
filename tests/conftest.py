"""Shared fixtures: deterministic synthetic hourly bars."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_bars(
    seed: int,
    n: int = 7 * 252,
    mu: float = 0.08,
    sigma: float = 0.25,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Fat-tailed hourly bars with overnight-style gaps between close and
    the next open, a high/low envelope and lognormal volume."""
    rng = np.random.default_rng(seed)
    dt = 1 / (252 * 7)
    index = pd.date_range("2024-01-02 14:30", periods=n, freq="h", tz="UTC")
    shocks = rng.standard_t(4, n) / np.sqrt(2)
    close = start_price * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks))
    gaps = sigma * np.sqrt(dt) * rng.normal(0, 0.5, n)
    open_ = np.r_[start_price, close[:-1]] * np.exp(gaps)
    wick = sigma * np.sqrt(dt) * 0.6
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, wick, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, wick, n)))
    volume = rng.lognormal(np.log(400_000), 0.4, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


@pytest.fixture
def synthetic_universe() -> dict[str, pd.DataFrame]:
    drifts = {"AAA": 0.12, "BBB": 0.06, "CCC": 0.15, "DDD": 0.02}
    return {symbol: make_synthetic_bars(i + 1, mu=mu) for i, (symbol, mu) in enumerate(drifts.items())}


@pytest.fixture
def synthetic_proxy() -> pd.DataFrame:
    return make_synthetic_bars(99, mu=0.08, sigma=0.16)
