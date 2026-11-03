"""Market-data adapter.

Wraps the broker's bar fetching and applies a liquidity filter at the edge.
If a symbol falls below the liquidity floor, downstream code receives an
empty DataFrame and treats the symbol as untradeable for that cycle.
"""
import math

import pandas as pd

from src.broker.alpaca_client import AlpacaClient
from config.settings import settings


# Number of bars per session, used to convert a daily-volume floor
# into a per-bar floor. NYSE = 6.5 hours = 390 minutes.
def _bars_per_session() -> int:
    return max(1, 390 // settings.bar_timeframe_minutes)


def correlation_lookback_days() -> int:
    sessions_needed = math.ceil(settings.correlation_lookback_bars / _bars_per_session())
    return max(10, sessions_needed + 5)


def close_returns_for_correlation(bars: pd.DataFrame) -> pd.Series:
    if bars.empty or "close" not in bars:
        return pd.Series(dtype=float)
    return bars["close"].pct_change().dropna().tail(settings.correlation_lookback_bars)


def get_recent_bars(
    broker: AlpacaClient,
    symbol: str,
    lookback_days: int = 10,
) -> pd.DataFrame:
    """Returns OHLCV bars for `symbol`, or empty if illiquid / unavailable."""
    bars = broker.get_bars(symbol, lookback_days=lookback_days)
    if bars.empty:
        return bars

    # Liquidity filter: discard symbols whose recent dollar volume is too thin.
    avg_dollar_volume_per_bar = (bars["close"] * bars["volume"]).mean()
    floor_per_bar = settings.min_avg_daily_dollar_volume / _bars_per_session()
    if avg_dollar_volume_per_bar < floor_per_bar:
        return pd.DataFrame()

    return bars
