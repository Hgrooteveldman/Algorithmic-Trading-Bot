"""Strategy contract.

Any strategy must accept a DataFrame of bars and return one of three signals.
The orchestration code stays the same regardless of which strategy is used.
"""
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy(ABC):
    """All strategies must implement generate_signal()."""

    @abstractmethod
    def generate_signal(self, bars: pd.DataFrame) -> Signal:
        """Given recent OHLCV bars (most recent last), return a signal.

        IMPORTANT: only use already-closed bars to make decisions. Acting on
        a still-forming bar is a classic look-ahead bug.
        """
        ...
