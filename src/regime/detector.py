"""Market regime detection and the position-size policy that follows from it.

Two detectors, both strictly causal (a regime label at bar ``t`` uses data
up to and including ``t`` only):

1. **Rules (default)** — ATR% relative to its own trailing history plus ADX.
   Transparent, robust, and cheap::

       high_vol_trend : ATR% in its top 20% AND ADX >= 25
       trend          : ADX >= 25
       low_vol_chop   : ATR% in its bottom 30% AND ADX < 20
       neutral        : everything else

2. **Gaussian HMM (optional)** — a two-state model on bar returns, fitted by
   Baum-Welch on a training window. Labels come from the *filtered*
   probability P(state_t | r_1..r_t), never the smoothed one, because
   smoothing uses future returns and would leak into a backtest.

The policy maps a regime to a size multiplier per strategy family: mean
reversion de-leverages in trends and stops in volatile trends; momentum /
breakout stops in quiet chop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class Regime(str, Enum):
    HIGH_VOL_TREND = "high_vol_trend"
    TREND = "trend"
    LOW_VOL_CHOP = "low_vol_chop"
    NEUTRAL = "neutral"


StrategyKind = Literal["mean_reversion", "momentum"]


@dataclass(frozen=True)
class RegimeThresholds:
    atr_window: int = 14
    adx_window: int = 14
    vol_lookback: int = 500          # bars of ATR% history for the percentile
    min_history: int = 100           # bars before a percentile is trusted
    high_vol_percentile: float = 0.80
    low_vol_percentile: float = 0.30
    trend_adx: float = 25.0
    chop_adx: float = 20.0


@dataclass(frozen=True)
class RegimePolicy:
    """Size multipliers by strategy family and regime."""

    mean_reversion: dict[Regime, float] = None  # type: ignore[assignment]
    momentum: dict[Regime, float] = None        # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mean_reversion is None:
            object.__setattr__(self, "mean_reversion", {
                Regime.HIGH_VOL_TREND: 0.0, Regime.TREND: 0.5,
                Regime.LOW_VOL_CHOP: 1.0, Regime.NEUTRAL: 1.0,
            })
        if self.momentum is None:
            object.__setattr__(self, "momentum", {
                Regime.HIGH_VOL_TREND: 1.0, Regime.TREND: 1.0,
                Regime.LOW_VOL_CHOP: 0.0, Regime.NEUTRAL: 1.0,
            })

    def multiplier(self, kind: StrategyKind, regime: Regime | str) -> float:
        table = self.mean_reversion if kind == "mean_reversion" else self.momentum
        return float(table.get(Regime(regime), 1.0))


# --- indicators -------------------------------------------------------------------

def _wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def average_true_range(bars: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), bars["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return _wilder(true_range, window)


def adx(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (vectorised)."""
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=bars.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=bars.index)
    atr = average_true_range(bars, window)
    plus_di = 100 * _wilder(plus_dm, window) / atr
    minus_di = 100 * _wilder(minus_dm, window) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, window)


def trailing_percentile(series: pd.Series, lookback: int, min_history: int) -> pd.Series:
    """Rank of each value within its own trailing window, in [0, 1].
    Causal: the window ends at the current bar."""
    def rank_last(window: np.ndarray) -> float:
        last = window[-1]
        valid = window[np.isfinite(window)]
        if len(valid) < min_history or not np.isfinite(last):
            return np.nan
        return float(np.mean(valid <= last))

    return series.rolling(lookback, min_periods=min_history).apply(rank_last, raw=True)


def regime_frame(bars: pd.DataFrame, thresholds: RegimeThresholds | None = None) -> pd.DataFrame:
    """Columns ``atr_pct``, ``atr_pct_rank``, ``adx`` and ``regime``."""
    t = thresholds or RegimeThresholds()
    out = pd.DataFrame(index=bars.index)
    out["atr_pct"] = average_true_range(bars, t.atr_window) / bars["close"].astype(float)
    out["atr_pct_rank"] = trailing_percentile(out["atr_pct"], t.vol_lookback, t.min_history)
    out["adx"] = adx(bars, t.adx_window)

    rank, strength = out["atr_pct_rank"], out["adx"]
    regime = np.full(len(out), Regime.NEUTRAL.value, dtype=object)
    trend = strength >= t.trend_adx
    regime[trend.to_numpy()] = Regime.TREND.value
    regime[(trend & (rank >= t.high_vol_percentile)).to_numpy()] = Regime.HIGH_VOL_TREND.value
    regime[((strength < t.chop_adx) & (rank <= t.low_vol_percentile)).to_numpy()] = Regime.LOW_VOL_CHOP.value
    regime[(rank.isna() | strength.isna()).to_numpy()] = Regime.NEUTRAL.value
    out["regime"] = regime
    return out


def regime_multipliers(
    bars: pd.DataFrame,
    kind: StrategyKind,
    policy: RegimePolicy | None = None,
    thresholds: RegimeThresholds | None = None,
) -> pd.Series:
    policy = policy or RegimePolicy()
    labels = regime_frame(bars, thresholds)["regime"]
    return labels.map(lambda r: policy.multiplier(kind, r)).astype(float)


# --- optional two-state Gaussian HMM ------------------------------------------------

@dataclass
class GaussianHMM2:
    """Two-state Gaussian HMM on returns, numpy only.

    After :meth:`fit`, state 1 is always the higher-variance state.
    """

    means: np.ndarray | None = None
    variances: np.ndarray | None = None
    transition: np.ndarray | None = None
    initial: np.ndarray | None = None

    def _emission(self, returns: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.variances is not None
        var = np.maximum(self.variances, 1e-12)
        z = (returns[:, None] - self.means[None, :]) ** 2 / var[None, :]
        return np.exp(-0.5 * z) / np.sqrt(2 * math.pi * var[None, :])

    def _forward(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.transition is not None and self.initial is not None
        emission = self._emission(returns)
        n = len(returns)
        alpha = np.zeros((n, 2))
        scale = np.zeros(n)
        alpha[0] = self.initial * emission[0]
        scale[0] = alpha[0].sum() or 1e-300
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.transition) * emission[t]
            scale[t] = alpha[t].sum() or 1e-300
            alpha[t] /= scale[t]
        return alpha, scale

    def fit(self, returns: np.ndarray | pd.Series, n_iter: int = 50, tol: float = 1e-6) -> "GaussianHMM2":
        r = np.asarray(pd.Series(returns).dropna(), dtype=float)
        if len(r) < 50:
            raise ValueError("need at least 50 returns to fit")
        q = np.quantile(np.abs(r - r.mean()), 0.7)
        calm, wild = r[np.abs(r - r.mean()) <= q], r[np.abs(r - r.mean()) > q]
        self.means = np.array([calm.mean(), wild.mean()])
        self.variances = np.array([calm.var() + 1e-12, wild.var() + 1e-12])
        self.transition = np.array([[0.95, 0.05], [0.10, 0.90]])
        self.initial = np.array([0.5, 0.5])
        previous = -np.inf
        for _ in range(n_iter):
            alpha, scale = self._forward(r)
            emission = self._emission(r)
            beta = np.ones((len(r), 2))
            for t in range(len(r) - 2, -1, -1):
                beta[t] = (self.transition @ (emission[t + 1] * beta[t + 1])) / scale[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (alpha[:-1, :, None] * self.transition[None] * (emission[1:] * beta[1:])[:, None, :]) / scale[1:, None, None]
            self.initial = gamma[0]
            self.transition = xi.sum(axis=0) / np.maximum(gamma[:-1].sum(axis=0)[:, None], 1e-300)
            self.transition /= self.transition.sum(axis=1, keepdims=True)
            weights = gamma.sum(axis=0)
            self.means = (gamma * r[:, None]).sum(axis=0) / weights
            self.variances = (gamma * (r[:, None] - self.means) ** 2).sum(axis=0) / weights + 1e-12
            log_likelihood = float(np.log(scale).sum())
            if abs(log_likelihood - previous) < tol:
                break
            previous = log_likelihood
        if self.variances[0] > self.variances[1]:   # make state 1 the volatile one
            order = [1, 0]
            self.means, self.variances = self.means[order], self.variances[order]
            self.transition = self.transition[np.ix_(order, order)]
            self.initial = self.initial[order]
        return self

    def filtered_high_vol_probability(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """P(high-volatility state at t | returns up to t). Causal."""
        if self.means is None:
            raise RuntimeError("fit the model first")
        alpha, _ = self._forward(np.asarray(returns, dtype=float))
        return alpha[:, 1]
