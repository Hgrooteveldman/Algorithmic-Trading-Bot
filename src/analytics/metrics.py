"""Performance metrics beyond return on investment.

All functions take a series of *periodic* simple returns (one per bar) or an
equity curve, plus ``periods_per_year`` for annualisation. Nothing here
depends on a strategy; the backtest engines, the parameter sweep and the
report generator all use the same definitions.

References
----------
- Sortino & van der Meer (1991), downside deviation.
- Keating & Shadwick (2002), the Omega ratio.
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier"
  (probabilistic Sharpe ratio, PSR).
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality" (DSR).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

_NORMAL = NormalDist()
_STD_EPS = 1e-12  # below this a series is treated as constant
EULER_MASCHERONI = 0.5772156649015329


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Simple per-period returns; empty for fewer than two points."""
    equity = pd.Series(equity, dtype=float).dropna()
    if len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free_per_period: float = 0.0) -> float:
    excess = pd.Series(returns, dtype=float).dropna() - risk_free_per_period
    std = excess.std(ddof=1)
    if len(excess) < 2 or not math.isfinite(std) or std < _STD_EPS:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, target_per_period: float = 0.0) -> float:
    """Annualised mean excess over target divided by downside deviation
    (root mean square of shortfalls below target, over *all* periods)."""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    shortfall = np.minimum(r - target_per_period, 0.0)
    downside = math.sqrt(float(np.mean(shortfall**2)))
    if downside == 0:
        return math.inf if (r - target_per_period).mean() > 0 else 0.0
    return float((r - target_per_period).mean() / downside * math.sqrt(periods_per_year))


def omega_ratio(returns: pd.Series, threshold_per_period: float = 0.0) -> float:
    """Probability-weighted gains over losses relative to a threshold."""
    r = pd.Series(returns, dtype=float).dropna() - threshold_per_period
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return math.inf if gains > 0 else 1.0
    return gains / losses


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    equity = pd.Series(equity, dtype=float).dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float               # most negative peak-to-trough, e.g. -0.12
    peak: object | None               # index label of the peak before the max drawdown
    trough: object | None
    recovery: object | None           # first label back at the old peak, None if never
    max_duration_periods: int         # longest stretch spent below a previous peak
    recovery_periods: int | None      # trough -> recovery of the max drawdown


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    equity = pd.Series(equity, dtype=float).dropna()
    if len(equity) < 2:
        return DrawdownStats(0.0, None, None, None, 0, None)
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough_pos = int(np.argmin(drawdown.to_numpy()))
    max_dd = float(drawdown.iloc[trough_pos])
    if max_dd == 0.0:
        return DrawdownStats(0.0, None, None, None, 0, None)
    peak_pos = int(np.argmax(equity.iloc[: trough_pos + 1].to_numpy()))
    peak_value = float(equity.iloc[peak_pos])
    after = np.flatnonzero(equity.iloc[trough_pos:].to_numpy() >= peak_value)
    recovery_pos = trough_pos + int(after[0]) if len(after) else None

    underwater = (drawdown < 0).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    return DrawdownStats(
        max_drawdown=max_dd,
        peak=equity.index[peak_pos],
        trough=equity.index[trough_pos],
        recovery=equity.index[recovery_pos] if recovery_pos is not None else None,
        max_duration_periods=int(longest),
        recovery_periods=(recovery_pos - trough_pos) if recovery_pos is not None else None,
    )


def calmar_ratio(equity: pd.Series, periods_per_year: float) -> float:
    dd = drawdown_stats(equity).max_drawdown
    growth = cagr(equity, periods_per_year)
    if dd == 0:
        return math.inf if growth > 0 else 0.0
    return growth / abs(dd)


# --- selection-bias-aware Sharpe ---------------------------------------------

def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe_per_period: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark), allowing for skew and fat tails.

    Uses the *non-annualised* Sharpe of the sample; the benchmark must be on
    the same per-period scale.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    std = r.std(ddof=1)
    if n < 3 or not math.isfinite(std) or std < _STD_EPS:
        return float("nan")
    sr = float(r.mean() / std)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas reports excess kurtosis
    denom_sq = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom_sq <= 0:
        return float("nan")
    z = (sr - benchmark_sharpe_per_period) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(_NORMAL.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe ratios drawn under
    the null of zero skill, given the cross-trial variance of their Sharpes
    (False Strategy Theorem, Bailey & López de Prado 2014)."""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    a = _NORMAL.inv_cdf(1 - 1 / n_trials)
    b = _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, trial_sharpes_per_period: list[float] | np.ndarray) -> float:
    """Probability that the selected strategy's Sharpe beats what the best
    of ``n_trials`` skill-less strategies would show by luck alone.

    ``trial_sharpes_per_period`` are the non-annualised Sharpes of every
    configuration that was tried (their variance sets the luck threshold).
    Values above ~0.95 are evidence of skill after correcting for multiple
    testing; around 0.5 or below means the result is what search alone
    produces.
    """
    trials = np.asarray([s for s in trial_sharpes_per_period if np.isfinite(s)], dtype=float)
    variance = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    threshold = expected_max_sharpe(max(n_trials, len(trials)), variance)
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe_per_period=threshold)


def per_period_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    std = r.std(ddof=1)
    if len(r) < 2 or not math.isfinite(std) or std < _STD_EPS:
        return 0.0
    return float(r.mean() / std)


# --- bundle ---------------------------------------------------------------------

@dataclass(frozen=True)
class PerformanceSummary:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    max_drawdown_duration_periods: int
    recovery_periods: int | None
    psr: float
    dsr: float | None
    n_periods: int

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(
    equity: pd.Series,
    periods_per_year: float,
    *,
    n_trials: int | None = None,
    trial_sharpes_per_period: list[float] | np.ndarray | None = None,
) -> PerformanceSummary:
    equity = pd.Series(equity, dtype=float).dropna()
    r = returns_from_equity(equity)
    dd = drawdown_stats(equity)
    dsr = None
    if n_trials and trial_sharpes_per_period is not None and len(trial_sharpes_per_period) > 1:
        dsr = deflated_sharpe_ratio(r, n_trials, trial_sharpes_per_period)
    return PerformanceSummary(
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
        cagr=cagr(equity, periods_per_year),
        volatility=float(r.std(ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else 0.0,
        sharpe=sharpe_ratio(r, periods_per_year),
        sortino=sortino_ratio(r, periods_per_year),
        calmar=calmar_ratio(equity, periods_per_year),
        omega=omega_ratio(r),
        max_drawdown=dd.max_drawdown,
        max_drawdown_duration_periods=dd.max_duration_periods,
        recovery_periods=dd.recovery_periods,
        psr=probabilistic_sharpe_ratio(r),
        dsr=dsr,
        n_periods=len(r),
    )
