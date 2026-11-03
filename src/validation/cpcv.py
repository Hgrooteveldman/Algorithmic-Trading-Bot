"""Combinatorial Purged Cross-Validation (CPCV) and backtest overfitting.

The old sweep made one chronological train/test split and then picked the
configuration with the best score. With hundreds of configurations, the best
in-sample result is mostly luck, and one test window is a single sample.

CPCV (López de Prado, *Advances in Financial Machine Learning*, ch. 7 and
12) instead:

1. cuts the timeline into ``n_groups`` contiguous groups;
2. for every combination of ``n_test_groups`` of them, trains on the rest
   and tests on those groups;
3. **purges** training observations whose outcome window overlaps a test
   group (a trade opened just before a test group is still running inside
   it), and adds an **embargo** after each test group, because serially
   correlated returns leak information forward;
4. stitches the test groups into ``k/N * C(N, k)`` complete out-of-sample
   paths, so you get a *distribution* of out-of-sample Sharpe ratios rather
   than one number.

:func:`probability_of_backtest_overfitting` implements the CSCV estimate of
Bailey, Borwein, López de Prado & Zhu (2015): how often the configuration
that wins in-sample ends up in the bottom half out-of-sample.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

ScoreFn = Callable[[np.ndarray], float]


def _sharpe(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    std = values.std(ddof=1)
    return float(values.mean() / std) if std > 1e-12 else 0.0


@dataclass(frozen=True)
class Split:
    train: np.ndarray        # positional indices
    test: np.ndarray
    test_groups: tuple[int, ...]


@dataclass
class CombinatorialPurgedKFold:
    """Index generator for CPCV.

    Attributes:
        n_groups: number of contiguous groups the sample is cut into.
        n_test_groups: groups held out per split.
        purge_periods: an observation's outcome is assumed to depend on the
            next ``purge_periods`` observations (e.g. the maximum holding
            period). Training observations within that distance *before* a
            test group are removed.
        embargo_periods: training observations within this distance *after*
            a test group are removed.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    purge_periods: int = 0
    embargo_periods: int = 0

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not 1 <= self.n_test_groups < self.n_groups:
            raise ValueError("n_test_groups must be in [1, n_groups)")
        if self.purge_periods < 0 or self.embargo_periods < 0:
            raise ValueError("purge and embargo must be non-negative")

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return self.n_splits * self.n_test_groups // self.n_groups

    def group_bounds(self, n_obs: int) -> list[tuple[int, int]]:
        if n_obs < self.n_groups:
            raise ValueError("fewer observations than groups")
        edges = np.linspace(0, n_obs, self.n_groups + 1).astype(int)
        return [(int(edges[g]), int(edges[g + 1])) for g in range(self.n_groups)]

    def split(self, n_obs: int) -> Iterator[Split]:
        bounds = self.group_bounds(n_obs)
        positions = np.arange(n_obs)
        for test_groups in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_mask = np.zeros(n_obs, dtype=bool)
            banned = np.zeros(n_obs, dtype=bool)
            for g in test_groups:
                start, stop = bounds[g]
                test_mask[start:stop] = True
                banned[max(0, start - self.purge_periods):start] = True        # purge
                banned[stop:min(n_obs, stop + self.embargo_periods)] = True    # embargo
            train_mask = ~test_mask & ~banned
            yield Split(positions[train_mask], positions[test_mask], tuple(test_groups))


@dataclass
class CPCVResult:
    path_sharpes: np.ndarray                     # per-period Sharpe of each OOS path
    path_returns: list[pd.Series]                # stitched OOS return series, one per path
    chosen: list[tuple[tuple[int, ...], str]]    # (test groups, configuration picked on train)
    selection_counts: dict[str, int] = field(default_factory=dict)

    def summary(self, periods_per_year: float) -> dict:
        annual = self.path_sharpes * math.sqrt(periods_per_year)
        return {
            "n_paths": int(len(annual)),
            "oos_sharpe_mean": float(np.mean(annual)) if len(annual) else float("nan"),
            "oos_sharpe_median": float(np.median(annual)) if len(annual) else float("nan"),
            "oos_sharpe_p10": float(np.percentile(annual, 10)) if len(annual) else float("nan"),
            "oos_sharpe_min": float(np.min(annual)) if len(annual) else float("nan"),
            "share_paths_positive": float(np.mean(annual > 0)) if len(annual) else float("nan"),
            "most_selected": max(self.selection_counts, key=self.selection_counts.get) if self.selection_counts else None,
        }


def cpcv_select_and_evaluate(
    returns: pd.DataFrame,
    cv: CombinatorialPurgedKFold,
    score: ScoreFn = _sharpe,
) -> CPCVResult:
    """Walk every CPCV split: pick the column (configuration) with the best
    ``score`` on the purged training rows, record its returns on the test
    groups, and stitch those into out-of-sample paths.

    ``returns`` is periods x configurations of per-period strategy returns,
    aligned on one index, with NaN where a configuration has no data.
    """
    if returns.shape[1] < 1:
        raise ValueError("need at least one configuration")
    values = returns.to_numpy(dtype=float)
    n_obs = len(returns)
    bounds = cv.group_bounds(n_obs)

    # For each group, the selected configuration's test returns, in order of
    # the splits in which that group was a test group.
    per_group: dict[int, list[np.ndarray]] = {g: [] for g in range(cv.n_groups)}
    chosen: list[tuple[tuple[int, ...], str]] = []
    counts: dict[str, int] = {}
    for split in cv.split(n_obs):
        train_scores = [score(values[split.train, j]) for j in range(values.shape[1])]
        best = int(np.nanargmax(train_scores))
        name = str(returns.columns[best])
        chosen.append((split.test_groups, name))
        counts[name] = counts.get(name, 0) + 1
        for g in split.test_groups:
            start, stop = bounds[g]
            per_group[g].append(values[start:stop, best])

    path_returns: list[pd.Series] = []
    for p in range(cv.n_paths):
        pieces = [per_group[g][p] for g in range(cv.n_groups)]
        series = pd.Series(np.concatenate(pieces), index=returns.index)
        path_returns.append(series)
    path_sharpes = np.array([score(s.to_numpy()) for s in path_returns])
    return CPCVResult(path_sharpes=path_sharpes, path_returns=path_returns, chosen=chosen, selection_counts=counts)


@dataclass(frozen=True)
class PBOResult:
    pbo: float                 # probability the in-sample winner is below the OOS median
    logits: np.ndarray
    n_combinations: int


def probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    n_blocks: int = 10,
    score: ScoreFn = _sharpe,
) -> PBOResult:
    """Combinatorially symmetric cross-validation (CSCV).

    Split the sample into ``n_blocks`` (even) blocks; for every half/half
    combination, find the best configuration in-sample and its relative
    rank out-of-sample. PBO is the share of combinations where that rank is
    below the median (logit <= 0).
    """
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    n_configs = returns.shape[1]
    if n_configs < 2:
        raise ValueError("PBO needs at least two configurations")
    values = returns.to_numpy(dtype=float)
    edges = np.linspace(0, len(values), n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    logits = []
    for in_sample in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[b] for b in in_sample])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in in_sample])
        is_scores = np.array([score(values[is_rows, j]) for j in range(n_configs)])
        oos_scores = np.array([score(values[oos_rows, j]) for j in range(n_configs)])
        winner = int(np.nanargmax(is_scores))
        # relative rank in (0, 1): share of configurations the winner beats OOS
        rank = (np.sum(oos_scores < oos_scores[winner]) + 0.5 * (np.sum(oos_scores == oos_scores[winner]) - 1) + 1) / (n_configs + 1)
        rank = min(max(rank, 1e-9), 1 - 1e-9)
        logits.append(math.log(rank / (1 - rank)))
    logits_arr = np.array(logits)
    return PBOResult(pbo=float(np.mean(logits_arr <= 0)), logits=logits_arr, n_combinations=len(logits_arr))


def returns_matrix(equity_curves: dict[str, pd.Series]) -> pd.DataFrame:
    """Align equity curves of several configurations into one periods x
    configurations table of per-period returns (missing periods -> 0, i.e.
    flat, which is what a strategy with no position earns)."""
    frame = pd.DataFrame({name: pd.Series(curve, dtype=float) for name, curve in equity_curves.items()})
    frame = frame.sort_index().ffill()
    return frame.pct_change().iloc[1:].fillna(0.0)


def validate_purge_inputs(cv: CombinatorialPurgedKFold, n_obs: int, max_holding: Sequence[int] | int) -> None:
    """Fail loudly if the purge is shorter than the longest holding period
    (then training labels would still overlap test groups)."""
    longest = max(max_holding) if isinstance(max_holding, Sequence) else int(max_holding)
    if cv.purge_periods < longest:
        raise ValueError(f"purge_periods ({cv.purge_periods}) is shorter than the longest holding period ({longest})")
    if n_obs // cv.n_groups <= cv.purge_periods + cv.embargo_periods:
        raise ValueError("groups are too small for this purge + embargo")
