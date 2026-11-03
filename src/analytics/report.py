"""Performance report generator (Markdown and self-contained HTML).

The report is meant to be read by someone deciding whether a backtest is
trustworthy, so it leads with the selection-bias-aware numbers (Deflated
Sharpe, CPCV out-of-sample distribution, PBO) next to the usual ones.
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.metrics import PerformanceSummary, drawdown_stats


@dataclass
class ValidationSummary:
    n_trials: int | None = None
    cpcv: dict | None = None      # CPCVResult.summary(...)
    pbo: float | None = None


def _fmt(value: float | int | None, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if kind == "pct":
        return f"{value:+.2%}"
    if kind == "pct0":
        return f"{value:.2%}"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "prob":
        return f"{value:.1%}"
    return str(value)


def _rows(summary: PerformanceSummary, validation: ValidationSummary | None) -> list[tuple[str, str, str]]:
    rows = [
        ("Total return", _fmt(summary.total_return, "pct"), "Whole period."),
        ("CAGR", _fmt(summary.cagr, "pct"), "Compound annual growth."),
        ("Volatility", _fmt(summary.volatility, "pct0"), "Annualised standard deviation of returns."),
        ("Sharpe", _fmt(summary.sharpe, "ratio"), "Annualised; risk-free rate 0."),
        ("Sortino", _fmt(summary.sortino, "ratio"), "Excess return over downside deviation."),
        ("Calmar", _fmt(summary.calmar, "ratio"), "CAGR over max drawdown."),
        ("Omega", _fmt(summary.omega, "ratio"), "Gains over losses relative to 0."),
        ("Max drawdown", _fmt(summary.max_drawdown, "pct"), "Worst peak-to-trough."),
        ("Longest time under water", f"{summary.max_drawdown_duration_periods} bars", "Longest stretch below a prior peak."),
        ("Recovery of max drawdown", f"{summary.recovery_periods} bars" if summary.recovery_periods is not None else "not recovered", "Trough back to the old peak."),
        ("Probabilistic Sharpe (PSR)", _fmt(summary.psr, "prob"), "P(true Sharpe > 0), adjusted for skew and fat tails."),
    ]
    if summary.dsr is not None:
        rows.append(("Deflated Sharpe (DSR)", _fmt(summary.dsr, "prob"),
                     "P(true Sharpe beats the best of N skill-less trials). Below ~95% is not evidence of skill."))
    if validation and validation.n_trials:
        rows.append(("Configurations tried", str(validation.n_trials), "Used by the DSR correction."))
    if validation and validation.cpcv:
        c = validation.cpcv
        rows += [
            ("CPCV out-of-sample paths", str(c["n_paths"]), "Purged, embargoed combinatorial splits."),
            ("CPCV OOS Sharpe, median", _fmt(c["oos_sharpe_median"], "ratio"), "Annualised, across paths."),
            ("CPCV OOS Sharpe, 10th pct", _fmt(c["oos_sharpe_p10"], "ratio"), "A bad but plausible outcome."),
            ("Paths with positive Sharpe", _fmt(c["share_paths_positive"], "prob"), ""),
        ]
    if validation and validation.pbo is not None:
        rows.append(("Probability of backtest overfitting", _fmt(validation.pbo, "prob"),
                     "Share of splits where the in-sample winner lands below the OOS median."))
    return rows


def _verdict(summary: PerformanceSummary, validation: ValidationSummary | None) -> str:
    warnings = []
    if summary.dsr is not None and summary.dsr < 0.95:
        warnings.append("the Deflated Sharpe is below 95%, so the result is consistent with search luck")
    if validation and validation.pbo is not None and validation.pbo > 0.5:
        warnings.append("more than half the splits show the in-sample winner underperforming out of sample")
    if validation and validation.cpcv and validation.cpcv.get("oos_sharpe_median", 0) <= 0:
        warnings.append("the median out-of-sample Sharpe is not positive")
    if not warnings:
        return "No red flags from the overfitting checks. Still paper-trade before going live."
    return "Treat with caution: " + "; ".join(warnings) + "."


def _svg_line(series: pd.Series, width: int = 720, height: int = 180, color: str = "#2f6fdf", fill_below: bool = False) -> str:
    values = pd.Series(series, dtype=float).dropna().to_numpy()
    if len(values) < 2:
        return ""
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo or 1.0
    xs = np.linspace(4, width - 4, len(values))
    ys = height - 4 - (values - lo) / span * (height - 8)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = ""
    if fill_below:
        area = f'<polygon points="4,{height - 4} {points} {width - 4},{height - 4}" fill="{color}" opacity="0.15"/>'
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" preserveAspectRatio="none">{area}'
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>')


def render_markdown(title: str, summary: PerformanceSummary, validation: ValidationSummary | None = None) -> str:
    lines = [f"# {title}", "", f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC_", "",
             f"**Verdict:** {_verdict(summary, validation)}", "", "| Metric | Value | Meaning |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in _rows(summary, validation)]
    return "\n".join(lines) + "\n"


def render_html(title: str, equity: pd.Series, summary: PerformanceSummary, validation: ValidationSummary | None = None) -> str:
    equity = pd.Series(equity, dtype=float).dropna()
    drawdown = equity / equity.cummax() - 1 if len(equity) else equity
    dd = drawdown_stats(equity)
    rows = "".join(
        f"<tr><th>{html.escape(n)}</th><td>{html.escape(v)}</td><td>{html.escape(note)}</td></tr>"
        for n, v, note in _rows(summary, validation)
    )
    period = f"{equity.index[0]} → {equity.index[-1]}" if len(equity) else ""
    dd_note = f"Max drawdown {dd.max_drawdown:.2%} from {dd.peak} to {dd.trough}" if dd.peak is not None else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1b1b1f;background:#fff}}
h1{{font-weight:500}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:.35rem .5rem;border-bottom:1px solid #e3e3e8;vertical-align:top}}
td:nth-child(2){{font-variant-numeric:tabular-nums;white-space:nowrap}} td:nth-child(3){{color:#5b5b66;font-size:13px}}
.verdict{{padding:.6rem .8rem;border-left:3px solid #c47f00;background:#faf6ee;margin:1rem 0}} .muted{{color:#5b5b66;font-size:13px}}
@media (prefers-color-scheme:dark){{body{{background:#141417;color:#e8e8ec}} th,td{{border-color:#2c2c33}} td:nth-child(3),.muted{{color:#a0a0aa}} .verdict{{background:#2a2418}}}}
</style></head><body>
<h1>{html.escape(title)}</h1><p class="muted">{html.escape(period)} · generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC</p>
<p class="verdict">{html.escape(_verdict(summary, validation))}</p>
<h2>Equity</h2>{_svg_line(equity, fill_below=True)}
<h2>Drawdown</h2>{_svg_line(drawdown, color="#c0392b", fill_below=True)}<p class="muted">{html.escape(dd_note)}</p>
<h2>Metrics</h2><table>{rows}</table>
</body></html>"""


def write_report(
    output_dir: Path,
    name: str,
    title: str,
    equity: pd.Series,
    summary: PerformanceSummary,
    validation: ValidationSummary | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{name}.md"
    html_path = output_dir / f"{name}.html"
    md_path.write_text(render_markdown(title, summary, validation), encoding="utf-8")
    html_path.write_text(render_html(title, equity, summary, validation), encoding="utf-8")
    return md_path, html_path
