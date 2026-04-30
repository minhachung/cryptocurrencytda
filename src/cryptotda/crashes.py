"""Crash labelling.

Two complementary objects:
    1. detect_crash_events  -- discrete event timestamps (peak before crash)
    2. label_crash_periods  -- a binary label series y_t marking dates that
                               *precede* a crash within a horizon. This is the
                               classification target for the detector.

A "crash" is defined operationally as a forward drawdown exceeding some
threshold (e.g., -20%) within H trading days. The drawdown is computed on the
benchmark series (BTC if available, else equal-weighted basket).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def benchmark_series(prices: pd.DataFrame, benchmark: str | None = "bitcoin") -> pd.Series:
    """BTC if present, otherwise equal-weighted average of the basket."""
    if benchmark is not None and benchmark in prices.columns:
        return prices[benchmark].dropna()
    return prices.mean(axis=1).dropna()


def forward_drawdown(prices: pd.Series, horizon: int = 30) -> pd.Series:
    """For each date t, min over the next H days of P[t+i] / max(P[t..t+i]) - 1.

    A value of -0.25 means: from t there is a future H-day window during which
    the asset draws down at least 25% from its post-t running peak. This is
    the canonical "is t followed by a crash" measure.
    """
    p = prices.to_numpy()
    n = len(p)
    out = np.zeros(n)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if end - t < 2:
            out[t] = 0.0
            continue
        future = p[t:end]
        running_peak = np.maximum.accumulate(future)
        dd = future / running_peak - 1.0
        out[t] = dd.min()
    return pd.Series(out, index=prices.index, name="fwd_drawdown")


def label_crash_periods(
    prices: pd.DataFrame,
    horizon: int = 30,
    drawdown_threshold: float = -0.20,
    benchmark: str | None = "bitcoin",
) -> pd.Series:
    """Binary label: 1 if a crash >= threshold occurs within `horizon` days.

    This is what we ask the detector to predict: at date t, is t in the
    "danger zone" preceding a large drawdown?
    """
    bench = benchmark_series(prices, benchmark=benchmark)
    dd = forward_drawdown(bench, horizon=horizon)
    y = (dd <= drawdown_threshold).astype(int)
    y.name = "crash_within_horizon"
    return y


def detect_crash_events(
    prices: pd.DataFrame,
    drawdown_threshold: float = -0.30,
    min_separation_days: int = 60,
    benchmark: str | None = "bitcoin",
) -> list[pd.Timestamp]:
    """Identify discrete crash *peak* dates.

    A crash is a local maximum followed by a drawdown of at least
    `drawdown_threshold` (e.g., -0.30) before any new peak. We return the
    *peak* date because that is what an early-warning system should anticipate.
    Successive peaks are separated by at least `min_separation_days` to avoid
    counting choppy noise as multiple events.
    """
    bench = benchmark_series(prices, benchmark=benchmark)
    p = bench.to_numpy()
    idx = bench.index
    events: list[pd.Timestamp] = []
    last_event_pos = -10**9
    running_peak = p[0]
    running_peak_pos = 0
    for t in range(1, len(p)):
        if p[t] > running_peak:
            running_peak = p[t]
            running_peak_pos = t
        else:
            dd = p[t] / running_peak - 1.0
            if dd <= drawdown_threshold and running_peak_pos - last_event_pos >= min_separation_days:
                events.append(idx[running_peak_pos])
                last_event_pos = running_peak_pos
                # restart peak search from current trough so we don't double-count
                running_peak = p[t]
                running_peak_pos = t
    return events
