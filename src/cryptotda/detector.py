"""Detection on top of the raw TDA signal.

The raw landscape norm L_t lives on a non-stationary scale (it depends on the
number of assets, the window, and on how dispersed returns happen to be in
that period). We convert it to a normalized signal via a *causal* rolling
z-score: at each t we use only data strictly before t to estimate the mean
and standard deviation. This avoids look-ahead bias which is the single most
common source of inflated results in financial backtests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_signal(
    raw: pd.Series,
    lookback: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """Causal rolling z-score. Uses (lookback) days *strictly before* t."""
    shifted = raw.shift(1)  # exclude current sample so it really is causal
    mu = shifted.rolling(lookback, min_periods=min_periods).mean()
    sd = shifted.rolling(lookback, min_periods=min_periods).std()
    return ((raw - mu) / sd).rename(f"{raw.name}_z" if raw.name else "z")


def threshold_signal(z: pd.Series, threshold: float = 1.5) -> pd.Series:
    """Binary alarm series."""
    return (z >= threshold).astype(int).rename("alarm")


def lead_time(
    alarms: pd.Series,
    crash_dates: list[pd.Timestamp],
    max_lookback: int = 60,
) -> list[float]:
    """For each crash, the number of days between the first alarm in the
    [crash - max_lookback, crash) window and the crash date.

    Returns an empty list element NaN if no alarm preceded the crash.
    """
    results = []
    for c in crash_dates:
        window = alarms[(alarms.index >= c - pd.Timedelta(days=max_lookback)) & (alarms.index < c)]
        if len(window) == 0 or window.sum() == 0:
            results.append(float("nan"))
            continue
        first_alarm = window[window == 1].index[0]
        results.append((c - first_alarm).days)
    return results


def hit_rate(
    alarms: pd.Series,
    crash_dates: list[pd.Timestamp],
    max_lookback: int = 30,
) -> float:
    """Fraction of crashes preceded by at least one alarm within `max_lookback` days."""
    if not crash_dates:
        return float("nan")
    hits = 0
    for c in crash_dates:
        window = alarms[(alarms.index >= c - pd.Timedelta(days=max_lookback)) & (alarms.index < c)]
        if window.sum() > 0:
            hits += 1
    return hits / len(crash_dates)
