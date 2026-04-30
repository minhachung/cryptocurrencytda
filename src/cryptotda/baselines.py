"""Baseline early-warning signals to compare TDA against.

If TDA does not beat these, it is not worth the complexity. The whole point
of running baselines is to falsify the headline claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_apply_matrix(returns: pd.DataFrame, window: int, fn) -> pd.Series:
    """Apply a function returning a scalar to each window of the (T, N) matrix."""
    arr = returns.to_numpy()
    idx = returns.index
    out = np.full(len(arr), np.nan)
    for t in range(window - 1, len(arr)):
        out[t] = fn(arr[t - window + 1: t + 1])
    return pd.Series(out, index=idx)


def realized_volatility(returns: pd.DataFrame, window: int = 50, benchmark: str | None = "bitcoin") -> pd.Series:
    """Annualized realized volatility of the benchmark (or basket mean if absent)."""
    if benchmark is not None and benchmark in returns.columns:
        r = returns[benchmark]
    else:
        r = returns.mean(axis=1)
    return (r.rolling(window).std() * np.sqrt(252)).rename("realized_vol")


def average_correlation(returns: pd.DataFrame, window: int = 50) -> pd.Series:
    """Mean off-diagonal entry of the rolling correlation matrix.

    A widely-used proxy for "everything is moving together" -- spikes during
    crashes. This is the strongest non-TDA baseline because it captures the
    same regime shift TDA picks up.
    """
    def fn(block: np.ndarray) -> float:
        c = np.corrcoef(block, rowvar=False)
        n = c.shape[0]
        if n < 2:
            return float("nan")
        mask = ~np.eye(n, dtype=bool)
        return float(c[mask].mean())
    return _rolling_apply_matrix(returns, window, fn).rename("avg_correlation")


def top_eigenvalue(returns: pd.DataFrame, window: int = 50) -> pd.Series:
    """Largest eigenvalue of the rolling correlation matrix.

    Equivalent to the explained variance of the first PCA mode -- another
    classic regime signal (Plerou et al., random matrix theory literature).
    """
    def fn(block: np.ndarray) -> float:
        c = np.corrcoef(block, rowvar=False)
        if not np.isfinite(c).all():
            return float("nan")
        w = np.linalg.eigvalsh(c)
        return float(w[-1])
    return _rolling_apply_matrix(returns, window, fn).rename("top_eigenvalue")


def equal_weight_drawdown(returns: pd.DataFrame, window: int = 50) -> pd.Series:
    """Recent (backward) drawdown of the equal-weighted basket. Pure-price baseline."""
    nav = (1 + returns.mean(axis=1)).cumprod()
    return (nav / nav.rolling(window).max() - 1.0).rename("recent_drawdown")
