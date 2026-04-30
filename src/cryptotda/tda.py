"""TDA core: sliding-window point clouds and persistent homology.

Construction (following Gidea & Katz 2018):
    Given a returns matrix R of shape (T, N) with N assets, the point cloud
    at time t is the W consecutive rows R[t-W+1:t+1, :]. Each row is one day's
    cross-section of returns viewed as a point in R^N. The persistence diagram
    of the Vietoris-Rips filtration on this point cloud summarizes how
    geometrically clustered / loopy the recent return co-movement is.

Why this captures crash precursors:
    In calm regimes coins move semi-independently -- the point cloud is roughly
    a W-point i.i.d. sample from a high-dimensional Gaussian and tends to
    contain non-trivial 1-cycles. As correlations spike toward a crash, returns
    collapse onto a near 1-D subspace ("everything moves together"), the cloud
    flattens, and 1-cycles are killed (low max-persistence in H_1) while H_0
    shows long-lived components (heavy tails). Total persistence and the
    landscape norm respond to *both* effects simultaneously and earlier than
    rolling volatility.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm


def sliding_window_point_clouds(
    returns: pd.DataFrame,
    window: int = 50,
    step: int = 1,
) -> Iterator[tuple[pd.Timestamp, np.ndarray]]:
    """Yield (end_timestamp, point_cloud) pairs.

    point_cloud has shape (window, n_assets). end_timestamp is the date of the
    *last* row included -- so the signal at time t uses only data up to t.
    """
    arr = returns.to_numpy()
    idx = returns.index
    if len(arr) < window:
        return
    for t in range(window - 1, len(arr), step):
        yield idx[t], arr[t - window + 1: t + 1]


def persistence_diagrams(
    point_cloud: np.ndarray,
    maxdim: int = 1,
    thresh: float | None = None,
) -> list[np.ndarray]:
    """Vietoris-Rips persistence diagrams up to dimension `maxdim`.

    Returns a list [H_0, H_1, ...]. The infinite H_0 bar is dropped (its death
    is the connected-component-merging time, which is fixed by the choice of
    threshold and contains no information about the data).
    """
    if not np.isfinite(point_cloud).all():
        # ripser silently returns garbage on NaNs -- guard explicitly
        raise ValueError("point cloud contains NaN/inf")
    kwargs = {"maxdim": maxdim}
    if thresh is not None:
        kwargs["thresh"] = thresh
    result = ripser(point_cloud, **kwargs)
    diagrams = result["dgms"]
    # Drop infinite H_0 bar
    h0 = diagrams[0]
    finite_h0 = h0[np.isfinite(h0[:, 1])]
    diagrams = [finite_h0] + list(diagrams[1:])
    return diagrams


def compute_diagrams_over_time(
    returns: pd.DataFrame,
    window: int = 50,
    step: int = 1,
    maxdim: int = 1,
    thresh: float | None = None,
    progress: bool = True,
) -> tuple[pd.DatetimeIndex, list[list[np.ndarray]]]:
    """Sliding-window persistence diagrams for every window.

    Returns
    -------
    timestamps : DatetimeIndex of length M (one per window)
    diagrams_list : list of length M, each item is [H_0, H_1, ...]
    """
    pcs = list(sliding_window_point_clouds(returns, window=window, step=step))
    iterator = tqdm(pcs, desc="persistence diagrams") if progress else pcs
    timestamps = []
    diagrams_list = []
    for ts, pc in iterator:
        diagrams_list.append(persistence_diagrams(pc, maxdim=maxdim, thresh=thresh))
        timestamps.append(ts)
    return pd.DatetimeIndex(timestamps), diagrams_list
