"""Persistence landscapes and scalar summaries.

A persistence landscape (Bubenik 2015) turns a persistence diagram into a
sequence of functions lambda_k : R -> R that live in a vector space of
real-valued functions. This makes them easy to feed into statistics and ML.

For each (b, d) in the diagram define the triangle
    f_{b,d}(x) = max(0, min(x - b, d - x)).
The k-th landscape is
    lambda_k(x) = k-th largest value of {f_{b,d}(x) : (b,d) in dgm}.

Scalar summaries used in this project:
    - L^p norm of the first K landscapes on a fixed grid.
    - Total persistence: sum of (d - b).
    - Total squared persistence: sum of (d - b)^2.
    - Max persistence: max(d - b).
"""

from __future__ import annotations

import numpy as np


def persistence_landscape(
    diagram: np.ndarray,
    num_landscapes: int = 5,
    resolution: int = 200,
    x_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the first `num_landscapes` landscape functions on a grid.

    Parameters
    ----------
    diagram : (n, 2) array of (birth, death) pairs. May be empty.
    num_landscapes : K
    resolution : grid resolution
    x_range : (x_min, x_max). If None, taken from the diagram. **For
        comparing landscapes across time you must pass a fixed x_range.**

    Returns
    -------
    landscapes : (num_landscapes, resolution)
    xs : (resolution,)
    """
    if x_range is None:
        if len(diagram) == 0:
            return np.zeros((num_landscapes, resolution)), np.linspace(0, 1, resolution)
        x_min = float(diagram[:, 0].min())
        x_max = float(diagram[:, 1].max())
        if x_max <= x_min:
            x_max = x_min + 1.0
    else:
        x_min, x_max = x_range
    xs = np.linspace(x_min, x_max, resolution)
    landscapes = np.zeros((num_landscapes, resolution))
    if len(diagram) == 0:
        return landscapes, xs

    b = diagram[:, 0][None, :]   # (1, n)
    d = diagram[:, 1][None, :]   # (1, n)
    x = xs[:, None]              # (resolution, 1)
    triangles = np.maximum(0.0, np.minimum(x - b, d - x))  # (resolution, n)
    # Sort each row descending; take top K columns
    triangles_sorted = -np.sort(-triangles, axis=1)
    k = min(num_landscapes, triangles_sorted.shape[1])
    landscapes[:k, :] = triangles_sorted[:, :k].T
    return landscapes, xs


def landscape_norm(
    landscapes: np.ndarray,
    xs: np.ndarray,
    p: float = 2.0,
) -> float:
    """L^p norm of the (multi-)landscape: (sum_k integral |lambda_k|^p)^{1/p}."""
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    integrand = np.abs(landscapes) ** p
    total = integrand.sum() * dx
    return float(total ** (1.0 / p))


def total_persistence(diagram: np.ndarray, power: float = 1.0) -> float:
    """sum (d - b)^power. power=1 is total persistence; power=2 is total squared."""
    if len(diagram) == 0:
        return 0.0
    return float(np.sum((diagram[:, 1] - diagram[:, 0]) ** power))


def max_persistence(diagram: np.ndarray) -> float:
    if len(diagram) == 0:
        return 0.0
    return float(np.max(diagram[:, 1] - diagram[:, 0]))


def landscape_signal_series(
    diagrams_list: list[list[np.ndarray]],
    homology_dim: int = 1,
    p: float = 2.0,
    num_landscapes: int = 5,
    resolution: int = 200,
    x_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute the L^p landscape norm at every timestep for a chosen H_k.

    If x_range is None we infer a global one from all diagrams so the norms
    are comparable across time -- this is the right default.
    """
    if x_range is None:
        all_births, all_deaths = [], []
        for dgms in diagrams_list:
            if homology_dim < len(dgms) and len(dgms[homology_dim]) > 0:
                all_births.append(dgms[homology_dim][:, 0].min())
                all_deaths.append(dgms[homology_dim][:, 1].max())
        if not all_births:
            return np.zeros(len(diagrams_list))
        x_range = (float(min(all_births)), float(max(all_deaths)))
        if x_range[1] <= x_range[0]:
            x_range = (x_range[0], x_range[0] + 1.0)

    out = np.zeros(len(diagrams_list))
    for i, dgms in enumerate(diagrams_list):
        if homology_dim >= len(dgms):
            continue
        L, xs = persistence_landscape(
            dgms[homology_dim],
            num_landscapes=num_landscapes,
            resolution=resolution,
            x_range=x_range,
        )
        out[i] = landscape_norm(L, xs, p=p)
    return out


def total_persistence_series(
    diagrams_list: list[list[np.ndarray]],
    homology_dim: int = 1,
    power: float = 2.0,
) -> np.ndarray:
    """Scalar summary that does not need a fixed grid (range-independent)."""
    out = np.zeros(len(diagrams_list))
    for i, dgms in enumerate(diagrams_list):
        if homology_dim >= len(dgms):
            continue
        out[i] = total_persistence(dgms[homology_dim], power=power)
    return out
