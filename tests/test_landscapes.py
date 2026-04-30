"""Sanity tests for the landscape implementation.

Reference values come from analytic formulas:
  - Single triangle (b, d): peak height = (d - b) / 2 at x = (b + d) / 2.
  - L^1 of full landscape = (d - b)^2 / 4 (area of a single triangle).
  - For two non-overlapping triangles, L^p norms are additive.
"""

import numpy as np
import pytest

from cryptotda.landscapes import (
    persistence_landscape,
    landscape_norm,
    total_persistence,
    max_persistence,
)


def test_empty_diagram_returns_zeros():
    L, xs = persistence_landscape(np.zeros((0, 2)), num_landscapes=3, resolution=50)
    assert L.shape == (3, 50)
    assert (L == 0).all()


def test_single_triangle_peak_height():
    dgm = np.array([[0.0, 2.0]])
    L, xs = persistence_landscape(dgm, num_landscapes=1, resolution=2001, x_range=(0.0, 2.0))
    assert L[0].max() == pytest.approx(1.0, abs=1e-3)
    # Peak at midpoint
    assert xs[L[0].argmax()] == pytest.approx(1.0, abs=2e-3)


def test_l1_norm_equals_triangle_area():
    dgm = np.array([[0.0, 2.0]])
    L, xs = persistence_landscape(dgm, num_landscapes=1, resolution=4001, x_range=(0.0, 2.0))
    # Area = base * height / 2 = 2 * 1 / 2 = 1
    assert landscape_norm(L, xs, p=1.0) == pytest.approx(1.0, rel=1e-2)


def test_two_disjoint_triangles_additive_l1():
    dgm = np.array([[0.0, 2.0], [4.0, 6.0]])
    L, xs = persistence_landscape(dgm, num_landscapes=2, resolution=8001, x_range=(0.0, 6.0))
    # Two disjoint unit-area triangles -> L1 = 2
    assert landscape_norm(L, xs, p=1.0) == pytest.approx(2.0, rel=1e-2)


def test_total_persistence_and_max():
    dgm = np.array([[0.0, 1.0], [2.0, 5.0], [3.0, 4.0]])
    assert total_persistence(dgm, power=1) == pytest.approx(1 + 3 + 1)
    assert total_persistence(dgm, power=2) == pytest.approx(1 + 9 + 1)
    assert max_persistence(dgm) == pytest.approx(3.0)
