"""TDA-pipeline tests on small, hand-controlled inputs."""

import numpy as np
import pandas as pd
import pytest

from cryptotda.tda import (
    sliding_window_point_clouds,
    persistence_diagrams,
    compute_diagrams_over_time,
)


def test_sliding_windows_have_correct_shape():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((100, 5)),
                      index=pd.date_range("2020-01-01", periods=100))
    windows = list(sliding_window_point_clouds(df, window=20))
    # First window ends at index 19, last at 99 -> 81 windows
    assert len(windows) == 81
    for ts, pc in windows:
        assert pc.shape == (20, 5)


def test_persistence_of_circle_has_one_long_h1_bar():
    """A clean S^1 sample should produce exactly one long-lived H1 feature."""
    n = 60
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pc = np.column_stack([np.cos(theta), np.sin(theta)]) + 1e-3 * np.random.default_rng(0).standard_normal((n, 2))
    dgms = persistence_diagrams(pc, maxdim=1)
    h1 = dgms[1]
    persistences = h1[:, 1] - h1[:, 0]
    # Largest H1 bar should dwarf the rest
    assert persistences.max() > 0.5
    others = np.sort(persistences)[:-1]
    assert (others < 0.1).all()


def test_no_h1_for_collapsed_point_cloud():
    """All points equal => zero H1 features (collapsed cluster)."""
    pc = np.zeros((30, 4))
    dgms = persistence_diagrams(pc, maxdim=1)
    assert len(dgms[1]) == 0


def test_compute_diagrams_over_time_lengths_match():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((80, 4)),
                      index=pd.date_range("2020-01-01", periods=80))
    ts, dgms = compute_diagrams_over_time(df, window=30, progress=False)
    assert len(ts) == len(dgms) == 51
    for d in dgms:
        assert isinstance(d, list) and len(d) == 2  # [H0, H1]
