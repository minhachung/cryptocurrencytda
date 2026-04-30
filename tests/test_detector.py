"""Detector + crash-labelling tests."""

import numpy as np
import pandas as pd
import pytest

from cryptotda.detector import zscore_signal, threshold_signal, lead_time, hit_rate
from cryptotda.crashes import forward_drawdown, label_crash_periods, detect_crash_events


def test_zscore_is_causal():
    """z at time t must not depend on values at time >= t."""
    rng = np.random.default_rng(0)
    base = pd.Series(rng.standard_normal(500), index=pd.date_range("2020-01-01", periods=500))
    z1 = zscore_signal(base, lookback=100, min_periods=50)
    # Mutating the future should not change the past
    perturbed = base.copy()
    perturbed.iloc[300:] = 999.0
    z2 = zscore_signal(perturbed, lookback=100, min_periods=50)
    pd.testing.assert_series_equal(z1.iloc[:300], z2.iloc[:300])


def test_threshold_and_lead_time():
    idx = pd.date_range("2020-01-01", periods=100)
    z = pd.Series(np.zeros(100), index=idx)
    z.iloc[40] = 3.0   # alarm 10 days before crash
    crash = idx[50]
    alarms = threshold_signal(z, threshold=1.5)
    leads = lead_time(alarms, [crash], max_lookback=30)
    assert leads == [10]
    assert hit_rate(alarms, [crash], max_lookback=30) == 1.0


def test_forward_drawdown_picks_up_drop():
    idx = pd.date_range("2020-01-01", periods=20)
    p = pd.Series([100.0] * 10 + [70.0] * 10, index=idx)  # 30% drop on day 10
    dd = forward_drawdown(p, horizon=5)
    # On day 9, looking forward 5 days, drawdown reaches -30%
    assert dd.iloc[9] == pytest.approx(-0.30, abs=1e-9)
    # After the drop and a flat period, no further drawdown
    assert dd.iloc[15] == pytest.approx(0.0, abs=1e-9)


def test_label_crash_periods_marks_pre_crash_window():
    idx = pd.date_range("2020-01-01", periods=100)
    p = pd.Series(np.concatenate([np.full(50, 100.0), np.full(50, 70.0)]), index=idx)
    df = p.to_frame("bitcoin")
    y = label_crash_periods(df, horizon=10, drawdown_threshold=-0.20)
    # Days 40..49 should label as positive (within 10 days of the drop)
    assert y.iloc[40:50].sum() >= 1
    # Day 0 is far from any drop and should be 0
    assert y.iloc[0] == 0


def test_detect_crash_events_finds_peak():
    idx = pd.date_range("2020-01-01", periods=200)
    # Ramp up to peak at index 100, then plunge
    p = np.concatenate([np.linspace(50, 100, 100), np.linspace(100, 60, 100)])
    df = pd.Series(p, index=idx).to_frame("bitcoin")
    events = detect_crash_events(df, drawdown_threshold=-0.30)
    assert len(events) == 1
    # Peak is at index 99 (end of the ramp)
    assert events[0] == idx[99]
