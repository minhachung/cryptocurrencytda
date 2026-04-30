"""Validation: turn (signal, label) into honest accuracy numbers.

Three layers of evaluation, in order of stringency:

1. evaluate_signal -- ROC AUC, PR AUC, F1 at best threshold. Treats the whole
   sample as one. Useful for sanity but suffers from look-ahead in threshold
   selection.

2. walk_forward_evaluation -- rolling-origin out-of-sample evaluation. The
   threshold is fit on a training window and then applied to a future test
   window. Reproduces the actual deployment setting.

3. event_metrics -- crash-event-level hit rate and mean lead time. Aggregate
   metrics in (1) and (2) can be high while still missing the events that
   matter, so we always report event metrics alongside.

We compare every signal (TDA + baselines) on identical splits so the
comparison is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from .detector import zscore_signal, threshold_signal, hit_rate, lead_time


@dataclass
class EvalResult:
    name: str
    roc_auc: float
    pr_auc: float
    best_f1: float
    best_threshold: float
    precision_at_best: float
    recall_at_best: float
    n_positive: int
    n_total: int


def _align(signal: pd.Series, label: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([signal.rename("s"), label.rename("y")], axis=1).dropna()
    return df["s"].to_numpy(), df["y"].to_numpy().astype(int)


def evaluate_signal(signal: pd.Series, label: pd.Series, name: str = "signal") -> EvalResult:
    """In-sample evaluation. Best threshold is chosen on the same data."""
    s, y = _align(signal, label)
    if len(s) == 0 or y.sum() == 0 or y.sum() == len(y):
        return EvalResult(name, float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"),
                          int(y.sum()), int(len(y)))
    roc = roc_auc_score(y, s)
    pr = average_precision_score(y, s)
    fpr, tpr, thr = roc_curve(y, s)
    f1s = []
    for t in thr:
        pred = (s >= t).astype(int)
        f1s.append(f1_score(y, pred, zero_division=0))
    f1s = np.array(f1s)
    best = int(np.argmax(f1s))
    pred = (s >= thr[best]).astype(int)
    return EvalResult(
        name=name,
        roc_auc=float(roc),
        pr_auc=float(pr),
        best_f1=float(f1s[best]),
        best_threshold=float(thr[best]),
        precision_at_best=float(precision_score(y, pred, zero_division=0)),
        recall_at_best=float(recall_score(y, pred, zero_division=0)),
        n_positive=int(y.sum()),
        n_total=int(len(y)),
    )


def walk_forward_evaluation(
    signal: pd.Series,
    label: pd.Series,
    train_size: int = 504,         # ~2 years of daily data
    test_size: int = 126,          # ~6 months
    step: int = 126,
    name: str = "signal",
) -> pd.DataFrame:
    """Rolling-origin out-of-sample evaluation.

    For each fold:
        - fit best threshold on train slice (maximize F1)
        - apply threshold to test slice
        - record OOS precision / recall / F1 / ROC AUC on test
    """
    s, y = _align(signal, label)
    rows = []
    n = len(s)
    start = 0
    while start + train_size + test_size <= n:
        s_tr, y_tr = s[start: start + train_size], y[start: start + train_size]
        s_te, y_te = s[start + train_size: start + train_size + test_size], y[start + train_size: start + train_size + test_size]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            start += step
            continue
        # Best F1 threshold from train
        fpr, tpr, thr = roc_curve(y_tr, s_tr)
        best_t = float(thr[np.argmax([f1_score(y_tr, (s_tr >= t).astype(int), zero_division=0) for t in thr])])
        pred_te = (s_te >= best_t).astype(int)
        roc = roc_auc_score(y_te, s_te) if y_te.sum() not in (0, len(y_te)) else float("nan")
        rows.append({
            "fold_start": start,
            "fold_end": start + train_size + test_size,
            "threshold": best_t,
            "test_roc_auc": roc,
            "test_precision": precision_score(y_te, pred_te, zero_division=0),
            "test_recall": recall_score(y_te, pred_te, zero_division=0),
            "test_f1": f1_score(y_te, pred_te, zero_division=0),
            "n_test_positive": int(y_te.sum()),
        })
        start += step
    df = pd.DataFrame(rows)
    df["signal"] = name
    return df


def event_metrics(
    signal: pd.Series,
    crash_dates: list[pd.Timestamp],
    threshold: float = 1.5,
    lookback: int = 30,
    zscore_lookback: int = 252,
    name: str = "signal",
) -> dict:
    """Crash-event-level summary: hit rate + mean / median lead time."""
    z = zscore_signal(signal, lookback=zscore_lookback)
    alarms = threshold_signal(z, threshold=threshold)
    hr = hit_rate(alarms, crash_dates, max_lookback=lookback)
    lts = lead_time(alarms, crash_dates, max_lookback=lookback)
    finite = [x for x in lts if x == x]  # drop NaNs
    return {
        "signal": name,
        "threshold": threshold,
        "hit_rate": hr,
        "mean_lead_days": float(np.mean(finite)) if finite else float("nan"),
        "median_lead_days": float(np.median(finite)) if finite else float("nan"),
        "n_events": len(crash_dates),
        "n_hits": int(round(hr * len(crash_dates))) if crash_dates else 0,
    }
