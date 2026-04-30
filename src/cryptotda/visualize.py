"""Diagnostic plots for the report / paper. Matplotlib only; no seaborn."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_persistence_diagram(diagram_h0: np.ndarray, diagram_h1: np.ndarray, ax=None, title: str = ""):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    pts = []
    if len(diagram_h0):
        pts.append(diagram_h0)
        ax.scatter(diagram_h0[:, 0], diagram_h0[:, 1], s=18, label="$H_0$", alpha=0.7)
    if len(diagram_h1):
        pts.append(diagram_h1)
        ax.scatter(diagram_h1[:, 0], diagram_h1[:, 1], s=18, label="$H_1$", alpha=0.7, marker="^")
    if pts:
        all_pts = np.vstack(pts)
        m = float(all_pts.max()) * 1.05 if len(all_pts) else 1.0
    else:
        m = 1.0
    ax.plot([0, m], [0, m], "k--", lw=0.8)
    ax.set_xlabel("birth"); ax.set_ylabel("death"); ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    return ax


def plot_signal_with_crashes(
    signal: pd.Series,
    benchmark: pd.Series,
    crash_dates: list[pd.Timestamp],
    threshold: float | None = None,
    title: str = "",
    out_path: Path | None = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].semilogy(benchmark.index, benchmark.values, lw=1.0, color="black")
    axes[0].set_ylabel("benchmark price (log)")
    axes[0].set_title(title)
    for c in crash_dates:
        for ax in axes:
            ax.axvline(c, color="crimson", lw=0.8, alpha=0.5, ls="--")
    axes[1].plot(signal.index, signal.values, lw=1.0, color="C0")
    axes[1].set_ylabel(signal.name or "signal")
    if threshold is not None:
        axes[1].axhline(threshold, color="C3", lw=0.8, ls=":")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_signal_comparison(
    signals: dict[str, pd.Series],
    benchmark: pd.Series,
    crash_dates: list[pd.Timestamp],
    out_path: Path | None = None,
):
    n = len(signals) + 1
    fig, axes = plt.subplots(n, 1, figsize=(11, 2 * n), sharex=True)
    axes[0].semilogy(benchmark.index, benchmark.values, lw=1.0, color="black")
    axes[0].set_ylabel("benchmark")
    for c in crash_dates:
        for ax in axes:
            ax.axvline(c, color="crimson", lw=0.8, alpha=0.5, ls="--")
    for i, (name, s) in enumerate(signals.items(), start=1):
        axes[i].plot(s.index, s.values, lw=1.0)
        axes[i].set_ylabel(name)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    signal_label_pairs: list[tuple[str, pd.Series, pd.Series]],
    out_path: Path | None = None,
):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, s, y in signal_label_pairs:
        df = pd.concat([s.rename("s"), y.rename("y")], axis=1).dropna()
        if df["y"].sum() in (0, len(df)):
            continue
        fpr, tpr, _ = roc_curve(df["y"], df["s"])
        auc = roc_auc_score(df["y"], df["s"])
        ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC: predicting forward 30-day drawdown >= 20%")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
