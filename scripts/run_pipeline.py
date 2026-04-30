"""End-to-end pipeline: data -> persistence -> signal -> evaluation -> figures.

Usage:
    # On real data (after fetch_data.py):
    python scripts/run_pipeline.py --prices data/prices.parquet

    # On reproducible synthetic data (no network needed):
    python scripts/run_pipeline.py --synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cryptotda.data import log_returns, synthetic_market
from cryptotda.tda import compute_diagrams_over_time
from cryptotda.landscapes import landscape_signal_series, total_persistence_series
from cryptotda.crashes import label_crash_periods, detect_crash_events, benchmark_series
from cryptotda.baselines import realized_volatility, average_correlation, top_eigenvalue
from cryptotda.detector import zscore_signal
from cryptotda.validation import evaluate_signal, walk_forward_evaluation, event_metrics
from cryptotda.visualize import (
    plot_signal_with_crashes,
    plot_signal_comparison,
    plot_roc_curves,
)


def build_signals(returns: pd.DataFrame, window: int):
    """Compute the TDA + baseline signals as time-aligned pd.Series."""
    print(f"Computing persistence diagrams (window={window}, n_windows={len(returns) - window + 1})...")
    timestamps, dgms = compute_diagrams_over_time(returns, window=window, maxdim=1, progress=True)

    # H1 captures loops / co-movement geometry; H0 captures cluster collapse
    tda_h1_l2 = pd.Series(
        landscape_signal_series(dgms, homology_dim=1, p=2, num_landscapes=5, resolution=200),
        index=timestamps, name="tda_h1_landscape_l2",
    )
    tda_h0_total = pd.Series(
        total_persistence_series(dgms, homology_dim=0, power=2),
        index=timestamps, name="tda_h0_total_sq",
    )
    # Topological-anomaly signal: |z-score| picks up deviation in either
    # direction (loops killed = stress; loops growing = also stress). This is
    # the right framing because topology can change in either sense and the
    # detector should not assume a fixed sign a priori.
    z_h1 = zscore_signal(tda_h1_l2, lookback=252).abs()
    z_h0 = zscore_signal(tda_h0_total, lookback=252).abs()
    tda_anomaly = pd.concat([z_h1, z_h0], axis=1).max(axis=1).rename("tda_anomaly")

    rv = realized_volatility(returns, window=window)
    ac = average_correlation(returns, window=window)
    te = top_eigenvalue(returns, window=window)

    return {
        "tda_h1_l2": tda_h1_l2,
        "tda_h0_total": tda_h0_total,
        "tda_anomaly": tda_anomaly,
        "realized_vol": rv,
        "avg_correlation": ac,
        "top_eigenvalue": te,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=None, help="CSV file from fetch_data.py")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic data (no network needed)")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--drawdown", type=float, default=-0.20)
    ap.add_argument("--event-drawdown", type=float, default=-0.30)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True, parents=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(exist_ok=True, parents=True)

    # ----- Load data -----
    if args.synthetic or args.prices is None:
        print("Generating synthetic market with embedded crash regimes...")
        prices, returns, planted_crashes = synthetic_market(
            n_assets=15, n_days=2500, seed=args.seed,
        )
        print(f"  planted crashes at: {[d.date() for d in planted_crashes]}")
        benchmark_name = None
    else:
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        returns = log_returns(prices)
        benchmark_name = "bitcoin"
        planted_crashes = None

    print(f"Returns shape: {returns.shape}, "
          f"date range {returns.index.min().date()} -> {returns.index.max().date()}")

    # ----- Crashes & labels -----
    crash_dates = detect_crash_events(
        prices, drawdown_threshold=args.event_drawdown, benchmark=benchmark_name,
    )
    print(f"Detected crash events: {[d.date() for d in crash_dates]}")
    if planted_crashes is not None:
        print(f"  (planted truth was {[d.date() for d in planted_crashes]})")
    y = label_crash_periods(
        prices, horizon=args.horizon, drawdown_threshold=args.drawdown,
        benchmark=benchmark_name,
    )
    print(f"Positive class rate (>= {args.drawdown:.0%} drawdown within {args.horizon}d): {y.mean():.3f}")

    # ----- Build signals -----
    signals = build_signals(returns, window=args.window)

    # Align signals to label and save
    aligned = pd.DataFrame(signals).reindex(y.index)
    aligned["label"] = y
    aligned.to_csv(results_dir / "signals_and_label.csv")

    # ----- In-sample evaluation -----
    print("\n--- In-sample ROC / PR / F1 ---")
    eval_rows = []
    for name, s in signals.items():
        r = evaluate_signal(s, y, name=name)
        eval_rows.append(r.__dict__)
        print(f"{name:25s}  ROC AUC={r.roc_auc:.3f}  PR AUC={r.pr_auc:.3f}  "
              f"F1*={r.best_f1:.3f}  P={r.precision_at_best:.3f}  R={r.recall_at_best:.3f}")
    pd.DataFrame(eval_rows).to_csv(results_dir / "in_sample_metrics.csv", index=False)

    # ----- Walk-forward (out-of-sample) -----
    print("\n--- Walk-forward OOS evaluation ---")
    wf_rows = []
    for name, s in signals.items():
        df = walk_forward_evaluation(s, y, name=name)
        if len(df):
            wf_rows.append(df)
            mean_auc = df["test_roc_auc"].mean()
            mean_f1 = df["test_f1"].mean()
            mean_p = df["test_precision"].mean()
            mean_r = df["test_recall"].mean()
            print(f"{name:25s}  mean OOS AUC={mean_auc:.3f}  F1={mean_f1:.3f}  "
                  f"P={mean_p:.3f}  R={mean_r:.3f}  ({len(df)} folds)")
    if wf_rows:
        pd.concat(wf_rows, ignore_index=True).to_csv(
            results_dir / "walk_forward_metrics.csv", index=False
        )

    # ----- Event-level hit rate / lead time -----
    print("\n--- Event-level metrics ---")
    ev_rows = []
    for name, s in signals.items():
        ev = event_metrics(s, crash_dates, threshold=1.5, lookback=30, name=name)
        ev_rows.append(ev)
        print(f"{name:25s}  hits={ev['n_hits']}/{ev['n_events']}  "
              f"mean_lead={ev['mean_lead_days']:.1f}d  median_lead={ev['median_lead_days']:.1f}d")
    pd.DataFrame(ev_rows).to_csv(results_dir / "event_metrics.csv", index=False)

    # ----- Figures -----
    print("\n--- Generating figures ---")
    bench = benchmark_series(prices, benchmark=benchmark_name)
    plot_signal_comparison(
        signals, bench, crash_dates,
        out_path=figures_dir / "signal_comparison.png",
    )
    plot_roc_curves(
        [(n, s, y) for n, s in signals.items()],
        out_path=figures_dir / "roc_curves.png",
    )
    plot_signal_with_crashes(
        signals["tda_h1_l2"], bench, crash_dates,
        title="TDA H1 landscape L^2 norm vs benchmark price",
        out_path=figures_dir / "tda_h1_signal.png",
    )

    # ----- Headline summary -----
    summary = {
        "config": {
            "window": args.window,
            "horizon": args.horizon,
            "drawdown_label_threshold": args.drawdown,
            "event_drawdown_threshold": args.event_drawdown,
            "synthetic": bool(args.synthetic or args.prices is None),
        },
        "n_observations": int(len(y)),
        "positive_rate": float(y.mean()),
        "n_crash_events": len(crash_dates),
        "in_sample": eval_rows,
        "events": ev_rows,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote summary -> {results_dir / 'summary.json'}")
    print(f"Figures -> {figures_dir}")


if __name__ == "__main__":
    main()
