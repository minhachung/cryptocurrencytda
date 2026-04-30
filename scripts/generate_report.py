"""Convert results/*.csv into a markdown summary.

Used by CI to populate $GITHUB_STEP_SUMMARY so metrics are visible directly
on the workflow run page (no artifact download required).

Usage:
    python scripts/generate_report.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without tabulate."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].apply(lambda v: float_fmt.format(v) if pd.notna(v) else "")
        else:
            df[c] = df[c].astype(str)
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(row[c] for c in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="-", help="output file or '-' for stdout")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    parts = ["# TDA Pipeline Results\n"]

    summary_path = rd / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        cfg = s.get("config", {})
        parts.append("**Config:** "
                     f"window={cfg.get('window')}, horizon={cfg.get('horizon')}, "
                     f"label_threshold={cfg.get('drawdown_label_threshold')}, "
                     f"event_threshold={cfg.get('event_drawdown_threshold')}, "
                     f"synthetic={cfg.get('synthetic')}")
        parts.append(f"**Sample:** n={s.get('n_observations')}, "
                     f"positive_rate={s.get('positive_rate'):.3f}, "
                     f"crash_events={s.get('n_crash_events')}\n")

    in_sample = rd / "in_sample_metrics.csv"
    if in_sample.exists():
        df = pd.read_csv(in_sample)
        keep = ["name", "roc_auc", "pr_auc", "best_f1",
                "precision_at_best", "recall_at_best", "n_positive", "n_total"]
        df = df[[c for c in keep if c in df.columns]].sort_values("roc_auc", ascending=False)
        parts.append("## In-sample (whole sample, F1-optimal threshold)\n")
        parts.append(md_table(df))
        parts.append("")

    wf = rd / "walk_forward_metrics.csv"
    if wf.exists():
        df = pd.read_csv(wf)
        agg = (df.groupby("signal")[["test_roc_auc", "test_f1",
                                     "test_precision", "test_recall"]]
                 .mean().round(3).reset_index()
                 .sort_values("test_roc_auc", ascending=False))
        parts.append("## Walk-forward OOS (mean across folds)\n")
        parts.append(md_table(agg))
        parts.append("")

    ev = rd / "event_metrics.csv"
    if ev.exists():
        df = pd.read_csv(ev)
        keep = ["signal", "n_hits", "n_events", "hit_rate",
                "mean_lead_days", "median_lead_days"]
        df = df[[c for c in keep if c in df.columns]].sort_values(
            "hit_rate", ascending=False)
        parts.append("## Event-level metrics (z >= 1.5 alarm, 30-day window)\n")
        parts.append(md_table(df))
        parts.append("")

    out = "\n".join(parts)
    if args.out == "-":
        print(out)
    else:
        Path(args.out).write_text(out)


if __name__ == "__main__":
    main()
