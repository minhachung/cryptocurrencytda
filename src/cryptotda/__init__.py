"""Persistent homology of cryptocurrency markets.

Pipeline:
    data  ->  log-returns  ->  sliding window point cloud
          ->  Vietoris-Rips persistence diagram
          ->  persistence landscape  ->  L^p norm signal
          ->  z-score detector  ->  crash-prediction metrics
"""

from .tda import sliding_window_point_clouds, persistence_diagrams
from .landscapes import persistence_landscape, landscape_norm, total_persistence
from .detector import zscore_signal, threshold_signal, lead_time
from .crashes import label_crash_periods, detect_crash_events
from .baselines import realized_volatility, average_correlation, top_eigenvalue
from .validation import evaluate_signal, walk_forward_evaluation

__all__ = [
    "sliding_window_point_clouds",
    "persistence_diagrams",
    "persistence_landscape",
    "landscape_norm",
    "total_persistence",
    "zscore_signal",
    "threshold_signal",
    "lead_time",
    "label_crash_periods",
    "detect_crash_events",
    "realized_volatility",
    "average_correlation",
    "top_eigenvalue",
    "evaluate_signal",
    "walk_forward_evaluation",
]
