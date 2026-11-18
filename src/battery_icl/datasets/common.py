"""Small helpers shared by the dataset parsers."""

import numpy as np
import pandas as pd


def find_runs(mask: np.ndarray, time_s: np.ndarray, min_duration_s: float) -> list[tuple[int, int]]:
    """Index ranges [start, stop) where `mask` stays true for at least min_duration_s."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    runs = zip(edges[::2], edges[1::2])
    return [(a, b) for a, b in runs if time_s[b - 1] - time_s[a] >= min_duration_s]


def discharged_ah(time_s: np.ndarray, current_a: np.ndarray) -> float:
    """Charge removed over a stretch of data, in amp-hours (charge-positive current)."""
    return float(-np.trapezoid(current_a, time_s) / 3600.0)


def drop_spans(table: pd.DataFrame, spans: list[tuple[float, float]]) -> pd.DataFrame:
    """Remove the rows whose time falls inside any (start, end) span."""
    keep = np.ones(len(table), dtype=bool)
    for start, end in spans:
        keep &= ~table["time_s"].between(start, end).to_numpy()
    return table[keep]
