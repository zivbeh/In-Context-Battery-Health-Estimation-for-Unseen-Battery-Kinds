"""Cut standardized cell tables into fixed-length clips with a health label.

A clip is a window of (current_c, voltage_norm) taken from one uninterrupted
block of ageing use. Its label is the cell's health at the middle of the
window. Windows that are mostly rest carry no information about the cell and
are dropped.
"""

import numpy as np
import pandas as pd

CHANNELS = ("current_c", "voltage_norm")
REST_C_RATE = 0.02   # below this the cell counts as resting


def cut_clips(table: pd.DataFrame, clip_len: int, min_active: float = 0.5):
    """Return (clips, health, time_s) for one cell.

    clips has shape (n, 2, clip_len); health and time_s have shape (n,).
    Windows do not overlap and never cross a block boundary.
    """
    clips, health, time_s = [], [], []
    for _, block in table.groupby("block"):
        signal = block[list(CHANNELS)].to_numpy(dtype=np.float32)
        label = block["health"].to_numpy()
        for start in range(0, len(block) - clip_len + 1, clip_len):
            window = signal[start : start + clip_len]
            middle = start + clip_len // 2
            active = np.mean(np.abs(window[:, 0]) > REST_C_RATE)
            if np.isnan(label[middle]) or active < min_active:
                continue
            clips.append(window.T)
            health.append(label[middle])
            time_s.append(block["time_s"].iloc[middle])
    if not clips:
        return np.empty((0, len(CHANNELS), clip_len), np.float32), np.empty(0, np.float32), np.empty(0)
    return np.stack(clips), np.asarray(health, np.float32), np.asarray(time_s)


def spread_evenly(n_available: int, n_wanted: int) -> np.ndarray:
    """Indices of at most n_wanted items spread evenly over n_available.

    Used to cap the clips per cell so that long-lived cells do not dominate,
    while still covering the cell's whole life.
    """
    if n_available <= n_wanted:
        return np.arange(n_available)
    return np.linspace(0, n_available - 1, n_wanted).round().astype(int)
