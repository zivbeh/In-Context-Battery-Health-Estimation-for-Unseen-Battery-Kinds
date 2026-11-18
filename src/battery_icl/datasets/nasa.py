"""NASA Randomized Battery Usage dataset: 28 LCO 18650 cells (RW1 to RW28).

One .mat file per cell holds a long list of steps. Each step has a text
comment that says what it is, for example "discharge (random walk)" or
"reference discharge". Ageing use is every random-walk step; the reference
capacity is the charge removed in each reference discharge (1 A, 4.2 V to 3.2 V).

NASA records discharge as positive current; we flip it to charge-positive.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from ..standardize import Cell
from .common import discharged_ah

CUTOFF_V = 3.2          # a reference discharge runs down to this voltage
CUTOFF_MARGIN_V = 0.1   # one that stops higher was aborted and is not a capacity


def _cell_files(raw_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in raw_dir.rglob("RW*.mat")}


def list_cells(raw_dir: Path) -> list[str]:
    return sorted(_cell_files(raw_dir), key=lambda name: int(name[2:]))


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    mat = scipy.io.loadmat(_cell_files(raw_dir)[cell_id], squeeze_me=True, struct_as_record=False)
    steps = np.atleast_1d(mat["data"].step)

    usage_parts, reference_rows = [], []
    for step in steps:
        time_s = np.atleast_1d(step.time).astype(float)
        current_a = -np.atleast_1d(step.current).astype(float)
        if "random walk" in step.comment:
            usage_parts.append(pd.DataFrame({
                "time_s": time_s,
                "current_a": current_a,
                "voltage_v": np.atleast_1d(step.voltage).astype(float),
                "temperature_c": np.atleast_1d(step.temperature).astype(float),
            }))
        elif step.comment == "reference discharge" and _reached_cutoff(step):
            reference_rows.append({"time_s": time_s[0], "capacity_ah": discharged_ah(time_s, current_a)})

    return Cell("nasa", cell_id, pd.concat(usage_parts, ignore_index=True), pd.DataFrame(reference_rows))


def _reached_cutoff(step) -> bool:
    return np.atleast_1d(step.voltage)[-1] <= CUTOFF_V + CUTOFF_MARGIN_V
