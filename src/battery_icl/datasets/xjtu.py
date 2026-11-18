"""XJTU dataset (Wang et al. 2024), Batch 5: 8 LISHEN NCM523 cells on random-walk discharge.

One .mat file per cell holds one record per cycle, sampled at 1 Hz. A text
description marks the capacity-test cycles ("... [test capacity]"): a 1C
charge and a 1C discharge to 2.5 V, run once every ten random-walk cycles
from cycle 21 on. The first 20 cycles have no test, so health is relative to
the capacity at cycle 21.

The other five batches hold the same kind of cell on constant or stepped
currents and are not parsed here.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from ..standardize import Cell
from .common import discharged_ah

BATCH_DIR = "Battery Dataset/Batch-5"


def list_cells(raw_dir: Path) -> list[str]:
    return sorted(p.stem for p in (raw_dir / BATCH_DIR).glob("RW_battery-*.mat"))


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    mat = scipy.io.loadmat(raw_dir / BATCH_DIR / f"{cell_id}.mat", squeeze_me=True, struct_as_record=False)
    cycles = np.atleast_1d(mat["data"])
    test_start = _start_time(cycles[0])

    usage_parts, reference_rows = [], []
    for cycle in cycles:
        current_a = np.atleast_1d(cycle.current_A).astype(float)   # already charge-positive
        # Samples are one second apart; the file's own time column restarts at every step.
        time_s = (_start_time(cycle) - test_start).total_seconds() + np.arange(len(current_a), dtype=float)

        if "test capacity" in str(cycle.description):
            discharge = np.minimum(current_a, 0.0)
            reference_rows.append({"time_s": time_s[0], "capacity_ah": discharged_ah(time_s, discharge)})
        else:
            usage_parts.append(pd.DataFrame({
                "time_s": time_s,
                "current_a": current_a,
                "voltage_v": np.atleast_1d(cycle.voltage_V).astype(float),
                "temperature_c": np.atleast_1d(cycle.temperature_C).astype(float),
            }))

    return Cell("xjtu", cell_id, pd.concat(usage_parts, ignore_index=True), pd.DataFrame(reference_rows))


def _start_time(cycle) -> datetime:
    return datetime.strptime(str(np.atleast_1d(cycle.system_time)[0]), "%Y-%m-%d,%H:%M:%S")
