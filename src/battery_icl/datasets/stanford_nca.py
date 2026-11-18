"""Stanford dynamic-cycling dataset (Geslin et al. 2025): 92 NCA cells.

The authors publish current already as a C-rate and capacity already divided
by the nominal capacity, so this kind uses a nominal capacity of 1 in
configs/datasets.yaml.

Reference capacities come from diagnostic_capacities.pkl, one row per
diagnostic, keyed by the cycle number at which the diagnostic starts. A
diagnostic occupies that cycle number and the next one in the raw file. We use
the C/2 constant-current discharge capacity, the rate closest to the reference
tests of the other datasets.

protocol_mapping_dic.json says which profile each cell ran (constant current,
periodic, synthetic, or recorded driving). All cells are parsed here; choosing
which profiles count as "driving" is left to the experiment.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..standardize import Cell

CAPACITY_COLUMN = "C/2 discharge CC Capacity (normalized)"
RAW_COLUMNS = {
    "Test (Sec)": "time_s",
    "Normalized Current (C-rate)": "current_a",   # charge-positive, in units of C
    "Volts": "voltage_v",
    "Cyc#": "cycle",
}


def list_cells(raw_dir: Path) -> list[str]:
    return sorted(p.stem.removeprefix("raw_data_") for p in raw_dir.glob("raw_data_cell_*.csv"))


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    raw = pd.read_csv(raw_dir / f"raw_data_{cell_id}.csv", usecols=list(RAW_COLUMNS)).rename(columns=RAW_COLUMNS)
    raw["temperature_c"] = np.nan   # not published

    capacities = pd.read_pickle(raw_dir / "diagnostic_capacities.pkl").loc[cell_id, CAPACITY_COLUMN].dropna()
    cycle_start = raw.groupby("cycle")["time_s"].first()
    diagnostics = [int(c) for c in capacities.index if int(c) in cycle_start.index]

    reference = pd.DataFrame({
        "time_s": cycle_start.loc[diagnostics].to_numpy(),
        "capacity_ah": capacities.loc[diagnostics].to_numpy(),
    })
    diagnostic_cycles = set(diagnostics) | {c + 1 for c in diagnostics}
    usage = raw[~raw["cycle"].isin(diagnostic_cycles)].drop(columns="cycle")
    return Cell("stanford_nca", cell_id, usage, reference)
