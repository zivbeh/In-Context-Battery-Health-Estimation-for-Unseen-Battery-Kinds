"""Stanford LG M50T dataset (Pozzato et al. 2022): 10 NMC 21700 cells on UDDS driving.

The test alternates a diagnostic and a cycling campaign: Diag 1, Cycling 1,
Diag 2, Cycling 2, and so on. We read the authors' processed .mat files:

    cycling_tests/Cycling_<n>/_processed_mat/<cell>.mat   current, voltage, time
    diagnostic_tests/_processed_mat/capacity_test.mat     C/20 discharge per diagnostic

Each cycling file starts its clock at zero, so the campaigns are laid end to
end on one time axis. Diagnostic n is placed at the start of campaign n, and
the last diagnostic at the end of the last campaign.

The files are sampled at 0.1 s and are read at 1 s to keep memory in check.
No temperature is included in the processed files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from ..standardize import Cell

GAP_BETWEEN_CAMPAIGNS_S = 3600.0   # long enough to start a new block


def _campaign_files(raw_dir: Path, cell_id: str) -> dict[int, Path]:
    files = (raw_dir / "cycling_tests").glob(f"Cycling_*/_processed_mat/{cell_id}.mat")
    return {int(p.parts[-3].split("_")[1]): p for p in files}


def list_cells(raw_dir: Path) -> list[str]:
    return sorted({p.stem for p in (raw_dir / "cycling_tests").glob("Cycling_*/_processed_mat/*.mat")})


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    capacity = _diagnostic_capacities(raw_dir, cell_id)

    usage_parts, reference_rows, clock = [], [], 0.0
    for number, path in sorted(_campaign_files(raw_dir, cell_id).items()):
        campaign = _read_campaign(path)
        campaign["time_s"] += clock
        usage_parts.append(campaign)
        if number in capacity:
            reference_rows.append({"time_s": clock, "capacity_ah": capacity[number]})
        clock = campaign["time_s"].iloc[-1] + GAP_BETWEEN_CAMPAIGNS_S
        if number + 1 in capacity:   # overwritten by the next campaign's start if there is one
            closing = {"time_s": clock, "capacity_ah": capacity[number + 1]}

    if reference_rows and closing["time_s"] > reference_rows[-1]["time_s"]:
        reference_rows.append(closing)
    return Cell("stanford_lg", cell_id, pd.concat(usage_parts, ignore_index=True), pd.DataFrame(reference_rows))


def _read_campaign(path: Path) -> pd.DataFrame:
    """One cycling campaign on a 1 s grid, in physical units."""
    mat = scipy.io.loadmat(path, squeeze_me=True)
    # Variable names look like "I_full_vec_M1_NMC25degC"; keep the part before "_full_vec".
    signals = {name.split("_full_vec")[0]: values for name, values in mat.items() if "_full_vec" in name}
    time_s, current_a, voltage_v = signals["t"], signals["I"], signals["V"]

    logged = voltage_v > 0   # the tester writes a zero-voltage row at start-up
    grid = np.arange(time_s[logged][0], time_s[logged][-1], 1.0)
    return pd.DataFrame({
        "time_s": grid - grid[0],
        "current_a": np.interp(grid, time_s[logged], current_a[logged]),   # already charge-positive
        "voltage_v": np.interp(grid, time_s[logged], voltage_v[logged]),
        "temperature_c": np.nan,
    })


def _diagnostic_capacities(raw_dir: Path, cell_id: str) -> dict[int, float]:
    """Capacity in Ah measured at each diagnostic that was run for this cell."""
    mat = scipy.io.loadmat(raw_dir / "diagnostic_tests/_processed_mat/capacity_test.mat", squeeze_me=True)
    column = list(mat["col_cell_label"]).index(cell_id)

    capacities = {}
    for row, number in enumerate(mat["row_diag_number"]):
        charge_removed = np.atleast_1d(mat["cap"][row, column]).astype(float)
        if charge_removed.size > 1:   # a missing diagnostic is stored as a single NaN
            capacities[int(number)] = float(np.nanmax(np.abs(charge_removed)))
    return capacities
