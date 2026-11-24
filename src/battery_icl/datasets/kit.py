"""KIT dataset (Luh et al. 2024): 228 LG INR18650HG2 cells, 3 Ah.

Only the 36 cells aged with a driving profile are used (parameter sets P065
to P076, three cells each, at 0, 10, 25 and 40 °C). One 2-second log per cell:

    log_age/cell_log_age_2s_P065_1_S01_C03.csv

Time counts seconds from the start of the experiment and current is already
charge-positive. The column cap_aged_est_Ah changes once per check-up, a few
minutes after its full discharge at 1 A; those values are the reference
capacities. A check-up warms the cell to 25 °C and runs for about a day and a
half around that moment, so that stretch is cut out of the ageing use.
"""

from pathlib import Path

import pandas as pd

from ..standardize import Cell
from .common import drop_spans

COLUMNS = {
    "timestamp_s": "time_s",
    "i_raw_A": "current_a",
    "v_raw_V": "voltage_v",
    "t_cell_degC": "temperature_c",
    "cap_aged_est_Ah": "capacity_ah",
}
CHECKUP_BEFORE_S = 24 * 3600   # the check-up starts up to this long before its capacity appears
CHECKUP_AFTER_S = 16 * 3600    # and the cell is back at its ageing temperature this long after


def list_cells(raw_dir: Path) -> list[str]:
    return sorted(_cell_id(path) for path in (raw_dir / "log_age").glob("cell_log_age_2s_*.csv"))


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    path = next((raw_dir / "log_age").glob(f"cell_log_age_2s_{cell_id}_*.csv"))
    dtypes = {column: "float64" if column == "timestamp_s" else "float32" for column in COLUMNS}
    log = pd.read_csv(path, sep=";", usecols=list(COLUMNS), dtype=dtypes)
    log = log.rename(columns=COLUMNS)

    capacity = log["capacity_ah"]
    reference = log.loc[capacity.notna() & (capacity != capacity.shift()), ["time_s", "capacity_ah"]]
    checkups = [(t - CHECKUP_BEFORE_S, t + CHECKUP_AFTER_S) for t in reference["time_s"]]
    usage = drop_spans(log.drop(columns="capacity_ah"), checkups)
    return Cell("kit", cell_id, usage.reset_index(drop=True), reference.reset_index(drop=True))


def _cell_id(path: Path) -> str:
    """'cell_log_age_2s_P065_1_S01_C03.csv' -> 'P065_1'."""
    return "_".join(path.stem.split("_")[4:6])
