"""TUM dataset (Schreiber et al. 2025): three 78 Ah VW ID.3 pouch cells.

One folder per cell, named after what the cell ran: highway driving,
commuter driving, or 1C/1C cycling. Each folder holds BaSyTec text files of
two sorts:

    *_cyc_*.txt   ageing use (driving or cycling)
    *_CU_*.txt    check-ups, each with a measured discharge capacity

Every file starts its clock at zero, so its "Start of Test" header line is
used to put all files of a cell on one time axis. The check-up capacity is
the column cap_dis_measure, which the tester stores in ampere-seconds.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..standardize import Cell

FOLDERS = {"highway": "highway_scenario", "commuter": "commuter_scenario", "1C1C": "1C1C_cycling"}
COLUMNS = {"~Time[s]": "time_s", "I[A]": "current_a", "U[V]": "voltage_v", "T1[°C]": "temperature_c"}


def list_cells(raw_dir: Path) -> list[str]:
    return [cell for cell, folder in FOLDERS.items() if any((raw_dir / folder).glob("*_cyc_*.txt"))]


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    folder = raw_dir / FOLDERS[cell_id]
    files = [(_start_of_test(p), p) for p in folder.glob("*.txt")]
    test_start = min(start for start, _ in files)

    usage_parts, reference_rows = [], []
    for start, path in sorted(files):
        offset_s = (start - test_start).total_seconds()
        if "_cyc_" in path.name:
            part = _read(path, list(COLUMNS)).rename(columns=COLUMNS)   # current is charge-positive
            part["time_s"] += offset_s
            usage_parts.append(part)
        elif "_CU_" in path.name:
            measured = _read(path, ["cap_dis_measure"])["cap_dis_measure"]
            measured = measured[measured != 0]
            if len(measured):   # some check-up files hold other tests and no capacity
                reference_rows.append({"time_s": offset_s, "capacity_ah": abs(measured.iloc[-1]) / 3600.0})

    return Cell("tum", cell_id, pd.concat(usage_parts, ignore_index=True), pd.DataFrame(reference_rows))


def _header(path: Path) -> list[str]:
    """The lines before the data: they all start with '~'; the last one names the columns."""
    lines = []
    with open(path, encoding="latin-1") as file:
        for line in file:
            if not line.startswith("~"):
                break
            lines.append(line)
    return lines


def _start_of_test(path: Path) -> datetime:
    line = next(l for l in _header(path) if l.startswith("~Start of Test:"))
    return datetime.strptime(line.split(":", 1)[1].strip(), "%d.%m.%Y %H:%M:%S")


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", skiprows=len(_header(path)) - 1, encoding="latin-1", usecols=columns)
