"""CMU eVTOL dataset (Bills et al. 2023): 22 Sony-Murata VTC6 cells.

One CSV per cell holds its whole life: eVTOL missions (take-off, cruise,
landing, recharge) with a capacity test before the first mission and after
every 50. The file has no column that marks the capacity tests and its cycle
counter is unreliable, so tests are found from the current: a capacity test is
the only place where the cell discharges at C/5 for hours.
"""

from pathlib import Path

import pandas as pd

from ..standardize import Cell
from .common import discharged_ah, drop_spans, find_runs

TEST_CURRENT_A = -0.6        # C/5 on a 3 Ah cell
TOLERANCE_A = 0.06
MIN_RUN_S = 1800             # the test's shortest C/5 leg is 1 h; no mission holds C/5 this long
SAME_TEST_GAP_S = 2 * 3600   # a test discharges in three legs with rests between them
MIN_CAPACITY_AH = 2.0        # anything smaller is a test that was cut short
MAX_CAPACITY_AH = 3.4        # anything larger is an aborted test merged with its repeat


def list_cells(raw_dir: Path) -> list[str]:
    return sorted(p.stem for p in raw_dir.glob("VAH*.csv") if "impedance" not in p.stem)


def load_cell(raw_dir: Path, cell_id: str) -> Cell:
    raw = pd.read_csv(raw_dir / f"{cell_id}.csv")
    table = pd.DataFrame({
        "time_s": raw["time_s"],
        "current_a": raw["I_mA"] / 1000.0,   # already charge-positive
        "voltage_v": raw["Ecell_V"],
        "temperature_c": raw["Temperature__C"],
    })
    tests = capacity_tests(table)
    reference = pd.DataFrame(
        [{"time_s": start, "capacity_ah": capacity} for start, _, capacity in tests]
    )
    usage = drop_spans(table, [(start, end) for start, end, _ in tests])
    return Cell("cmu", cell_id, usage, reference)


def capacity_tests(table: pd.DataFrame) -> list[tuple[float, float, float]]:
    """Find the capacity tests: a list of (start_s, end_s, capacity_ah)."""
    time_s = table["time_s"].to_numpy()
    current = table["current_a"].to_numpy()
    at_test_current = abs(current - TEST_CURRENT_A) < TOLERANCE_A

    tests = []
    for a, b in find_runs(at_test_current, time_s, MIN_RUN_S):
        start, end = time_s[a], time_s[b - 1]
        capacity = discharged_ah(time_s[a:b], current[a:b])
        if tests and start - tests[-1][1] < SAME_TEST_GAP_S:
            tests[-1] = (tests[-1][0], end, tests[-1][2] + capacity)
        else:
            tests.append((start, end, capacity))
    return [t for t in tests if MIN_CAPACITY_AH <= t[2] <= MAX_CAPACITY_AH]
