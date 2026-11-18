"""Bring every dataset to one format and one scale.

Each dataset parser returns a `Cell` in physical units. The functions here turn
it into the standardized table every later step reads:

    time_s        seconds since the start of the cell's test, on a uniform grid
    current_c     current as a C-rate (amps / nominal amp-hours), charge positive
    voltage_norm  voltage scaled so the datasheet limits map to 0 and 1
    temperature_c cell temperature, NaN where the dataset has none
    health        measured capacity / highest measured capacity, interpolated in time
    block         id of the uninterrupted stretch of ageing use the row belongs to

See docs/standardization.md for the reasoning behind each choice.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cell:
    """One cell as parsed from a dataset, in physical units.

    usage     : ageing use only (random or driving), columns
                time_s, current_a (charge positive), voltage_v, temperature_c
    reference : the periodic capacity tests, columns time_s, capacity_ah
    """

    dataset: str
    cell_id: str
    usage: pd.DataFrame
    reference: pd.DataFrame


@dataclass(frozen=True)
class KindSpec:
    """Datasheet facts for one kind of cell (from configs/datasets.yaml)."""

    nominal_capacity_ah: float
    voltage_min: float
    voltage_max: float


def health_at(reference: pd.DataFrame, time_s: np.ndarray) -> np.ndarray:
    """Health at the given times, as a fraction of the cell's highest reference capacity.

    The highest capacity stands for "when new". It is usually the first test,
    but some cells gain a little capacity early on, and dividing by the first
    test would then give health above 100%.

    Linear in time between reference tests. Times outside the tested span get
    NaN: we do not extrapolate a label.
    """
    ref = reference.sort_values("time_s")
    health = ref["capacity_ah"].to_numpy() / ref["capacity_ah"].max()
    return np.interp(time_s, ref["time_s"], health, left=np.nan, right=np.nan)


def resample_block(block: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    """Put one uninterrupted stretch of data on a uniform time grid.

    Signals are first interpolated to 1 s and then averaged over each dt_s
    bin. Averaging (rather than picking one sample per bin) keeps the charge
    that passed in the bin correct when fast data is brought to a slower grid.
    """
    t = block["time_s"].to_numpy()
    per_bin = max(int(round(dt_s)), 1)
    fine = np.arange(t[0], t[-1], 1.0)
    n_bins = len(fine) // per_bin
    fine = fine[: n_bins * per_bin]

    out = {"time_s": fine.reshape(n_bins, per_bin)[:, 0]}
    for column in ("current_a", "voltage_v", "temperature_c"):
        values = np.interp(fine, t, block[column].to_numpy(dtype=float))
        out[column] = values.reshape(n_bins, per_bin).mean(axis=1)
    return pd.DataFrame(out)


def standardize(cell: Cell, spec: KindSpec, dt_s: float = 10.0, max_gap_s: float = 300.0) -> pd.DataFrame:
    """Return the standardized table for one cell."""
    usage = cell.usage.sort_values("time_s", kind="stable").drop_duplicates("time_s")

    # A logging gap longer than max_gap_s starts a new block, so that no clip
    # is ever interpolated across a pause or a reference test.
    block_id = (usage["time_s"].diff() > max_gap_s).cumsum()
    blocks = []
    for i, block in usage.groupby(block_id):
        resampled = resample_block(block, dt_s) if len(block) > 1 else block.iloc[:0]
        if len(resampled):
            blocks.append(resampled.assign(block=i))
    table = pd.concat(blocks, ignore_index=True)

    voltage_span = spec.voltage_max - spec.voltage_min
    table["current_c"] = table.pop("current_a") / spec.nominal_capacity_ah
    table["voltage_norm"] = (table.pop("voltage_v") - spec.voltage_min) / voltage_span
    table["health"] = health_at(cell.reference, table["time_s"].to_numpy())

    columns = ["time_s", "current_c", "voltage_norm", "temperature_c", "health", "block"]
    return table[columns].astype({"current_c": "float32", "voltage_norm": "float32",
                                  "temperature_c": "float32", "health": "float32"})


def check(table: pd.DataFrame) -> list[str]:
    """Return a list of problems with a standardized table (empty means fine)."""
    problems = []
    if table[["current_c", "voltage_norm"]].isna().any().any():
        problems.append("NaN in current or voltage")
    if not table["voltage_norm"].between(-0.1, 1.1).mean() > 0.99:
        problems.append("voltage outside datasheet limits: check voltage_min/max")
    if table["current_c"].abs().max() > 20:
        problems.append("current above 20C: check units or nominal capacity")
    labelled = table["health"].dropna()
    if labelled.empty:
        problems.append("no row has a health label: check reference tests")
    elif labelled.min() < 0.2:
        problems.append("health below 20%: check reference capacities")
    return problems
