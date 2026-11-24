"""X24 — capacity ratio from any use between two rests, against a new-cell model of the kind. No label of the cell.

    python scripts/run_x24_rest_to_rest.py [--rest 30]

The real-world setting: a car knows only its own recent current and voltage, and
how a few new cells of its kind behaved (measured once by the maker). After the
cell rests (|current| < 0.02C for at least --rest samples), its voltage is close
to the resting (open-circuit) voltage, which fixes how full it is. Between two
rests, whatever happened (driving, discharge or a charge), counting the current
gives the charge dQ that moved the cell from resting voltage V1 to V2.

New-cell model: Qfull(V), the charge a NEW cell of the kind needs to fill up from
resting voltage V, piecewise linear on a 0.025 grid. It is fitted by least squares
on the reference cells' rest-to-rest segments (the kind's example cells from
splits.example_and_scored_cells, only while their health >= NEW_HEALTH, each dQ
divided by its health). The scored cells' own early life is never used.

  s = dQ / (Qfull(V1) - Qfull(V2))        capacity ratio, the driving version of Q/Q_new

Qfull is fitted separately for discharge and charge segments (resting voltage\nhas hysteresis). Scored on segments of the scored cells with |V2 - V1| >= MIN_DV. Reported per
kind and per segment type (net discharge = driving/discharge, net charge):
tracking (within-cell correlation with health) and MAE of s taken as health.
Writes results/x24_rest_to_rest/{segments,results}[_rest<N>].csv.
"""

import argparse

import numpy as np
import pandas as pd

from battery_icl.metrics import cell_mae, tracking
from battery_icl.paths import PROCESSED_DIR, RESULTS_DIR
from battery_icl.splits import example_and_scored_cells, kinds

DT_H = 10 / 3600
GRID = np.linspace(0, 1, 41)            # voltage knots of the new-cell curve Qfull(V)
NEW_HEALTH, MIN_DV, SMOOTH = 0.95, 0.1, 1.0


def segments(table: pd.DataFrame, rest_min: int) -> list[dict]:
    """One row per pair of consecutive rests in a block: resting voltages, charge between, health."""
    rows = []
    for _, block in table.groupby("block"):
        i = block["current_c"].to_numpy(np.float64)
        v = block["voltage_norm"].to_numpy(np.float64)
        h = block["health"].to_numpy(np.float64)
        edges = np.flatnonzero(np.diff(np.r_[0, (np.abs(i) < 0.02).astype(int), 0]))
        rest_ends = [e - 1 for s, e in zip(edges[::2], edges[1::2]) if e - s >= rest_min]  # most relaxed sample
        for a, b in zip(rest_ends, rest_ends[1:]):
            active = i[a + 1 : b + 1]
            mid = (a + b) // 2
            rows.append({"time_s": block["time_s"].iloc[mid], "v1": v[a], "v2": v[b],
                         "dq": active.sum() * DT_H, "health": h[mid],
                         "current_std": float(active.std()), "hours": len(active) / 360})
    return rows


def interp_weights(v: np.ndarray) -> np.ndarray:
    """Rows of weights so that weights @ knots = piecewise-linear value at v."""
    w = np.zeros((len(v), len(GRID)))
    pos = np.clip(v, 0, 1) * (len(GRID) - 1)
    lo = np.minimum(pos.astype(int), len(GRID) - 2)
    frac = pos - lo
    w[np.arange(len(v)), lo] = 1 - frac
    w[np.arange(len(v)), lo + 1] = frac
    return w


def fit_new_cell(ref: pd.DataFrame) -> np.ndarray:
    """Knots of Qfull(V) from reference segments: dq / health = Qfull(V1) - Qfull(V2), smooth, Qfull(1) = 0."""
    a = interp_weights(ref.v1.to_numpy()) - interp_weights(ref.v2.to_numpy())
    y = ref.dq.to_numpy() / ref.health.to_numpy()
    second_diff = np.diff(np.eye(len(GRID)), 2, axis=0) * SMOOTH
    pin = np.eye(len(GRID))[-1:] * 10
    lhs = np.vstack([a, second_diff, pin])
    rhs = np.r_[y, np.zeros(len(second_diff) + 1)]     # charge in (dq > 0) moves V up, so Qfull falls
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rest", type=int, default=30, help="samples (10 s) of rest that define a resting voltage")
    args = parser.parse_args()
    tag = "" if args.rest == 30 else f"_rest{args.rest}"
    frames, summary = [], []
    for kind in kinds("development") + kinds("train_only"):
        rows = []
        for path in sorted((PROCESSED_DIR / kind).glob("*.parquet")):
            table = pd.read_parquet(path, columns=["time_s", "current_c", "voltage_norm", "health", "block"])
            rows += [dict(r, cell=path.stem) for r in segments(table, args.rest)]
        f = pd.DataFrame(rows).dropna(subset=["health"])
        f = f[(f.v2 - f.v1).abs() >= MIN_DV]
        example_cells, scored_cells = example_and_scored_cells(f.cell.to_numpy())
        ref = f[f.cell.isin(example_cells) & (f.health >= NEW_HEALTH)]
        if len(ref) < 20:
            print(kind, "skipped: only", len(ref), "reference segments")
            continue
        # One new-cell curve per direction: resting voltage after a discharge and after a charge differ.
        f = f.assign(kind=kind, role=np.where(f.cell.isin(example_cells), "example", "scored"), s=np.nan)
        for sign in (-1, 1):
            knots = fit_new_cell(ref[np.sign(ref.dq) == sign])
            rows = np.sign(f.dq) == sign
            dq_new = (interp_weights(f.v1[rows].to_numpy()) - interp_weights(f.v2[rows].to_numpy())) @ knots
            f.loc[rows, "s"] = f.dq[rows] / dq_new
        frames.append(f)
        scored = f[f.role == "scored"]
        for name, part in [("all", scored), ("discharge", scored[scored.dq < 0]), ("charge", scored[scored.dq > 0])]:
            if part.cell.nunique() < 2:
                continue
            y, s, c = part.health.to_numpy(), part.s.to_numpy(), part.cell.to_numpy()
            summary.append({"kind": kind, "segments": name, "n": len(part), "cells": part.cell.nunique(),
                            "reference_segments": len(ref), "median_dv": (part.v2 - part.v1).abs().median(),
                            "tracking": tracking(y, s, c), "mae_s_as_health": cell_mae(y, s, c),
                            "median_s": np.median(s)})
            print(summary[-1], flush=True)
    out = RESULTS_DIR / "x24_rest_to_rest"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).round(5).to_csv(out / f"segments{tag}.csv", index=False)
    table = pd.DataFrame(summary).round(3)
    table.to_csv(out / f"results{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
