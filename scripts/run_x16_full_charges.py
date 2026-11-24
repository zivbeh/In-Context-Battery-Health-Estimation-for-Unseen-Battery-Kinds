"""X16 — health from full charges started at rest: charge to fill up from a resting voltage.

    python scripts/run_x16_full_charges.py

X15 fails on CMU's ~1C charges: growing resistance slides a fixed voltage window
along the charge curve. A full constant-current-then-constant-voltage charge ends
at "full" whatever the resistance, and the voltage after at least REST_MIN
samples of rest is close to the equilibrium voltage, so it marks the start
without a resistance effect. The charge needed to fill up from that voltage V0
is capacity times the empty share at V0. Compared with the same cell's earliest
full charges at the same V0 (linear interpolation over their V0; only within
0.05 of one of them), the ratio follows health. No label is used.

A full charge: current above 0.02C without a break, at least 10 minutes, ending
with voltage >= 0.95 and the current below half its median (the constant-voltage
taper), right after >= REST_MIN samples below 0.02C. Charges before the cell's
capacity peak are skipped. Reference: earliest 20% of the cell's full charges
(5 to 10), assumed new.

--rest sets the rest required before the charge (default 30 samples = 5 min; after
high-power use, e.g. CMU's flights, the voltage needs longer to settle).
Writes results/x13_charge_capacity/per_clip_full_charges[_rest<N>].csv (same format as X15).
"""

import argparse

import numpy as np
import pandas as pd

from battery_icl.metrics import cell_mae, tracking
from battery_icl.paths import PROCESSED_DIR, RESULTS_DIR
from battery_icl.splits import kinds

DT_H = 10 / 3600
REST_MIN, MIN_CHARGE_SAMPLES, V0_MATCH = 30, 60, 0.05


def full_charges(table: pd.DataFrame, rest_min: int = REST_MIN):
    """(mid time, start voltage V0, charge passed, health at mid) for every full charge after rest."""
    out = []
    for _, block in table.groupby("block"):
        i = block["current_c"].to_numpy(np.float64)
        v = block["voltage_norm"].to_numpy(np.float64)
        temp = block["temperature_c"].to_numpy(np.float64)
        on = i > 0.02
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s < MIN_CHARGE_SAMPLES or s < rest_min:
                continue
            if not (np.abs(i[s - rest_min : s]) < 0.02).all():
                continue
            if v[e - 1] < 0.95 or i[e - 1] > 0.5 * np.median(i[s:e]):
                continue
            mid = (s + e) // 2
            out.append((block["time_s"].iloc[mid], v[s - 1], i[s:e].sum() * DT_H, block["health"].iloc[mid],
                        float(np.nanmean(temp[s:e])) if np.isfinite(temp[s:e]).any() else np.nan))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rest", type=int, default=REST_MIN, help="samples (10 s) of rest before the charge")
    args = parser.parse_args()
    variant = "full_charges" if args.rest == REST_MIN else f"full_charges_rest{args.rest}"
    rows, summary = [], []
    for kind in kinds("development") + kinds("train_only"):
        n_found = 0
        for path in sorted((PROCESSED_DIR / kind).glob("*.parquet")):
            table = pd.read_parquet(path, columns=["time_s", "current_c", "voltage_norm", "temperature_c", "health", "block"])
            labelled = table.dropna(subset=["health"])
            peak_time = labelled["time_s"].iloc[labelled["health"].to_numpy().argmax()]
            found = sorted([c for c in full_charges(table, args.rest) if c[0] >= peak_time and not np.isnan(c[3])])
            n_found += len(found)
            n_ref = max(5, min(10, len(found) // 5))
            if len(found) <= n_ref:
                continue
            ref_age = (np.mean([c[0] for c in found[:n_ref]]) - table["time_s"].iloc[0]) / 86400
            ref = sorted(found[:n_ref], key=lambda c: c[1])
            ref_v0 = np.array([c[1] for c in ref])
            ref_q = np.array([c[2] for c in ref])
            ref_temp = np.nanmean([c[4] for c in ref]) if np.isfinite([c[4] for c in ref]).any() else np.nan
            for t, v0, q, health, temp in found[n_ref:]:
                if np.min(np.abs(ref_v0 - v0)) > V0_MATCH:
                    continue
                rows.append({"kind": kind, "cell": path.stem, "time_s": t, "rate": v0, "span": q,
                             "health": health, "estimate": q / np.interp(v0, ref_v0, ref_q), "ref_age_days": ref_age,
                             "delta_temp": temp - ref_temp})
        f = pd.DataFrame([r for r in rows if r["kind"] == kind])
        if len(f):
            summary.append({"kind": kind, "full_charges": n_found, "estimated": len(f), "cells": f.cell.nunique(),
                            "mae_uncalibrated": cell_mae(f.health.to_numpy(), f.estimate.to_numpy(), f.cell.to_numpy()),
                            "tracking": tracking(f.health.to_numpy(), f.estimate.to_numpy(), f.cell.to_numpy())})
        else:
            summary.append({"kind": kind, "full_charges": n_found, "estimated": 0})
        print(summary[-1], flush=True)
    out = RESULTS_DIR / "x13_charge_capacity"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(5).to_csv(out / f"per_clip_{variant}.csv", index=False)
    pd.DataFrame(summary).round(3).to_csv(out / f"results_{variant}.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
