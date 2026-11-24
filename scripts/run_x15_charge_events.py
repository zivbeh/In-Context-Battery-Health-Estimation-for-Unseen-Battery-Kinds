"""X15 — health from steady charges: charge across a fixed voltage window, now vs the same cell when new.

    python scripts/run_x15_charge_events.py

The charge needed to cross a fixed voltage window during a steady charge is
proportional to capacity, and is precise only over a long window; 30-minute clips
cut charges into pieces. Here every steady charge (constant
current within 5% + 0.02C, at least 10 min, inside one block of use) is taken
whole from the cell's standardized table (data/processed/<kind>/<cell>.parquet).
Steady charges: see charges(). Per cell and charge rate (within 10%), the window is the widest run of voltage
levels covered by at least half of the charges; only charges covering all of it
are used. Reference: the cell's earliest charges (20% of its usable ones, 5 to 10), assumed new.
Charges before the cell's capacity peak (break-in, cleaning rule) are skipped.

--align (X22) fits each charge's fine dQ/dV curve as h x the reference curve shifted by
up to 0.06 in voltage, so growing resistance (a shift) is not mistaken for lost capacity.
--ir corrects the voltage for current x resistance (see charges), because at higher
charge rates the growing resistance slides the window along the charge curve (CMU).

Writes results/x13_charge_capacity/per_clip_events_<detector>[_ir][_align].csv
(one row per charge: kind, cell, time, rate, window, health, estimate = ratio).
First run (11-23): steady charges were searched only as the longest run inside each
stretch of positive current and REF_RUNS was 10; xjtu got no estimates.
"""

import argparse

import numpy as np
import pandas as pd

from battery_icl.metrics import cell_mae, tracking
from battery_icl.paths import PROCESSED_DIR, RESULTS_DIR
from battery_icl.splits import kinds

LEVELS = np.round(np.arange(0, 1.0001, 0.025), 3)
DT_H = 10 / 3600
MIN_RUN, RATE_MATCH = 60, 0.10
MIN_ESTIMATES = 1    # auto detector: no estimate at all from long charges -> use all steady runs (first try: 20)


def steady_runs(i: np.ndarray, detector: str):
    """(start, end) of steady charges in one block.

    longest: inside each stretch of current above 0.05C, the longest run within
             5% + 0.02C of that stretch's median current (one per charge).
    all:     every run in which the current changes by less than 0.02C + 2%
             between samples (also catches short steady pieces of driving).
    """
    if detector == "all":
        steady = (i[1:] > 0.05) & (np.abs(np.diff(i)) < 0.02 + 0.02 * i[1:])
        edges = np.flatnonzero(np.diff(np.r_[0, steady.astype(int), 0]))
        return [(a + 1, b + 1) for a, b in zip(edges[::2], edges[1::2])]
    runs = []
    on_edges = np.flatnonzero(np.diff(np.r_[0, (i > 0.05).astype(int), 0]))
    for s, e in zip(on_edges[::2], on_edges[1::2]):
        steady = np.abs(i[s:e] - np.median(i[s:e])) < 0.05 * np.median(i[s:e]) + 0.02
        edges = np.flatnonzero(np.diff(np.r_[0, steady.astype(int), 0]))
        if len(edges):
            a, b = max(zip(edges[::2], edges[1::2]), key=lambda r: r[1] - r[0])
            runs.append((s + a, s + b))
    return runs


def charges(table: pd.DataFrame, ir: bool, detector: str, align: bool = False):
    """(mid time, rate, charge between consecutive LEVELS, health at mid) for every steady charge.

    A steady charge is a run of at least MIN_RUN samples above 0.05C in which the
    current changes by less than 0.02C + 2% between samples. With ir, the voltage
    is corrected by current x resistance, the resistance taken from the voltage
    jump when the current switches on (median of the cell's last 5 such jumps).
    """
    out, jumps = [], []
    for _, block in table.groupby("block"):
        i = block["current_c"].to_numpy(np.float64)
        v = block["voltage_norm"].to_numpy(np.float64)
        temp = block["temperature_c"].to_numpy(np.float64)
        for a, b in steady_runs(i, detector):
            if b - a < MIN_RUN:
                continue
            rate = float(np.median(i[a:b]))
            step = i[a] - i[a - 2] if a >= 2 else 0.0          # current switched on just before the run
            if step > 0.1:
                jumps.append((v[a] - v[a - 2]) / step)
            resistance = np.median(jumps[-5:]) if (ir and jumps) else 0.0
            vv = np.maximum.accumulate(v[a:b] - i[a:b] * resistance)
            q = np.cumsum(i[a:b]) * DT_H
            inside = (LEVELS >= vv[0]) & (LEVELS <= vv[-1])
            q_at = np.full(len(LEVELS), np.nan)
            q_at[inside] = np.interp(LEVELS[inside], vv, q)
            mid = (a + b) // 2
            curve = fine_curve(vv, q) if align else np.diff(q_at)
            out.append((block["time_s"].iloc[mid], rate, curve, block["health"].iloc[mid],
                        float(np.nanmean(temp[a:b])) if np.isfinite(temp[a:b]).any() else np.nan))
    return out


FINE = np.round(np.arange(0, 1.0001, 0.005), 3)       # --align: fine voltage grid
SHIFTS = np.arange(-12, 13)                            # --align: shifts of up to 0.06, in FINE steps
MIN_OVERLAP = 20                                       # --align: least overlap (0.1 of the voltage scale)


def fine_curve(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Charge per FINE voltage step (NaN outside the run), smoothed over 5 steps."""
    inside = (FINE >= v[0]) & (FINE <= v[-1])
    q_at = np.full(len(FINE), np.nan)
    q_at[inside] = np.interp(FINE[inside], v, q)
    dq = np.diff(q_at)
    return pd.Series(dq).rolling(5, center=True, min_periods=5).mean().to_numpy()


def aligned_ratio(dq: np.ndarray, ref: np.ndarray) -> float:
    """h such that dq(V) ~ h * ref(V - s), with the shift s chosen by least squares."""
    best = (np.inf, np.nan)
    for shift in SHIFTS:
        shifted = np.roll(ref, shift)
        if shift > 0:
            shifted[:shift] = np.nan
        elif shift < 0:
            shifted[shift:] = np.nan
        ok = ~np.isnan(dq) & ~np.isnan(shifted)
        if ok.sum() < MIN_OVERLAP:
            continue
        h = dq[ok].sum() / shifted[ok].sum()
        err = np.mean((dq[ok] - h * shifted[ok]) ** 2) / np.mean(dq[ok] ** 2)
        if err < best[0]:
            best = (err, h)
    return best[1]


def widest_window(dqs) -> np.ndarray:
    covered = np.mean([~np.isnan(d) for d in dqs], axis=0) >= 0.5
    edges = np.flatnonzero(np.diff(np.r_[0, covered.astype(int), 0]))
    mask = np.zeros(len(covered), bool)
    if len(edges):
        s, e = max(zip(edges[::2], edges[1::2]), key=lambda r: r[1] - r[0])
        mask[s:e] = True
    return mask


def estimate_cell(found, kind: str, cell: str, start_s: float = 0.0, align: bool = False) -> list[dict]:
    """Estimates for one cell's charges: per rate group, fixed widest window, earliest charges as reference."""
    rows, groups = [], []
    for c in sorted(found, key=lambda c: c[1]):   # group by rate, within RATE_MATCH of the group's first
        if groups and abs(c[1] - groups[-1][0][1]) <= RATE_MATCH * groups[-1][0][1]:
            groups[-1].append(c)
        else:
            groups.append([c])
    for group in groups:
        if align:
            rows += aligned_cell(group, kind, cell, start_s)
            continue
        window = widest_window([c[2] for c in group])
        if window.sum() * 0.025 < 0.1:
            continue
        usable = sorted([c for c in group if not np.isnan(c[2][window]).any()], key=lambda c: c[0])
        n_ref = max(5, min(10, len(usable) // 5))
        if len(usable) <= n_ref:
            continue
        reference = np.mean([c[2][window].sum() for c in usable[:n_ref]])
        ref_age = (np.mean([c[0] for c in usable[:n_ref]]) - start_s) / 86400
        ref_temps = [c[4] for c in usable[:n_ref]]
        ref_temp = np.nanmean(ref_temps) if np.isfinite(ref_temps).any() else np.nan
        for t, rate, dq, health, temp in usable[n_ref:]:
            rows.append({"kind": kind, "cell": cell, "time_s": t, "rate": rate,
                         "span": window.sum() * 0.025, "health": health, "estimate": dq[window].sum() / reference,
                         "ref_age_days": ref_age, "delta_temp": temp - ref_temp})
    return rows


def aligned_cell(group, kind: str, cell: str, start_s: float) -> list[dict]:
    """--align: each charge against the mean fine curve of the group's earliest charges."""
    usable = sorted([c for c in group if np.isfinite(c[2]).sum() >= MIN_OVERLAP], key=lambda c: c[0])
    n_ref = max(5, min(10, len(usable) // 5))
    if len(usable) <= n_ref:
        return []
    with np.errstate(all="ignore"):
        ref = np.nanmean(np.stack([c[2] for c in usable[:n_ref]]), axis=0)
    ref_age = (np.mean([c[0] for c in usable[:n_ref]]) - start_s) / 86400
    ref_temps = [c[4] for c in usable[:n_ref]]
    ref_temp = np.nanmean(ref_temps) if np.isfinite(ref_temps).any() else np.nan
    rows = []
    for t, rate, dq, health, temp in usable[n_ref:]:
        h = aligned_ratio(dq, ref)
        if np.isfinite(h):
            rows.append({"kind": kind, "cell": cell, "time_s": t, "rate": rate, "span": np.isfinite(dq).sum() * 0.005,
                         "health": health, "estimate": h, "ref_age_days": ref_age, "delta_temp": temp - ref_temp})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ir", action="store_true", help="correct voltage by current x resistance")
    parser.add_argument("--detector", choices=["longest", "all", "auto"], default="auto")
    parser.add_argument("--align", action="store_true",
                        help="fit each charge curve as health x the reference curve shifted in voltage (X22)")
    args = parser.parse_args()
    variant = f"events_{args.detector}" + ("_ir" if args.ir else "") + ("_align" if args.align else "")
    rows, summary = [], []
    for kind in kinds("development") + kinds("train_only"):
        n_charges = 0
        for path in sorted((PROCESSED_DIR / kind).glob("*.parquet")):
            table = pd.read_parquet(path, columns=["time_s", "current_c", "voltage_norm", "temperature_c", "health", "block"])
            labelled = table.dropna(subset=["health"])
            peak_time = labelled["time_s"].iloc[labelled["health"].to_numpy().argmax()]
            detectors = ["longest", "all"] if args.detector == "auto" else [args.detector]
            for detector in detectors:   # auto: fall back to all steady runs if too few long ones
                found = [c for c in charges(table, args.ir, detector, args.align)
                         if c[0] >= peak_time and not np.isnan(c[3])]
                cell_rows = estimate_cell(found, kind, path.stem, table["time_s"].iloc[0], args.align)
                if len(cell_rows) >= MIN_ESTIMATES or detector == detectors[-1]:
                    break
            n_charges += len(found)
            rows += cell_rows
        f = pd.DataFrame([r for r in rows if r["kind"] == kind])
        if len(f):
            summary.append({"kind": kind, "steady_charges": n_charges, "estimated": len(f), "cells": f.cell.nunique(),
                            "mae_uncalibrated": cell_mae(f.health.to_numpy(), f.estimate.to_numpy(), f.cell.to_numpy()),
                            "tracking": tracking(f.health.to_numpy(), f.estimate.to_numpy(), f.cell.to_numpy()),
                            "median_window": f.span.median()})
            print(summary[-1], flush=True)
    out = RESULTS_DIR / "x13_charge_capacity"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(5).to_csv(out / f"per_clip_{variant}.csv", index=False)
    pd.DataFrame(summary).round(3).to_csv(out / f"results_{variant}.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
