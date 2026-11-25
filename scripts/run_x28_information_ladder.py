"""X28 — how much information is needed: from Park et al.'s single charge reading to X27's full vector.

    python scripts/run_x28_information_ladder.py     # needs X15, X16, X24 outputs (as X27)

1. Park et al. (2026) protocol, re-implemented as written (Methods, "Rigorous dQ
   measurement protocol"): constant-current charging segments (longest run within 25% of the
   charge's median current) with current CV < 25%,
   > 5 min after trimming the first and last 3 min, rising voltage, temperature
   10-35 C (where recorded); charge dQ in a fixed 0.1 V window, the window chosen
   per kind for maximum coverage; dQ normalised by the kind's 90th percentile (P90)
   ("relative capacity"). Used with no label (as in the paper) and with the same K-label
   line as every other method here.
2. Every method is scored on the SAME moments: X27's charges (the car estimates at each
   steady charge), with the latest Park session before it attached. Ladder (K-label ridge,
   adding one group at a time): Park -> own-reference charge ratio (X15) -> + charge
   history -> + charge conditions -> + full charge -> + between-charge drives -> +
   between-charge short charges (= X27) -> + Park.
3. Measurement counts: X27's full vector with the between-charge window n_stops in
   {1, 3, 10, 30, 100} parking-stop pairs, and the charge history n_charges in {1, 3, 5, 10}.

Ridge penalty from {1e-6, 1, 10, 100} by leave-one-example-cell-out (1e-6 = plain least squares).
Writes results/x28_information_ladder/{park_sessions,ladder,counts}.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import run_x27_fused_incontext as x27  # noqa: E402
from battery_icl.metrics import cell_mae  # noqa: E402
from battery_icl.paths import DATASETS_YAML, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from battery_icl.splits import example_and_scored_cells, kinds  # noqa: E402

x27.LAMBDAS = [1e-6, 1.0, 10.0, 100.0]
OUT = RESULTS_DIR / "x28_information_ladder"
TRIM, MIN_LEN, MAX_CV, WINDOW_V = 18, 30, 0.25, 0.1        # 3 min, 5 min, CV 25%, 0.1 V (Park et al.)
GRID_V = np.round(np.arange(2.5, 4.3001, 0.01), 2)


# ---------------------------------------------------------------- Park et al. protocol
def park_sessions(kind: str, path: Path, vmin: float, vmax: float) -> list[dict]:
    table = pd.read_parquet(path, columns=["time_s", "current_c", "voltage_norm", "temperature_c", "health", "block"]) \
        if kind != "stanford_lg" else pd.read_parquet(path, columns=["time_s", "current_c", "voltage_norm", "health", "block"])
    rows = []
    for _, block in table.groupby("block"):
        i = block["current_c"].to_numpy(np.float64)
        v = vmin + block["voltage_norm"].to_numpy(np.float64) * (vmax - vmin)
        temp = block["temperature_c"].to_numpy(np.float64) if "temperature_c" in block else np.full(len(i), np.nan)
        edges = np.flatnonzero(np.diff(np.r_[0, (i > 0.02).astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            med = np.median(i[s:e])                      # the constant-current segment: longest run near the median
            cc = np.flatnonzero(np.diff(np.r_[0, (np.abs(i[s:e] - med) < MAX_CV * med).astype(int), 0]))
            if not len(cc):
                continue
            c0, c1 = max(zip(cc[::2], cc[1::2]), key=lambda r: r[1] - r[0])
            a, b = s + c0 + TRIM, s + c1 - TRIM
            if b - a < MIN_LEN or i[a:b].std() / i[a:b].mean() >= MAX_CV or v[b - 1] <= v[a]:
                continue
            if np.isfinite(temp[a:b]).any() and not 10 <= np.nanmean(temp[a:b]) <= 35:
                continue
            vv, q = np.maximum.accumulate(v[a:b]), np.cumsum(i[a:b]) * 10 / 3600
            inside = (GRID_V >= vv[0]) & (GRID_V <= vv[-1])
            q_at = np.full(len(GRID_V), np.nan)
            q_at[inside] = np.interp(GRID_V[inside], vv, q)
            rows.append({"time_s": block["time_s"].iloc[(a + b) // 2], "health": block["health"].iloc[(a + b) // 2],
                         "q_at": q_at})
    return rows


def park_table() -> pd.DataFrame:
    config = yaml.safe_load(DATASETS_YAML.read_text())
    frames = []
    for kind in kinds("development") + kinds("train_only"):
        rows = []
        for path in sorted((PROCESSED_DIR / kind).glob("*.parquet")):
            rows += [dict(r, cell=path.stem) for r in park_sessions(kind, path, config[kind]["voltage_min"],
                                                                     config[kind]["voltage_max"])]
        if not rows:
            continue
        q = np.stack([r["q_at"] for r in rows])
        step = int(round(WINDOW_V / 0.01))
        covered = ~np.isnan(q[:, :-step]) & ~np.isnan(q[:, step:])
        w = int(covered.sum(0).argmax())                                    # window with the most sessions
        dq = q[:, w + step] - q[:, w]
        f = pd.DataFrame({"kind": kind, "cell": [r["cell"] for r in rows], "time_s": [r["time_s"] for r in rows],
                          "health": [r["health"] for r in rows], "dq": dq, "window_v": GRID_V[w]}).dropna(subset=["dq"])
        f["park_q"] = f.dq / np.percentile(f.dq, 90)
        frames.append(f)
        print(kind, "Park sessions", len(f), "cells", f.cell.nunique(), "window from", GRID_V[w], "V", flush=True)
    return pd.concat(frames, ignore_index=True)


def attach_park(park: pd.DataFrame):
    def extra(kind, cell, ev):
        p = park[(park.kind == kind) & (park.cell == cell)].sort_values("time_s")
        if not len(p):
            return ev.assign(extra_park_q=np.nan, has_park=0.0)
        ev = pd.merge_asof(ev.sort_values("time_s"), p[["time_s", "park_q"]].rename(columns={"park_q": "extra_park_q"}),
                           on="time_s", direction="backward")
        return ev.assign(has_park=ev.extra_park_q.notna().astype(float))
    return extra


# ---------------------------------------------------------------- scoring on identical moments
LADDER = [("Park et al. protocol", ["extra_park_q"]),
          ("own-reference charge ratio (X15)", ["est"]),
          ("+ charge history", ["est_med5", "est_std5"]),
          ("+ charge conditions", ["rate", "span", "delta_temp"]),
          ("+ last full charge", ["full_est", "full_age_days", "has_full"]),
          ("+ between-charge drives", ["drive_med", "drive_iqr", "drive_current_std", "drive_hours", "has_drive"]),
          ("+ between-charge short charges (= X27)", ["charge_seg_med"]),
          ("+ Park reading", ["extra_park_q"])]


def ladder_sets() -> dict:
    """Park's reading alone, then the own-reference charge ratio with one group added per rung."""
    sets, used = {LADDER[0][0]: LADDER[0][1]}, []
    for name, feats in LADDER[1:]:
        used = used + [x for x in feats if x not in used]
        sets[name] = list(used)
    return sets


def score(f: pd.DataFrame, sets: dict, k: int, only_park: bool) -> list[dict]:
    rows = []
    for kind in kinds("development"):
        h = f[f.kind == kind]
        if only_park:
            h = h[h.has_park == 1]
        if h.cell.nunique() < 3:
            continue
        example_cells, scored_cells = example_and_scored_cells(h.cell.to_numpy())
        pool, scored = h[h.cell.isin(example_cells)], h[h.cell.isin(scored_cells)]
        y, c = scored.health.to_numpy(), scored.cell.to_numpy()
        row = {"kind": kind, "k": k, "events": len(scored), "scored_cells": scored.cell.nunique()}
        maes = {name: [] for name in ["floor", "Park, no label"] + list(sets)}
        for draw in range(x27.DRAWS):
            ex = x27.spread_sample(pool, k, draw)
            ey, ecells = ex.health.to_numpy(), ex.cell.to_numpy()
            maes["floor"].append(cell_mae(y, np.full(len(y), ey.mean()), c))
            maes["Park, no label"].append(cell_mae(y, scored.extra_park_q.to_numpy(), c))
            for name, feats in sets.items():
                cols = [f"z_{x}" for x in feats]
                zx = ex[cols].to_numpy()
                lam = x27.loo_lambda(zx, ey, ecells)
                maes[name].append(cell_mae(y, x27.ridge_fit(zx, ey, lam)(scored[cols].to_numpy()), c))
        rows.append(row | {m: float(np.median(v)) for m, v in maes.items()})
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    park = park_table()
    park.round(5).to_csv(OUT / "park_sessions.csv", index=False)
    f = x27.build_events(extra=attach_park(park))
    sets = ladder_sets()
    rows = []
    for only_park in (False, True):
        for k in (5, 20):
            rows += [r | {"events_with_park_only": only_park} for r in score(f, sets, k, only_park)]
    ladder = pd.DataFrame(rows).round(3)
    ladder.to_csv(OUT / "ladder.csv", index=False)
    pd.set_option("display.width", 250)
    names = ["floor", "Park, no label"] + list(sets)
    print(ladder.groupby(["events_with_park_only", "k"])[names].mean().round(3).T.to_string())

    counts = []
    all_feats = {"all": sets["+ between-charge short charges (= X27)"]}
    for n_stops, n_charges in [(1, 5), (3, 5), (10, 5), (30, 5), (100, 5), (30, 1), (30, 3), (30, 10)]:
        g = x27.build_events(n_charges=n_charges, n_stops=n_stops, extra=attach_park(park))
        for r in score(g, all_feats, 20, False):
            counts.append(r | {"n_stops": n_stops, "n_charges": n_charges})
    counts = pd.DataFrame(counts).round(3)
    counts.to_csv(OUT / "counts.csv", index=False)
    print(counts.groupby(["n_stops", "n_charges"])[["floor", "all"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
