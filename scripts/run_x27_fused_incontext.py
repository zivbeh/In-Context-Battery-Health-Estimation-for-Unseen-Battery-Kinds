"""X27 — fuse each charge with everything the car saw since, calibrated in context.

    python scripts/run_x27_fused_incontext.py      # needs X15 (events_auto), X16 and X24 outputs

Imitates a car: it estimates its health at every steady charge, using only what
it has measured up to then, and no label of its own. One feature vector per charge:

  charge     the charge ratio (X15) now, and the median / spread of its last 5
             ratios; rate, voltage span, temperature change vs its reference charges
  full       the latest full charge from rest ratio (X16) and how old it is
  driving    over its last 30 rest-to-rest discharges (X24): median and spread of the
             capacity ratio, relative to the cell's own first 30; current variability,
             trip length; the same median for rest-to-rest charges
  missing    indicators where a part is unavailable (filled with the kind's median)

Features are standardised per kind (no labels needed). Leave one kind out over the
development kinds; on the held-out kind, K labelled charges from its example cells
(spread over them) calibrate, its other cells are scored (20 draws).

  charge_only       least squares on the charge ratio alone (the current method)
  charge_smoothed   least squares on the median of the last 5 charge ratios
  ridge_all         ridge on the whole vector (penalty chosen leave-one-example-cell-out)
  meta_mlp          small MLP, trained only on the other kinds so that ridge on its
                    embedding from K examples predicts the rest (closed-form ridge
                    inside the training loss); the vector's charge features are kept
                    next to the embedding

--ablation (after a full run): ridge on each feature group alone and on all but one group,
K = 20, written to ablation_k20.csv.
Writes results/x27_fused_incontext/{events.csv,results.csv}.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from battery_icl.metrics import cell_mae
from battery_icl.paths import RESULTS_DIR
from battery_icl.splits import example_and_scored_cells, kinds

K_VALUES, DRAWS, LAMBDAS = [5, 10, 20, 50], 20, [1.0, 10.0, 100.0]
FEATURES = ["est", "est_med5", "est_std5", "rate", "span", "delta_temp", "full_est", "full_age_days",
            "drive_med", "drive_iqr", "drive_current_std", "drive_hours", "charge_seg_med", "has_full", "has_drive"]
KEEP = ["est", "est_med5", "full_est", "drive_med"]          # passed next to the MLP embedding
torch.manual_seed(0)


# ---------------------------------------------------------------- one feature vector per charge
def trailing(series: pd.Series, n: int, how: str) -> pd.Series:
    roll = series.rolling(n, min_periods=1)
    if how == "iqr":
        return roll.quantile(0.75) - roll.quantile(0.25)
    return getattr(roll, how)()


def cell_features(ev: pd.DataFrame, full: pd.DataFrame, seg: pd.DataFrame,
                  n_charges: int = 5, n_stops: int = 30) -> pd.DataFrame:
    ev = ev.sort_values("time_s").copy()
    ev["est_med5"] = trailing(ev.estimate, n_charges, "median")          # "5" = default n_charges
    ev["est_std5"] = trailing(ev.estimate, n_charges, "std").fillna(0)
    if len(full):
        full = full.sort_values("time_s").assign(full_est=lambda f: trailing(f.estimate, 3, "median"),
                                                 full_time=lambda f: f.time_s)[["time_s", "full_est", "full_time"]]
        ev = pd.merge_asof(ev, full, on="time_s", direction="backward")
        ev["full_age_days"] = (ev.time_s - ev.full_time) / 86400
    for sign, prefix in [(-1, "drive"), (1, "charge_seg")]:
        part = seg[np.sign(seg.dq) == sign].sort_values("time_s")
        if len(part):
            part = part.assign(rel=part.s / part.s.iloc[:30].median())
            part = part.assign(**{f"{prefix}_med": trailing(part.rel, n_stops, "median")})
            cols = ["time_s", f"{prefix}_med"]
            if prefix == "drive":
                part = part.assign(drive_iqr=trailing(part.rel, n_stops, "iqr"),
                                   drive_current_std=trailing(part.current_std, n_stops, "median"),
                                   drive_hours=trailing(part.hours, n_stops, "median"))
                cols += ["drive_iqr", "drive_current_std", "drive_hours"]
            ev = pd.merge_asof(ev, part[cols], on="time_s", direction="backward")
    ev = ev.rename(columns={"estimate": "est"})
    for col in FEATURES:
        if col not in ev:
            ev[col] = np.nan
    ev["has_full"], ev["has_drive"] = ev.full_est.notna().astype(float), ev.drive_med.notna().astype(float)
    return ev


def build_events(n_charges: int = 5, n_stops: int = 30, extra=None) -> pd.DataFrame:
    """One row per charge; extra(kind, cell, events) may add columns before standardising."""
    x13 = RESULTS_DIR / "x13_charge_capacity"
    events = pd.read_csv(x13 / "per_clip_events_auto.csv")
    full = pd.read_csv(x13 / "per_clip_full_charges.csv")
    segs = pd.read_csv(RESULTS_DIR / "x24_rest_to_rest" / "segments.csv")
    frames = []
    for (kind, cell), ev in events.groupby(["kind", "cell"]):
        row = cell_features(ev, full[(full.kind == kind) & (full.cell == cell)],
                            segs[(segs.kind == kind) & (segs.cell == cell)], n_charges, n_stops)
        frames.append(extra(kind, cell, row) if extra else row)
    f = pd.concat(frames, ignore_index=True)
    f = f[np.isfinite(f.est) & f.est.between(0.3, 1.5)]
    for col in FEATURES + [c for c in f.columns if c.startswith("extra_")]:   # per kind: fill, standardise
        g = f.groupby("kind")[col]
        f[col] = f[col].fillna(g.transform("median")).fillna(0)
        f[f"z_{col}"] = ((f[col] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0).clip(-5, 5)
    return f


# ---------------------------------------------------------------- in-context calibration
def ridge_fit(z: np.ndarray, y: np.ndarray, lam: float):
    zm, ym = z.mean(0), y.mean()
    zc = z - zm
    w = np.linalg.solve(zc.T @ zc + lam * np.eye(z.shape[1]), zc.T @ (y - ym))
    return lambda q: ym + (q - zm) @ w


def loo_lambda(z, y, cells) -> float:
    def err(lam):
        e = [np.abs(ridge_fit(z[cells != c], y[cells != c], lam)(z[cells == c]) - y[cells == c]).mean()
             for c in np.unique(cells) if (cells != c).sum() >= 3]
        return np.mean(e) if e else 0.0
    return min(LAMBDAS, key=err)


def spread_sample(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_cell = [g.sample(frac=1, random_state=int(rng.integers(1e9))) for _, g in pool.groupby("cell")]
    rng.shuffle(per_cell)
    return pool.loc[[g.index[i] for i in range(k) for g in per_cell if i < len(g)][:k]]


# ---------------------------------------------------------------- meta-trained MLP embedding
class Embed(nn.Module):
    def __init__(self, d_in: int, width: int = 64, d_out: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, d_out))

    def forward(self, x):
        return torch.cat([self.net(x), x[:, [FEATURES.index(c) for c in KEEP]]], dim=1)


def torch_ridge(zs, ys, zq, lam: float = 1.0):
    zm, ym = zs.mean(0), ys.mean()
    zc = zs - zm
    w = torch.linalg.solve(zc.T @ zc + lam * torch.eye(zs.shape[1]), zc.T @ (ys - ym))
    return ym + (zq - zm) @ w


def train_meta(train: pd.DataFrame, steps: int = 3000) -> Embed:
    cols = [f"z_{c}" for c in FEATURES]
    data = {k: (torch.tensor(g[cols].to_numpy(np.float32)), torch.tensor(100 * g.health.to_numpy(np.float32)),
                g.cell.to_numpy()) for k, g in train.groupby("kind") if g.cell.nunique() >= 3}
    model = Embed(len(cols))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    names = list(data)
    for _ in range(steps):
        x, y, cells = data[names[rng.integers(len(names))]]
        unique = np.unique(cells)
        support_cells = rng.choice(unique, max(1, len(unique) // 3), replace=False)
        s_idx = np.flatnonzero(np.isin(cells, support_cells))
        q_idx = np.flatnonzero(~np.isin(cells, support_cells))
        s_idx = rng.choice(s_idx, min(len(s_idx), int(rng.integers(5, 51))), replace=False)
        q_idx = rng.choice(q_idx, min(len(q_idx), 128), replace=False)
        z = model(x[np.r_[s_idx, q_idx]])
        pred = torch_ridge(z[: len(s_idx)], y[s_idx], z[len(s_idx):])
        loss = nn.functional.smooth_l1_loss(pred, y[q_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


# ---------------------------------------------------------------- ablation: which part of the vector helps
GROUPS = {"charge": ["est", "est_med5", "est_std5", "rate", "span", "delta_temp"],
          "full": ["full_est", "full_age_days", "has_full"],
          "driving": ["drive_med", "drive_iqr", "drive_current_std", "drive_hours", "charge_seg_med", "has_drive"]}


def ablation(f: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Ridge in context on each group of features, and on all but one group."""
    sets = {g: c for g, c in GROUPS.items()} | {f"all_but_{g}": [x for x in FEATURES if x not in c]
                                                for g, c in GROUPS.items()} | {"all": FEATURES}
    rows = []
    for kind in kinds("development"):
        h = f[f.kind == kind]
        example_cells, scored_cells = example_and_scored_cells(h.cell.to_numpy())
        pool, scored = h[h.cell.isin(example_cells)], h[h.cell.isin(scored_cells)]
        y, c = scored.health.to_numpy(), scored.cell.to_numpy()
        row = {"kind": kind, "k": k}
        for name, feats in sets.items():
            cols, maes = [f"z_{x}" for x in feats], []
            for draw in range(DRAWS):
                ex = spread_sample(pool, k, draw)
                zx, ey = ex[cols].to_numpy(), ex.health.to_numpy()
                maes.append(cell_mae(y, ridge_fit(zx, ey, loo_lambda(zx, ey, ex.cell.to_numpy()))(scored[cols].to_numpy()), c))
            row[name] = float(np.median(maes))
        rows.append(row)
        print(row, flush=True)
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------- evaluation
def main() -> None:
    import sys
    if "--ablation" in sys.argv:            # reuses events.csv from a full run
        out = RESULTS_DIR / "x27_fused_incontext"
        table = ablation(pd.read_csv(out / "events.csv"))
        table.to_csv(out / "ablation_k20.csv", index=False)
        pd.set_option("display.width", 250)
        print(table.to_string(index=False))
        print(table.drop(columns=["kind", "k"]).mean().round(3).to_string())
        return
    out = RESULTS_DIR / "x27_fused_incontext"
    out.mkdir(parents=True, exist_ok=True)
    f = build_events()
    f.round(5).to_csv(out / "events.csv", index=False)
    cols = [f"z_{c}" for c in FEATURES]
    rows = []
    for kind in kinds("development"):
        h = f[f.kind == kind]
        if h.cell.nunique() < 3:
            continue
        model = train_meta(f[f.kind != kind])
        with torch.no_grad():
            emb = model(torch.tensor(h[cols].to_numpy(np.float32))).numpy()
        h = h.assign(row=np.arange(len(h)))
        example_cells, scored_cells = example_and_scored_cells(h.cell.to_numpy())
        pool, scored = h[h.cell.isin(example_cells)], h[h.cell.isin(scored_cells)]
        y, c = scored.health.to_numpy(), scored.cell.to_numpy()
        for k in K_VALUES:
            maes = {m: [] for m in ["floor", "charge_only", "charge_smoothed", "ridge_all", "meta_mlp"]}
            for draw in range(DRAWS):
                ex = spread_sample(pool, k, draw)
                ey, ecells = ex.health.to_numpy(), ex.cell.to_numpy()
                maes["floor"].append(cell_mae(y, np.full(len(y), ey.mean()), c))
                for name, col in [("charge_only", ["est"]), ("charge_smoothed", ["est_med5"])]:
                    maes[name].append(cell_mae(y, ridge_fit(ex[col].to_numpy(), ey, 1e-9)(scored[col].to_numpy()), c))
                zx, zs = ex[cols].to_numpy(), scored[cols].to_numpy()
                maes["ridge_all"].append(cell_mae(y, ridge_fit(zx, ey, loo_lambda(zx, ey, ecells))(zs), c))
                ex_emb, sc_emb = emb[ex.row.to_numpy().astype(int)], emb[scored.row.to_numpy()]
                maes["meta_mlp"].append(cell_mae(y, ridge_fit(ex_emb, ey, loo_lambda(ex_emb, ey, ecells))(sc_emb), c))
            rows.append({"kind": kind, "k": k, "scored_cells": len(scored_cells)} |
                        {m: float(np.median(v)) for m, v in maes.items()})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows).round(3)
    table.to_csv(out / "results.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(table.groupby("k")[["floor", "charge_only", "charge_smoothed", "ridge_all", "meta_mlp"]].mean().round(3))


if __name__ == "__main__":
    main()
