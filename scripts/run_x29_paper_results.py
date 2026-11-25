"""X29 — every number and plot of the report, from one run with one set of settings.

    python scripts/run_x29_paper_results.py          # needs X15, X16, X24 outputs (as X27/X28)

Same events, features, splits and scoring as X27/X28. Saves per-draw results so the
report can show the spread over example draws, a paired cell bootstrap for the main
comparison, and the meta-MLP training curves.

  by_k.csv          per kind, K and draw: floor, Park et al. + line, single charge + line,
                    fused ridge, fused meta-MLP
  ladder.csv        per kind and draw at K = 20: each rung of the information ladder
  counts.csv        per kind and draw at K = 20: fused ridge with n_stops / n_charges varied
  cells_k20.csv     per scored cell at K = 20: error of single charge and fused ridge,
                    averaged over draws (for the paired bootstrap)
  trajectory.csv    one scored CMU cell, draw 0, K = 20: truth, single-charge and fused estimates
  training.csv      meta-MLP training curve per fold: training-episode MAE, held-out-kind MAE
Writes results/x29_paper_results/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import run_x27_fused_incontext as x27  # noqa: E402
import run_x28_information_ladder as x28  # noqa: E402
from battery_icl.metrics import cell_mae  # noqa: E402
from battery_icl.paths import RESULTS_DIR  # noqa: E402
from battery_icl.splits import example_and_scored_cells, kinds  # noqa: E402

OUT = RESULTS_DIR / "x29_paper_results"
K_VALUES, K_MAIN, DRAWS = [5, 10, 20, 50], 20, 20
PLAIN = 1e-6                                  # a plain least-squares line, as in the published methods
COLS = [f"z_{c}" for c in x27.FEATURES]
x27.LAMBDAS = [PLAIN, 1.0, 10.0, 100.0]


def fit_predict(ex, scored, cols, lam=None):
    zx, ey = ex[cols].to_numpy(), ex.health.to_numpy()
    lam = x27.loo_lambda(zx, ey, ex.cell.to_numpy()) if lam is None else lam
    return x27.ridge_fit(zx, ey, lam)(scored[cols].to_numpy())


def split(h):
    example_cells, scored_cells = example_and_scored_cells(h.cell.to_numpy())
    return h[h.cell.isin(example_cells)], h[h.cell.isin(scored_cells)]


# ---------------------------------------------------------------- meta-MLP with a logged training curve
def train_logged(train: pd.DataFrame, held: pd.DataFrame, kind: str, steps=3000, every=100):
    data = {k: (torch.tensor(g[COLS].to_numpy(np.float32)), torch.tensor(100 * g.health.to_numpy(np.float32)),
                g.cell.to_numpy()) for k, g in train.groupby("kind") if g.cell.nunique() >= 3}
    pool, scored = split(held)
    rng_val = np.random.default_rng(1)
    q = scored.sample(min(len(scored), 2000), random_state=0)
    xq, yq = torch.tensor(q[COLS].to_numpy(np.float32)), 100 * q.health.to_numpy()
    model = x27.Embed(len(COLS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng, names, log, recent = np.random.default_rng(0), list(data), [], []
    for step in range(1, steps + 1):
        x, y, cells = data[names[rng.integers(len(names))]]
        unique = np.unique(cells)
        support = rng.choice(unique, max(1, len(unique) // 3), replace=False)
        s_idx = rng.choice(np.flatnonzero(np.isin(cells, support)),
                           min(np.isin(cells, support).sum(), int(rng.integers(5, 51))), replace=False)
        q_idx = rng.choice(np.flatnonzero(~np.isin(cells, support)), min((~np.isin(cells, support)).sum(), 128),
                           replace=False)
        model.train()
        z = model(x[np.r_[s_idx, q_idx]])
        pred = x27.torch_ridge(z[: len(s_idx)], y[s_idx], z[len(s_idx):])
        loss = nn.functional.smooth_l1_loss(pred, y[q_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        recent.append(float((pred - y[q_idx]).abs().mean()))
        if step % every == 0:                 # held-out kind: monitoring only, never used to select anything
            model.eval()
            with torch.no_grad():
                val = []
                for _ in range(5):
                    ex = x27.spread_sample(pool, K_MAIN, int(rng_val.integers(1e9)))
                    zs = model(torch.tensor(ex[COLS].to_numpy(np.float32)))
                    p = x27.torch_ridge(zs, torch.tensor(100 * ex.health.to_numpy(np.float32)), model(xq))
                    val.append(np.mean(np.abs(p.numpy() - yq)))
            log.append({"kind": kind, "step": step, "train_mae": np.mean(recent), "heldout_mae": np.mean(val)})
            recent = []
    return model.eval(), log


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    park_path = x28.OUT / "park_sessions.csv"
    park = pd.read_csv(park_path) if park_path.exists() else x28.park_table()
    f = x27.build_events(extra=x28.attach_park(park))
    by_k, ladder, cells, training, trajectory = [], [], [], [], None
    sets = {name: [f"z_{x}" for x in feats] for name, feats in x28.ladder_sets().items()}
    for kind in kinds("development"):
        h = f[f.kind == kind].reset_index(drop=True)
        model, log = train_logged(f[f.kind != kind], h, kind)
        training += log
        with torch.no_grad():
            emb = model(torch.tensor(h[COLS].to_numpy(np.float32))).numpy()
        h = h.assign(row=np.arange(len(h)))
        pool, scored = split(h)
        y, c = scored.health.to_numpy(), scored.cell.to_numpy()
        emb_cols = [f"e{i}" for i in range(emb.shape[1])]
        h[emb_cols] = emb
        pool, scored = split(h)
        per_cell = {"single": [], "fused": []}
        for k in K_VALUES:
            for draw in range(DRAWS):
                ex = x27.spread_sample(pool, k, draw)
                est = {"floor": np.full(len(y), ex.health.mean()),
                       "park_line": fit_predict(ex, scored, ["z_extra_park_q"], PLAIN),
                       "single_charge": fit_predict(ex, scored, ["z_est"], PLAIN),
                       "fused_ridge": fit_predict(ex, scored, COLS),
                       "fused_mlp": fit_predict(ex, scored, emb_cols)}
                by_k.append({"kind": kind, "k": k, "draw": draw} | {m: cell_mae(y, p, c) for m, p in est.items()})
                if k == K_MAIN:
                    ladder.append({"kind": kind, "draw": draw, "Mean of the K labels": by_k[-1]["floor"],
                                   "Park et al. + line": by_k[-1]["park_line"],
                                   "single charge + line": by_k[-1]["single_charge"]} |
                                  {n: cell_mae(y, fit_predict(ex, scored, cols), c) for n, cols in list(sets.items())[2:]})
                    for m, key in [("single", "single_charge"), ("fused", "fused_ridge")]:
                        per_cell[m].append(pd.Series(np.abs(est[key] - y)).groupby(c).mean())
                    if kind == "cmu" and draw == 0:
                        cell = scored.cell.value_counts().index[0]
                        m = scored.cell == cell
                        trajectory = pd.DataFrame({"time_s": scored.time_s[m], "health": y[m],
                                                   "single_charge": est["single_charge"][m],
                                                   "fused_ridge": est["fused_ridge"][m], "cell": cell})
        cells.append(pd.DataFrame({"kind": kind, "single": pd.concat(per_cell["single"], axis=1).mean(1),
                                   "fused": pd.concat(per_cell["fused"], axis=1).mean(1)}).rename_axis("cell").reset_index())
        print(kind, pd.DataFrame(by_k).query("kind == @kind and k == @K_MAIN").median(numeric_only=True).round(3).to_dict(),
              flush=True)
    pd.DataFrame(by_k).round(4).to_csv(OUT / "by_k.csv", index=False)
    pd.DataFrame(ladder).round(4).to_csv(OUT / "ladder.csv", index=False)
    pd.concat(cells).round(4).to_csv(OUT / "cells_k20.csv", index=False)
    trajectory.round(5).to_csv(OUT / "trajectory.csv", index=False)
    pd.DataFrame(training).round(4).to_csv(OUT / "training.csv", index=False)

    counts = []
    for n_stops, n_charges in [(1, 5), (3, 5), (10, 5), (30, 5), (100, 5), (30, 1), (30, 3), (30, 10)]:
        g = x27.build_events(n_charges=n_charges, n_stops=n_stops, extra=x28.attach_park(park))
        for kind in kinds("development"):
            pool, scored = split(g[g.kind == kind])
            y, c = scored.health.to_numpy(), scored.cell.to_numpy()
            for draw in range(DRAWS):
                ex = x27.spread_sample(pool, K_MAIN, draw)
                counts.append({"kind": kind, "draw": draw, "n_stops": n_stops, "n_charges": n_charges,
                               "fused_ridge": cell_mae(y, fit_predict(ex, scored, COLS), c)})
    pd.DataFrame(counts).round(4).to_csv(OUT / "counts.csv", index=False)
    print(pd.DataFrame(by_k).groupby("k").median(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
