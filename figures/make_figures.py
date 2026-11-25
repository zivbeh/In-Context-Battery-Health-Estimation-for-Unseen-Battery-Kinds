"""Every figure and table of the report, from the saved results. Run from the repo root:

    python figures/make_figures.py

Reads results/x29_paper_results/ (written by scripts/run_x29_paper_results.py); needs only
numpy, pandas and matplotlib, no data download or GPU. Writes figures/<name>.pdf and .png
and figures/tables.md.

Error bars: the K labelled examples are drawn 20 times; for each draw the error is averaged
over the five unseen kinds; points are the median over draws and bars the 2.5-97.5
percentiles, i.e. how much the result depends on which examples were labelled. The main
comparison also gets a paired bootstrap over scored cells (tables.md).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "x29_paper_results"
OUT = ROOT / "figures"
WIDTH = 5.5                                     # NeurIPS text width, inches

# One colour and marker per method, everywhere (palette checked for colour-vision separation).
METHODS = {"floor": ("Mean of the K labels", "#8c8a85", "x", "--"),
           "park_line": ("Park et al. protocol + line", "#eda100", "D", "-"),
           "single_charge": ("Single charge ratio + line", "#eb6834", "s", "-"),
           "fused_mlp": ("Fused, meta-trained MLP + ridge", "#1baf7a", "^", "-"),
           "fused_ridge": ("Fused, ridge (ours)", "#2a78d6", "o", "-")}
KINDS = {"kit": "KIT", "xjtu": "XJTU", "cmu": "CMU", "stanford_lg": "Stanford LG", "stanford_nca": "Stanford NCA"}
INK, MUTED, GRID = "#1a1a1a", "#5f5e5a", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def draw_stats(frame: pd.DataFrame, column: str, by: list) -> pd.DataFrame:
    """Median and 2.5/97.5 percentiles over draws of the error averaged over kinds."""
    per_draw = frame.groupby(by + ["draw"])[column].mean()
    g = per_draw.groupby(by)
    return pd.DataFrame({"mid": g.median(), "lo": g.quantile(0.025), "hi": g.quantile(0.975)})


# ---------------------------------------------------------------- figure 1: the method
def box(ax, x, y, w, h, text, fill="#ffffff", size=8.0):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fill, ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.35)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK, shrinkA=0, shrinkB=0))


def fig_method():
    """Schematic: record -> events -> label-free ratios -> feature vector -> in-context ridge."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.35))
    ax.set(xlim=(-0.05, 12.05), ylim=(-0.75, 4.2))
    ax.axis("off")
    rows = [3.0, 1.6, 0.2]                       # y of the three measurement rows
    h, mid = 0.9, 1.85
    box(ax, 0.0, 0.95, 1.55, 1.8, "$I(\\tau),\\ V(\\tau),$\n$T(\\tau),\\ \\tau \\leq t$", size=7.5)
    events = ["steady\ncharge", "full charge\nfrom rest", "rest to rest"]
    ratios = ["$r_c = Q_{[v_1,v_2]} \\,/\\, \\bar Q^{\\,ref}_{[v_1,v_2]}$",
              "$r_f = Q_{fill}(V_0) \\,/\\, \\bar Q^{\\,ref}_{fill}(V_0)$",
              "$s = \\Delta Q \\,/\\, \\Delta Q_{new}(V_1, V_2)$"]
    for y, ev, ra in zip(rows, events, ratios):
        box(ax, 2.05, y, 1.7, h, ev, size=7.5)
        box(ax, 4.25, y, 3.2, h, ra, size=7.5)
        arrow(ax, (1.55, mid), (2.05, y + h / 2))
        arrow(ax, (3.75, y + h / 2), (4.25, y + h / 2))
        arrow(ax, (7.45, y + h / 2), (7.95, mid))
    box(ax, 7.95, 1.15, 1.35, 1.4, "$x_t \\in \\mathbb{R}^{15}$", size=7.5)
    box(ax, 9.8, 1.15, 2.2, 1.4, "$\\hat y_t = a + w^{\\top} x_t$", size=8)
    box(ax, 9.8, -0.35, 2.2, 0.85, "$\\{(x_k, y_k)\\}_{k=1}^{K}$", size=7.5)
    box(ax, 9.8, 3.1, 2.2, 0.8, "health $\\hat y_t$", size=7.5)
    arrow(ax, (9.3, mid), (9.8, mid))
    arrow(ax, (10.9, 0.5), (10.9, 1.15))
    arrow(ax, (10.9, 2.55), (10.9, 3.1))
    for x, label in [(0.775, "(a) record"), (2.9, "(b) events"), (5.85, "(c) label-free ratios"),
                     (8.625, "(d) features"), (10.9, "(e) in-context ridge")]:
        ax.text(x, -0.72, label, ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig1_method")


# ---------------------------------------------------------------- figure 2: error vs K
def fig_k():
    f = pd.read_csv(RES / "by_k.csv")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.5))
    for m, (label, color, marker, ls) in METHODS.items():
        s = draw_stats(f, m, ["k"])
        ax.errorbar(s.index, s.mid, yerr=[s.mid - s.lo, s.hi - s.mid], color=color, marker=marker, ms=4,
                    ls=ls, lw=1.2, capsize=2, elinewidth=0.7, label=label)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set(xlabel="labelled examples of the new kind, $K$ (log scale)", ylabel="MAE (points of health)", ylim=(0, None))
    grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig2_error_vs_k")


# ---------------------------------------------------------------- figure 3: information ladder
LADDER_LABELS = {"Mean of the K labels": "Mean of the K labels",
                 "Park et al. + line": "Park et al. protocol + line",
                 "single charge + line": "Single charge ratio + line",
                 "+ charge history": "+ last five charge ratios",
                 "+ charge conditions": "+ charge conditions",
                 "+ last full charge": "+ latest full charge",
                 "+ between-charge drives": "+ rest-to-rest drives",
                 "+ between-charge short charges (= X27)": "+ rest-to-rest charges (ours)",
                 "+ Park reading": "+ Park et al. reading"}


def fig_ladder():
    f = pd.read_csv(RES / "ladder.csv")
    names = [n for n in LADDER_LABELS if n in f.columns]
    stats = [draw_stats(f.assign(all=0), n, ["all"]).iloc[0] for n in names]
    per_kind = [f.groupby("kind")[n].median() for n in names]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    y = np.arange(len(names))[::-1]
    colors = ["#8c8a85", "#eda100", "#eb6834"] + ["#2a78d6"] * (len(names) - 3)
    for yi, s, c in zip(y, stats, colors):
        ax.barh(yi, s.mid, color=c, height=0.62, alpha=0.9)
        ax.errorbar(s.mid, yi, xerr=[[s.mid - s.lo], [s.hi - s.mid]], color=INK, capsize=2, lw=0.7)
        ax.text(0.06, yi, f"{s.mid:.2f}", va="center", fontsize=7.5, color="#ffffff", weight="bold")
    for k, kind in enumerate(KINDS):
        ax.scatter([pk[kind] for pk in per_kind], y, s=8, color="#ffffff", edgecolor=INK, lw=0.5, zorder=3,
                   label="median of one kind" if k == 0 else None)
    ax.set_yticks(y, [LADDER_LABELS[n] for n in names])
    ax.set(xlabel="MAE at $K$ = 20 (points of health); bars: mean over the five kinds", xlim=(0, None))
    grid(ax, "x")
    ax.legend(loc="lower right")
    save(fig, "fig3_information_ladder")


# ---------------------------------------------------------------- figure 4: per kind
def fig_per_kind():
    f = pd.read_csv(RES / "by_k.csv").query("k == 20")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.4))
    offsets = np.linspace(-0.3, 0.3, len(METHODS))
    for (m, (label, color, marker, _)), off in zip(METHODS.items(), offsets):
        for i, kind in enumerate(KINDS):
            v = f[f.kind == kind][m]
            ax.errorbar(v.median(), i + off, xerr=[[v.median() - v.quantile(0.025)], [v.quantile(0.975) - v.median()]],
                        color=color, marker=marker, ms=3.5, lw=0.7, capsize=1.5, label=label if i == 0 else None)
    ax.set_yticks(range(len(KINDS)), list(KINDS.values()))
    ax.invert_yaxis()
    ax.set(xlabel="MAE at $K$ = 20 (points of health)", xlim=(0, None))
    grid(ax, "x")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig4_per_kind")


# ---------------------------------------------------------------- figure 5: how many measurements
def fig_counts():
    f = pd.read_csv(RES / "counts.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.0), sharey=True)
    color, marker = METHODS["fused_ridge"][1], METHODS["fused_ridge"][2]
    s = draw_stats(f[f.n_charges == 5], "fused_ridge", ["n_stops"])
    a.errorbar(s.index, s.mid, yerr=[s.mid - s.lo, s.hi - s.mid], color=color, marker=marker, ms=4, capsize=2, lw=1.2)
    a.set_xscale("log")
    a.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    a.minorticks_off()
    a.set(xlabel="rest-to-rest segments summarised (log scale)", ylabel="MAE at $K$ = 20")
    s = draw_stats(f[f.n_stops == 30], "fused_ridge", ["n_charges"])
    b.errorbar(s.index, s.mid, yerr=[s.mid - s.lo, s.hi - s.mid], color=color, marker=marker, ms=4, capsize=2, lw=1.2)
    b.set_xticks([1, 3, 5, 10])
    b.set(xlabel="past charge ratios summarised")
    for ax in (a, b):
        grid(ax)
    save(fig, "fig5_measurement_counts")


# ---------------------------------------------------------------- figure 6: meta-MLP training curves
def fig_training():
    f = pd.read_csv(RES / "training.csv")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.3))
    for column, color, label in [("train_mae", "#5f5e5a", "training kinds"),
                                 ("heldout_mae", METHODS["fused_mlp"][1], "held-out kind, $K$ = 20")]:
        for _, g in f.groupby("kind"):
            ax.plot(g.step, g[column], color=color, lw=0.5, alpha=0.45)
        mean = f.groupby("step")[column].mean()
        ax.plot(mean.index, mean, color=color, lw=1.6, label=label + " (thin: folds,\nthick: mean)")
    ax.set(xlabel="training step", ylabel="MAE (points of health)", ylim=(0, None))
    grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig6_mlp_training")


# ---------------------------------------------------------------- figure 7: one cell over time
def fig_trajectory():
    f = pd.read_csv(RES / "trajectory.csv").sort_values("time_s")
    days = (f.time_s - f.time_s.min()) / 86400
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.2))
    ax.plot(days, 100 * f.health, color=INK, lw=1.4, label="measured health (label)")
    for m in ("single_charge", "fused_ridge"):
        label, color, marker, _ = METHODS[m]
        ax.scatter(days, 100 * f[m], s=9, color=color, marker=marker, lw=0, alpha=0.85, label=label)
    ax.set(xlabel="days since the cell's first scored charge", ylabel="health (%)")
    grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig7_trajectory")


# ---------------------------------------------------------------- tables
def paired_bootstrap(n_boot: int = 2000, seed: int = 0):
    """Mean over kinds of (single charge - fused) cell MAE, resampling cells within each kind."""
    cells = pd.read_csv(RES / "cells_k20.csv")
    rng = np.random.default_rng(seed)
    diffs = {k: 100 * (g.single - g.fused).to_numpy() for k, g in cells.groupby("kind")}   # points
    boots = [np.mean([rng.choice(d, len(d)).mean() for d in diffs.values()]) for _ in range(n_boot)]
    point = np.mean([d.mean() for d in diffs.values()])
    return point, np.percentile(boots, [2.5, 97.5]), {k: d.mean() for k, d in diffs.items()}


def tables():
    by_k = pd.read_csv(RES / "by_k.csv")
    lines = ["# Tables (generated by figures/make_figures.py)", "",
             "## Mean MAE over the five unseen kinds (median over 20 example draws [2.5-97.5%])", "",
             "| K | " + " | ".join(v[0] for v in METHODS.values()) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for k in sorted(by_k.k.unique()):
        cells = []
        for m in METHODS:
            s = draw_stats(by_k[by_k.k == k], m, ["k"]).iloc[0]
            cells.append(f"{s.mid:.2f} [{s.lo:.2f}, {s.hi:.2f}]")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Per kind at K = 20 (median over draws)", "",
              "| kind | " + " | ".join(v[0] for v in METHODS.values()) + " |", "|---" * (len(METHODS) + 1) + "|"]
    k20 = by_k[by_k.k == 20]
    for kind, label in KINDS.items():
        lines.append(f"| {label} | " + " | ".join(f"{k20[k20.kind == kind][m].median():.2f}" for m in METHODS) + " |")
    ladder = pd.read_csv(RES / "ladder.csv")
    lines += ["", "## Information ladder at K = 20", "", "| information | MAE [2.5-97.5%] |", "|---|---|"]
    for n, label in LADDER_LABELS.items():
        if n in ladder.columns:
            s = draw_stats(ladder.assign(all=0), n, ["all"]).iloc[0]
            lines.append(f"| {label} | {s.mid:.2f} [{s.lo:.2f}, {s.hi:.2f}] |")
    point, ci, per_kind = paired_bootstrap()
    lines += ["", "## Paired bootstrap over scored cells, K = 20 (single charge minus fused ridge)", "",
              f"Mean over kinds {point:.2f} points, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]; per kind: " +
              ", ".join(f"{KINDS[k]} {v:+.2f}" for k, v in per_kind.items())]
    counts = pd.read_csv(RES / "counts.csv")
    lines += ["", "## Measurement counts at K = 20 (fused ridge)", "", "| rest-to-rest segments | past charges | MAE |",
              "|---|---|---|"]
    for (ns, nc), g in counts.groupby(["n_stops", "n_charges"]):
        s = draw_stats(g.assign(all=0), "fused_ridge", ["all"]).iloc[0]
        lines.append(f"| {ns} | {nc} | {s.mid:.2f} [{s.lo:.2f}, {s.hi:.2f}] |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print("wrote tables.md")


if __name__ == "__main__":
    fig_method()
    if (RES / "by_k.csv").exists():
        fig_k()
        fig_ladder()
        fig_per_kind()
        fig_counts()
        fig_training()
        fig_trajectory()
        tables()
