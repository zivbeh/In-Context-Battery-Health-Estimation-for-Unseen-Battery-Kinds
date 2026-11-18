"""Collect the E1 runs into one table: results/e1_finetune/summary.csv.

One row per mode and held-out kind, averaged over seeds, scored after the last
training step.
"""

import json

import pandas as pd

from battery_icl.paths import RESULTS_DIR

COLUMNS = ["zero_shot_mae", "zero_shot_skill", "calibrated_mae", "calibrated_skill", "tracking", "train_minutes"]


def main() -> None:
    rows = []
    for path in sorted((RESULTS_DIR / "e1_finetune").glob("*/*/summary.json")):
        run = json.loads(path.read_text())
        rows.append({
            "mode": run["mode"],
            "held_out": run["held_out"],
            "seed": run["seed"],
            **{key: run["final"][key] for key in COLUMNS},
            "trainable_millions": round(run["trainable_parameters"] / 1e6, 2),
        })
    runs = pd.DataFrame(rows)
    table = runs.groupby(["mode", "held_out"]).mean(numeric_only=True).drop(columns="seed").round(3)
    table["seeds"] = runs.groupby(["mode", "held_out"]).size()
    table.to_csv(RESULTS_DIR / "e1_finetune" / "summary.csv")

    pd.set_option("display.width", 200)
    print(table)
    print("\nMean over held-out kinds:")
    print(table.groupby("mode")[COLUMNS].mean().round(3))


if __name__ == "__main__":
    main()
