"""E0 — pipeline sanity check on one kind.

Question: do frozen Chronos-2 embeddings of our standardized clips carry any
health information at all? If not, something upstream is broken.

Method: embed every clip of one dataset with the frozen encoder, fit a ridge
regression on the embeddings, and score it on cells it has not seen
(five-fold split by cell). Compared with always guessing the mean health of
the training cells.

This is a seen-kind check and says nothing about transfer to new kinds. Run it
on training kinds only.

    python scripts/run_e0_frozen_sanity.py cmu
"""

import json
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_icl.embed import frozen_embeddings, load_clips
from battery_icl.metrics import cell_mae, skill
from battery_icl.paths import RESULTS_DIR


def main(name: str) -> None:
    data = load_clips(name)
    health, cell = data["health"], data["cell"]
    features = frozen_embeddings(name)

    predicted = np.empty_like(health)
    average_guess = np.empty_like(health)
    for train, test in GroupKFold(n_splits=5).split(features, health, groups=cell):
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11)))
        predicted[test] = ridge.fit(features[train], health[train]).predict(features[test])
        average_guess[test] = health[train].mean()

    mae = cell_mae(health, predicted, cell)
    mae_average = cell_mae(health, average_guess, cell)
    result = {
        "dataset": name,
        "clips": int(len(health)),
        "cells": int(len(set(cell))),
        "mae_points": round(mae, 2),
        "mae_average_guess_points": round(mae_average, 2),
        "skill": round(skill(mae, mae_average), 3),
    }
    print(result)

    out_dir = RESULTS_DIR / "e0_frozen_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
