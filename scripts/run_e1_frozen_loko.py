"""E1a — frozen Chronos-2 on an unseen kind, with no examples (K = 0).

The cheapest point of the fine-tuning ladder: the encoder is not trained at
all, and a ridge regression on its embeddings is the only thing fitted.

For each development kind in turn: fit on the other development kinds and
score on the held-out kind (see metrics.transfer_report for what is scored).
Every training kind gets the same total weight, so the kind with the most
cells does not decide the fit.

    python scripts/run_e1_frozen_loko.py
"""

import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_icl.embed import frozen_embeddings, load_clips
from battery_icl.metrics import transfer_report
from battery_icl.paths import RESULTS_DIR
from battery_icl.splits import example_and_scored_cells, kinds, leave_one_kind_out


def main() -> None:
    data = {name: load_clips(name) for name in kinds("development") + kinds("train_only")}
    features = {name: frozen_embeddings(name) for name in data}

    results = []
    for train_kinds, held_out in leave_one_kind_out():
        x = np.concatenate([features[k] for k in train_kinds])
        y = np.concatenate([data[k]["health"] for k in train_kinds])
        weight = np.concatenate([np.full(len(data[k]["health"]), 1.0 / len(data[k]["health"])) for k in train_kinds])
        weight *= len(y) / weight.sum()

        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
        ridge.fit(x, y, ridgecv__sample_weight=weight)

        held = data[held_out]
        example_cells, _ = example_and_scored_cells(held["cell"])
        report = transfer_report(held["health"], ridge.predict(features[held_out]), held["cell"], example_cells, y)
        results.append({"held_out": held_out, "trained_on": train_kinds, **report})
        print(results[-1])

    out_dir = RESULTS_DIR / "e1a_frozen_loko"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
