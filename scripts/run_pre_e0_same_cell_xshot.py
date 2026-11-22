"""Estimate SoH for later clips of one cell with frozen Chronos-2.

    python scripts/run_pre_e0_same_cell_xshot.py cmu --cell VAH07 --device cuda

One shot is a 30-minute usage clip with its retrospective, interpolated SoH.
Each query is a separate one-step prediction from the same support history.
"""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from battery_icl.clips import spread_evenly
from battery_icl.metrics import cell_mae
from battery_icl.paths import REPO_ROOT, RESULTS_DIR


def select_clips(cell, time_s, cell_id, max_shots):
    """Return indices for one early support pool and all later queries."""
    if max_shots < 1:
        raise ValueError("Shot counts must be positive.")
    rows = np.flatnonzero(cell == cell_id)
    if not len(rows):
        raise ValueError(f"No clips found for cell {cell_id!r}.")
    rows = rows[np.argsort(time_s[rows], kind="stable")]
    times = time_s[rows]
    if not np.isfinite(times).all():
        raise ValueError(f"Non-finite clip time for {cell_id}.")
    cutoff = times[0] + (times[-1] - times[0]) / 2
    early, queries = rows[times <= cutoff], rows[times > cutoff]
    if len(early) < max_shots or not len(queries):
        raise ValueError(
            f"{cell_id} has {len(early)} early and {len(queries)} later clips "
            f"for a requested maximum of {max_shots} shots."
        )
    # A standalone one-shot run must use the same latest label as a sweep.
    if max_shots == 1:
        return early[-1:], queries
    return early[spread_evenly(len(early), max_shots)], queries


def evaluate_cell(data, cell_id, shots, predictor):
    """Return query-level predictions and a score report for each shot count."""
    if not shots or min(shots) < 1:
        raise ValueError("Provide at least one positive shot count.")
    clips, health, times = data["clips"], data["health"], data["time_s"]
    pool, queries = select_clips(data["cell"], times, cell_id, max(shots))
    query_clips = clips[queries]
    truth = health[queries]
    if not np.isfinite(truth).all():
        raise ValueError(f"Non-finite query health for {cell_id}.")

    frames, runs = [], []
    for count in sorted(set(shots)):
        support = pool[-count:]
        labels = health[support]
        predicted = predictor.predict(clips[support], labels, query_clips)

        # Every query has the same one-step horizon and observed SoH history.
        quantiles, _ = predictor.pipeline.predict_quantiles(
            [{"target": labels}], prediction_length=1, quantile_levels=[0.5]
        )
        health_only = quantiles[0][0, 0, 0].item()
        if not np.isfinite(predicted).all() or not np.isfinite(health_only):
            raise ValueError(f"Non-finite Chronos-2 forecast for {cell_id}, {count} shots.")

        frame = pd.DataFrame({
            "cell": cell_id,
            "shots": count,
            "query_time_s": times[queries],
            "hours_since_last_shot": (times[queries] - times[support[-1]]) / 3600,
            "predicted_soh": predicted,
            "last_known_soh": labels[-1],
            "health_only_chronos": health_only,
            "true_soh": truth,
        })
        frames.append(frame)
        runs.append({
            "shots": count,
            "support_time_s": times[support].tolist(),
            "query_count": len(queries),
            "mae_points": {
                column: cell_mae(truth, frame[column].to_numpy(), data["cell"][queries])
                for column in ("predicted_soh", "last_known_soh", "health_only_chronos")
            },
        })
    return pd.concat(frames, ignore_index=True), runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="name of the processed clip archive, e.g. cmu")
    parser.add_argument("--cell", required=True, help="one physical cell ID")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    from battery_icl.embed import load_clips
    from battery_icl.model import CHRONOS_ID, ChronosHealthPredictor

    data = load_clips(args.dataset)
    torch.manual_seed(0)
    snapshot = Path(snapshot_download(
        CHRONOS_ID, allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"]
    ))
    predictor = ChronosHealthPredictor.from_pretrained(str(snapshot), device=args.device)
    predictions, runs = evaluate_cell(data, args.cell, args.shots, predictor)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    summary = {
        "dataset": args.dataset,
        "cell": args.cell,
        "cell_clip_count": int(np.count_nonzero(data["cell"] == args.cell)),
        "split": "First half of elapsed clip time for support; second half for queries.",
        "label": "Clip midpoint SoH interpolated between capacity tests.",
        "checkpoint": CHRONOS_ID,
        "checkpoint_revision": snapshot.name,
        "chronos_version": version("chronos-forecasting"),
        "device": args.device,
        "seed": 0,
        "git_commit": commit,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": runs,
    }
    out = RESULTS_DIR / "pre_e0_same_cell_xshot" / args.dataset / args.cell
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame({run["shots"]: run["mae_points"] for run in runs}).T.rename_axis("shots"))
    print(f"Saved {len(predictions)} query predictions to {out}")


if __name__ == "__main__":
    main()
