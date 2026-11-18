"""E1 — train the health model with one development kind held out (K = 0).

    python scripts/train.py --mode lora --held-out cmu

Modes (see battery_icl/model.py): frozen, lora, full, scratch. The model trains
on the other development kinds and is scored on the held-out kind every
--eval-every steps (see metrics.transfer_report for what is scored). Training time is logged next to every score,
because H2 compares the modes by GPU time, not by steps.

Writes results/e1_finetune/<mode>/<held_out>_seed<seed>/{log.csv, summary.json}.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from battery_icl.embed import load_clips
from battery_icl.metrics import transfer_report
from battery_icl.model import FINETUNE_MODES, HealthRegressor, count_trainable, load_encoder
from battery_icl.paths import RESULTS_DIR
from battery_icl.splits import example_and_scored_cells, leave_one_kind_out

ENCODER_LR = {"frozen": 0.0, "lora": 1e-4, "full": 2e-5, "scratch": 1e-4}
HEAD_LR = 1e-3


def balanced_batch(data: dict, train_kinds: list[str], batch_size: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch in which every training kind is equally likely, whatever its size."""
    clips, health = [], []
    for kind in rng.choice(train_kinds, size=batch_size):
        i = rng.integers(len(data[kind]["health"]))
        clips.append(data[kind]["clips"][i])
        health.append(data[kind]["health"][i])
    return torch.from_numpy(np.stack(clips)).cuda(), torch.tensor(health).cuda()


@torch.no_grad()
def predict(model: HealthRegressor, clips: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(clips), batch_size):
        batch = torch.from_numpy(clips[start : start + batch_size]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs.append(model(batch).float().cpu().numpy())
    model.train()
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=FINETUNE_MODES, required=True)
    parser.add_argument("--held-out", required=True, help="development kind to hold out")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_kinds = next(train for train, held_out in leave_one_kind_out() if held_out == args.held_out)
    data = {k: load_clips(k) for k in train_kinds + [args.held_out]}

    held = data[args.held_out]
    example_cells, _ = example_and_scored_cells(held["cell"])
    train_health = np.concatenate([data[k]["health"] for k in train_kinds])
    train_mean = np.mean([data[k]["health"].mean() for k in train_kinds])

    model = HealthRegressor(load_encoder(args.mode)).cuda()
    # Start the head at the mean health, so early steps do not waste time finding it.
    torch.nn.init.constant_(model.head[-1].bias, float(train_mean))
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": model.head.parameters(), "lr": HEAD_LR}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": ENCODER_LR[args.mode]})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    out_dir = RESULTS_DIR / "e1_finetune" / args.mode / f"{args.held_out}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, train_seconds, losses = [], 0.0, []

    for step in range(1, args.steps + 1):
        started = time.time()
        clips, health = balanced_batch(data, train_kinds, args.batch_size, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(clips)
        loss = torch.nn.functional.l1_loss(predicted.float(), health)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        train_seconds += time.time() - started
        losses.append(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            report = transfer_report(held["health"], predict(model, held["clips"]), held["cell"], example_cells, train_health)
            log.append({
                "step": step,
                "train_minutes": round(train_seconds / 60, 2),
                "train_mae": round(100 * float(np.mean(losses)), 2),
                **report,
            })
            losses = []
            print(log[-1], flush=True)
            pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)

    trainable, total = count_trainable(model)
    summary = {
        **vars(args),
        "trained_on": train_kinds,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "final": log[-1],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
