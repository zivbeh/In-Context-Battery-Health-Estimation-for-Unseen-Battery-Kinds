"""Error measures. Health is a fraction in the code and percent points in reports."""

import numpy as np
import pandas as pd


def cell_mae(health: np.ndarray, predicted: np.ndarray, cell: np.ndarray) -> float:
    """Mean absolute error in points of health.

    Averaged within each cell first and then across cells, so that cells
    with many clips do not dominate.
    """
    error = pd.Series(np.abs(predicted - health) * 100.0)
    return float(error.groupby(np.asarray(cell)).mean().mean())


def skill(mae_model: float, mae_average_guess: float) -> float:
    """1 means perfect, 0 means no better than always guessing the average."""
    return 1.0 - mae_model / mae_average_guess


def tracking(health: np.ndarray, predicted: np.ndarray, cell: np.ndarray) -> float:
    """Mean within-cell correlation between predicted and true health.

    1 means the prediction follows each cell's ageing perfectly, even if its
    level is off; 0 means it does not follow it at all.
    """
    frame = pd.DataFrame({"health": health, "predicted": predicted, "cell": np.asarray(cell)})
    per_cell = frame.groupby("cell").apply(lambda g: g["health"].corr(g["predicted"]), include_groups=False)
    return float(per_cell.mean())


def transfer_report(health, predicted, cell, example_cells, train_health) -> dict:
    """Score predictions on a held-out kind, only where the comparison is fair.

    Only clips whose health lies inside the range of the training labels are
    scored: outside it the model would have to extrapolate, and an error
    there says nothing about transfer between kinds.

    Two settings are reported:
      zero-shot   predictions as they are, against guessing the training mean.
      calibrated  the kind's mean error, measured on its example cells, is
                  removed first; against guessing the example cells' mean health.
                  This asks whether reading the clip adds anything once a few
                  labelled cells of the kind are available.
    """
    in_range = (health >= train_health.min()) & (health <= train_health.max())
    is_example = np.isin(cell, example_cells)
    scored, examples = in_range & ~is_example, in_range & is_example
    y, p, c = health[scored], predicted[scored], cell[scored]

    offset = np.mean(predicted[examples] - health[examples])
    mae_zero = cell_mae(y, p, c)
    mae_zero_guess = cell_mae(y, np.full_like(y, train_health.mean()), c)
    mae_cal = cell_mae(y, p - offset, c)
    mae_cal_guess = cell_mae(y, np.full_like(y, health[examples].mean()), c)
    return {
        "scored_cells": int(len(set(c))),
        "share_of_clips_in_range": round(float(in_range.mean()), 3),
        "zero_shot_mae": round(mae_zero, 2),
        "zero_shot_guess_mae": round(mae_zero_guess, 2),
        "zero_shot_skill": round(skill(mae_zero, mae_zero_guess), 3),
        "calibrated_mae": round(mae_cal, 2),
        "calibrated_guess_mae": round(mae_cal_guess, 2),
        "calibrated_skill": round(skill(mae_cal, mae_cal_guess), 3),
        "tracking": round(tracking(y, p, c), 3),
    }
