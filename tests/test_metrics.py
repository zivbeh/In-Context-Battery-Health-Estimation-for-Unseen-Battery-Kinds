import numpy as np

from battery_icl.metrics import cell_mae, tracking, transfer_report


def test_cell_mae_weights_cells_equally():
    health = np.array([1.0, 1.0, 1.0, 0.5])
    predicted = np.array([0.9, 0.9, 0.9, 0.5])
    cell = np.array(["a", "a", "a", "b"])
    assert np.isclose(cell_mae(health, predicted, cell), 5.0)   # cell a: 10 points, cell b: 0


def test_report_ignores_out_of_range_clips_and_removes_the_kind_offset():
    rng = np.random.default_rng(0)
    cell = np.repeat(["ex", "s1", "s2"], 50)
    health = np.tile(np.linspace(0.6, 1.0, 50), 3)
    predicted = health + 0.08 + rng.normal(0, 0.005, 150)   # right shape, wrong level
    train_health = np.array([0.8, 1.0])                     # training covers 80–100% only

    report = transfer_report(health, predicted, cell, ["ex"], train_health)
    assert report["scored_cells"] == 2
    assert 0.45 < report["share_of_clips_in_range"] < 0.55
    assert report["zero_shot_mae"] > 7           # the offset dominates
    assert report["calibrated_mae"] < 1          # and disappears once calibrated
    assert report["tracking"] > 0.99
    assert np.isclose(tracking(health, -predicted, cell), -tracking(health, predicted, cell))
