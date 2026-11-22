from pathlib import Path
from runpy import run_path

import numpy as np


script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pre_e0_same_cell_xshot.py"
runner = run_path(str(script_path))
select_clips = runner["select_clips"]
evaluate_cell = runner["evaluate_cell"]


def shuffled_clips():
    # Both cells have the same sampling times. Clip values identify the cell and hour.
    hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20])
    markers = np.concatenate([hours, hours + 100])
    n = len(hours)
    order = np.random.default_rng(5).permutation(2 * n)
    return {
        "clips": np.broadcast_to(markers[:, None, None], (2 * n, 2, 180))[order],
        "cell": np.repeat(["A", "B"], n)[order],
        "time_s": np.tile(hours * 3600, 2)[order],
        "health": (1.0 - markers / 200.0)[order],
    }


class RecordingPredictor:
    def __init__(self):
        self.pipeline = self
        self.calls = []
        self.health_inputs = []

    def predict(self, support_clips, support_health, query_clips):
        support = support_clips[:, 0, 0]
        queries = query_clips[:, 0, 0]
        self.calls.append((support.copy(), support_health.copy(), queries.copy()))
        return support_health[-1] - queries / 1000.0

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        assert len(inputs) == 1 and set(inputs[0]) == {"target"}
        assert prediction_length == 1 and quantile_levels == [0.5]
        target = np.asarray(inputs[0]["target"])
        self.health_inputs.append(target.copy())
        return [np.array([[[target.mean()]]])], None


def test_one_cell_uses_nested_support_and_identical_later_queries():
    data = shuffled_clips()
    model = RecordingPredictor()
    frame, runs = evaluate_cell(data, "A", [1, 2, 4, 8], model)
    expected_queries = np.array([12, 16, 20])
    assert len(model.calls) == len(model.health_inputs) == len(runs) == 4
    full_support = model.calls[-1][0]
    assert np.all(np.diff(full_support) > 0)
    assert full_support[0] == 0 and full_support[-1] == 9

    for x, (support, labels, queries), target, run in zip(
        [1, 2, 4, 8], model.calls, model.health_inputs, runs
    ):
        rows = frame[frame["shots"] == x]
        assert (rows["cell"] == "A").all()
        np.testing.assert_array_equal(support, full_support[-x:])
        np.testing.assert_array_equal(queries, expected_queries)
        np.testing.assert_array_equal(target, labels)
        np.testing.assert_allclose(labels, 1 - support / 200.0)
        np.testing.assert_array_equal(rows["query_time_s"], queries * 3600)
        np.testing.assert_allclose(rows["hours_since_last_shot"], queries - 9)
        np.testing.assert_allclose(rows["true_soh"], 1 - queries / 200.0)
        np.testing.assert_allclose(rows["predicted_soh"], labels[-1] - queries / 1000.0)
        np.testing.assert_allclose(rows["last_known_soh"], labels[-1])
        np.testing.assert_allclose(rows["health_only_chronos"], labels.mean())
        assert run["shots"] == x and run["query_count"] == len(expected_queries)
        np.testing.assert_array_equal(run["support_time_s"], support * 3600)
        for method in ["predicted_soh", "last_known_soh", "health_only_chronos"]:
            expected_mae = 100 * np.abs(rows[method] - rows["true_soh"]).mean()
            np.testing.assert_allclose(run["mae_points"][method], expected_mae)


def test_changing_query_labels_does_not_change_any_prediction():
    data = shuffled_clips()
    before, _ = evaluate_cell(data, "A", [1, 4], RecordingPredictor())
    query_rows = (data["cell"] == "A") & (data["time_s"] > 10 * 3600)
    data["health"][query_rows] -= 0.25
    after, _ = evaluate_cell(data, "A", [1, 4], RecordingPredictor())
    for name in ["predicted_soh", "last_known_soh", "health_only_chronos"]:
        np.testing.assert_array_equal(after[name], before[name])
    np.testing.assert_allclose(after["true_soh"], before["true_soh"] - 0.25)


def test_one_shot_uses_latest_eligible_clip_even_when_run_alone():
    data = shuffled_clips()
    support, queries = select_clips(data["cell"], data["time_s"], "A", 1)
    np.testing.assert_array_equal(data["time_s"][support], [9 * 3600])
    np.testing.assert_array_equal(data["time_s"][queries], np.array([12, 16, 20]) * 3600)


def test_invalid_cell_or_shot_count_fails_before_inference():
    for cell_id, shots in [("missing", [1]), ("A", [11]), ("A", [0, 2])]:
        model = RecordingPredictor()
        with np.testing.assert_raises(ValueError):
            evaluate_cell(shuffled_clips(), cell_id, shots, model)
        assert not model.calls and not model.health_inputs
