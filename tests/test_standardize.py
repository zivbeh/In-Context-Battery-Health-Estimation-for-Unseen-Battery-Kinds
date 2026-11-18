import numpy as np
import pandas as pd

from battery_icl.standardize import Cell, KindSpec, check, health_at, standardize

SPEC = KindSpec(nominal_capacity_ah=2.0, voltage_min=3.0, voltage_max=4.2)


def make_cell() -> Cell:
    """Two 100 s stretches of 1 A discharge at 3.6 V, logged every 2 s, 1 h apart."""
    t = np.concatenate([np.arange(0, 100, 2.0), np.arange(3600, 3700, 2.0)])
    usage = pd.DataFrame({"time_s": t, "current_a": -1.0, "voltage_v": 3.6, "temperature_c": 25.0})
    reference = pd.DataFrame({"time_s": [0.0, 4000.0], "capacity_ah": [2.0, 1.6]})
    return Cell("toy", "cell_1", usage, reference)


def test_health_is_relative_to_highest_reference_and_not_extrapolated():
    reference = pd.DataFrame({"time_s": [0.0, 100.0], "capacity_ah": [2.0, 1.0]})
    health = health_at(reference, np.array([-1.0, 0.0, 50.0, 100.0, 101.0]))
    assert np.isnan(health[0]) and np.isnan(health[-1])
    np.testing.assert_allclose(health[1:4], [1.0, 0.75, 0.5])


def test_health_never_exceeds_one_when_capacity_rises_after_the_first_test():
    reference = pd.DataFrame({"time_s": [0.0, 100.0, 200.0], "capacity_ah": [1.9, 2.0, 1.0]})
    np.testing.assert_allclose(health_at(reference, np.array([0.0, 100.0, 200.0])), [0.95, 1.0, 0.5])


def test_standardize_scales_and_resamples():
    table = standardize(make_cell(), SPEC, dt_s=1.0)
    np.testing.assert_allclose(table["current_c"], -0.5)     # 1 A on a 2 Ah cell
    np.testing.assert_allclose(table["voltage_norm"], 0.5)   # 3.6 V between 3.0 and 4.2
    assert np.allclose(table.groupby("block")["time_s"].diff().dropna(), 1.0)
    assert check(table) == []


def test_gap_starts_a_new_block():
    table = standardize(make_cell(), SPEC, dt_s=1.0, max_gap_s=60.0)
    assert table["block"].nunique() == 2
    # Nothing is interpolated across the one-hour pause.
    assert not table["time_s"].between(200, 3500).any()


def test_check_flags_wrong_voltage_limits():
    wrong = KindSpec(nominal_capacity_ah=2.0, voltage_min=3.8, voltage_max=4.2)
    problems = check(standardize(make_cell(), wrong))
    assert any("voltage" in p for p in problems)
