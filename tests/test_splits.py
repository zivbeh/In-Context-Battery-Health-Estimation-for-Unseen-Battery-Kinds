from battery_icl.splits import example_and_scored_cells, kinds, leave_one_kind_out


def test_test_kind_never_appears_in_a_development_fold():
    test_kinds = set(kinds("test"))
    assert test_kinds == {"nasa"}
    for train, held_out in leave_one_kind_out():
        assert held_out not in train
        assert not test_kinds & (set(train) | {held_out})


def test_every_development_kind_is_held_out_once():
    held_out = [fold[1] for fold in leave_one_kind_out()]
    assert sorted(held_out) == sorted(kinds("development"))


def test_example_and_scored_cells_are_disjoint_and_repeatable():
    cells = [f"cell_{i}" for i in range(10)] * 3   # repeated, as in a clip table
    example, scored = example_and_scored_cells(cells, seed=0)
    assert not set(example) & set(scored)
    assert len(example) == 3 and len(scored) == 7
    assert (example, scored) == example_and_scored_cells(cells, seed=0)
