"""Which kinds and cells play which role. The reasoning is in docs/splits.md.

Kinds: the development kinds rotate, each one held out in turn while the
model trains on the others (leave one kind out). The test kind is never
touched until the method is frozen.

Cells: inside a held-out kind, some cells only supply in-context examples and
the rest are only scored, so no cell is ever both.
"""

import numpy as np
import yaml

from .paths import DATASETS_YAML

EXAMPLE_FRACTION = 1 / 3   # share of a held-out kind's cells that supply examples


def kinds(split: str) -> list[str]:
    """Names of the kinds with the given split, e.g. 'development', 'train_only' or 'test'."""
    config = yaml.safe_load(DATASETS_YAML.read_text())
    return [name for name, entry in config.items() if entry["split"] == split]


def leave_one_kind_out() -> list[tuple[list[str], str]]:
    """One fold per development kind: (kinds to train on, kind held out).

    Train-only kinds are in every fold's training set.
    """
    development, always = kinds("development"), kinds("train_only")
    return [([k for k in development if k != held_out] + always, held_out) for held_out in development]


def example_and_scored_cells(cells, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split a held-out kind's cells into example cells and scored cells.

    The draw depends only on the cell names and the seed, so every experiment
    that uses the same seed scores the same cells.
    """
    names = sorted(set(cells))
    order = np.random.default_rng(seed).permutation(len(names))
    n_example = max(1, round(len(names) * EXAMPLE_FRACTION))
    example = sorted(names[i] for i in order[:n_example])
    scored = sorted(names[i] for i in order[n_example:])
    return example, scored
