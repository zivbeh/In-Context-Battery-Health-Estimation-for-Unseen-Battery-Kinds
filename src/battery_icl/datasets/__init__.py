"""One parser module per dataset.

Every parser exposes the same two functions:

    list_cells(raw_dir) -> list of cell ids
    load_cell(raw_dir, cell_id) -> battery_icl.standardize.Cell

Add a new dataset by writing such a module and listing it in PARSERS.
"""

from . import cmu, nasa, stanford_lg, stanford_nca, tum, xjtu

PARSERS = {
    "cmu": cmu,
    "nasa": nasa,
    "stanford_lg": stanford_lg,
    "stanford_nca": stanford_nca,
    "tum": tum,
    "xjtu": xjtu,
}
