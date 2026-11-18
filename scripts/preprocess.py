"""Standardize raw datasets: one parquet per cell plus a summary table.

    python scripts/preprocess.py cmu nasa        # named datasets
    python scripts/preprocess.py                 # every dataset with a parser

Writes data/processed/<dataset>/<cell>.parquet and summary.csv.
"""

import sys

import pandas as pd
import yaml

from battery_icl.datasets import PARSERS
from battery_icl.paths import DATASETS_YAML, PROCESSED_DIR, RAW_DIR
from battery_icl.standardize import KindSpec, check, standardize


def summarize(cell, table, problems) -> dict:
    """One row of facts about a standardized cell, for eyeballing and for docs."""
    labelled = table.dropna(subset=["health"])
    return {
        "cell": cell.cell_id,
        "reference_tests": len(cell.reference),
        "highest_capacity_ah": round(cell.reference["capacity_ah"].max(), 3) if len(cell.reference) else None,
        "health_min": round(labelled["health"].min(), 3),
        "labelled_hours": round(len(labelled) * table["time_s"].diff().median() / 3600, 1),
        "blocks": table["block"].nunique(),
        "current_c_p01": round(table["current_c"].quantile(0.01), 2),
        "current_c_p99": round(table["current_c"].quantile(0.99), 2),
        "voltage_norm_min": round(table["voltage_norm"].min(), 3),
        "voltage_norm_max": round(table["voltage_norm"].max(), 3),
        "problems": "; ".join(problems),
    }


def preprocess(name: str) -> None:
    config = yaml.safe_load(DATASETS_YAML.read_text())[name]
    spec = KindSpec(config["nominal_capacity_ah"], config["voltage_min"], config["voltage_max"])
    parser, raw_dir, out_dir = PARSERS[name], RAW_DIR / name, PROCESSED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell_id in parser.list_cells(raw_dir):
        cell = parser.load_cell(raw_dir, cell_id)
        table = standardize(cell, spec)
        table.to_parquet(out_dir / f"{cell_id}.parquet", index=False)
        rows.append(summarize(cell, table, check(table)))
        print(rows[-1])

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"{name}: {len(summary)} cells, {summary['problems'].ne('').sum()} with problems")


if __name__ == "__main__":
    for dataset in sys.argv[1:] or PARSERS:
        preprocess(dataset)
