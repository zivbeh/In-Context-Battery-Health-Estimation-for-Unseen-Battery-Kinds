"""Where things live on disk.

The data root can sit outside the repo (it does on the GPU box). Point the
BATTERY_DATA environment variable at it; otherwise `<repo>/data` is used.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("BATTERY_DATA", REPO_ROOT / "data"))

RAW_DIR = DATA_ROOT / "raw"              # files exactly as downloaded
PROCESSED_DIR = DATA_ROOT / "processed"  # one standardized parquet per cell
RESULTS_DIR = REPO_ROOT / "results"
DATASETS_YAML = REPO_ROOT / "configs" / "datasets.yaml"
