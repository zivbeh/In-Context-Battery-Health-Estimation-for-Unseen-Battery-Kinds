# data/

Nothing in this folder is tracked by git except this file. The data can live
anywhere: set the `BATTERY_DATA` environment variable to its folder (default: this one).

```
data/
├── raw/<dataset>/               files exactly as published (scripts/download_data.sh)
└── processed/
    ├── <dataset>/<cell>.parquet standardized table per cell (scripts/preprocess.py)
    ├── <dataset>/summary.csv    one row of facts and check results per cell
    └── clips/<dataset>.npz      fixed-length clips with health labels (scripts/make_clips.py)
```

On a laptop, `scripts/download_data.sh samples` fetches about 240 MB of
representative files into `data/raw/`. Sources and formats are in
[docs/datasets.md](../docs/datasets.md); the standardized format is in
[docs/standardization.md](../docs/standardization.md).

Raw files are never edited. Anything derived can be rebuilt from `raw/` with the scripts.
