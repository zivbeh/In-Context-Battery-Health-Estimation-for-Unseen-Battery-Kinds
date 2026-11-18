# In-Context Battery Health Estimation for Unseen Battery Kinds

CS182/282 project — Ziv Behar, Jorgen Rokne Dahle, Junbo Fu, Kasper Andreas Haugaard.

**Question.** Can one network estimate the health of a lithium-ion cell of a *kind it
never trained on*, from a short clip of ordinary driving data, when it is shown a few
labelled clips from other cells of that kind?

**Health** is the charge a cell holds now as a percentage of what it held when new.
A **kind** is one chemistry and cell design (one dataset here).

**Current approach.** Following the professor's feedback we start from a pretrained
time-series foundation model (Amazon's Chronos-2) and fine-tune it, instead of
training an encoder from scratch. The claim to test is that this is easier and
faster for the same or better accuracy. See [docs/hypotheses.md](docs/hypotheses.md).

## Where to look

| Path | What it is |
|---|---|
| [docs/progress_log.md](docs/progress_log.md) | Week-by-week lab notebook: what we did, what we found, what is next |
| [docs/hypotheses.md](docs/hypotheses.md) | The hypotheses, each with its metric and pass condition |
| [docs/splits.md](docs/splits.md) | Which kinds train, validate and test, and why |
| [docs/standardization.md](docs/standardization.md) | How every dataset is brought to one common format and scale |
| [docs/datasets.md](docs/datasets.md) | The nine datasets: source, size, format, quirks |
| [docs/proposal/](docs/proposal/) | The original project proposal |
| [TODO.md](TODO.md) | Open tasks, with owners |
| [papers/](papers/) | Index of the papers we cite; `papers/download_papers.sh` fetches the PDFs |
| [configs/datasets.yaml](configs/datasets.yaml) | Per-dataset facts the code needs (capacity, voltage limits, split) |
| [src/battery_icl/](src/battery_icl/) | The library: dataset parsers, standardization, clips, models |
| [scripts/](scripts/) | Entry points: download, preprocess, train, evaluate, sync |
| `data/` | Not in git. `raw/` as downloaded, `processed/` standardized |
| `results/` | One folder per experiment run (metrics and plots are tracked, weights are not) |

## Getting started

```bash
git clone https://github.com/zivbeh/In-Context-Battery-Health-Estimation-for-Unseen-Battery-Kinds.git
cd In-Context-Battery-Health-Estimation-for-Unseen-Battery-Kinds
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # data cleaning only
pip install -e ".[dev,model]"    # plus torch + Chronos (GPU box)
pytest                           # sanity checks
```

The full datasets are tens of gigabytes, so keep them on a machine with room and
point `BATTERY_DATA` at that folder. To look at the data on a laptop, fetch one
small file per dataset:

```bash
scripts/download_data.sh samples
```

## Pipeline

```
scripts/download_data.sh   →  data/raw/<dataset>/          files as published
scripts/preprocess.py      →  data/processed/<dataset>/    one parquet per cell, common format
scripts/make_clips.py      →  data/processed/clips/        30-minute clips with health labels
scripts/run_e*.py, train.py →  results/<experiment>/       metrics per experiment
```

Parsers exist for the five kinds in use: `cmu`, `xjtu`, `stanford_lg`, `stanford_nca`
and `nasa`. Open work is in [TODO.md](TODO.md).

## Conventions

- **Splits are by kind.** Four development kinds rotate (leave one kind out) and NASA
  is the test kind. The split is fixed in `configs/datasets.yaml` and never changed
  after looking at test numbers.
- **Test sets stay closed** until the model and settings are frozen on validation.
- **Every run is reproducible**: its config, git commit and seed are saved next to its results.
- **Write it down.** A finished experiment gets an entry in the progress log the same day,
  including the ones that failed.
- **Code style**: short functions, plain names, comments that say *why*. No frameworks
  where a function will do.
