# TODO

Put your name on a task when you take it. Move finished tasks to the progress log.

## Decisions the team needs to make

- [ ] **Confirm the new split** in [docs/splits.md](docs/splits.md): NASA as the only test kind, the other four rotating.
- [ ] **Stanford NCA when held out:** score all 92 cells, or only the 8 recorded-driving cells (089–096)?
- [ ] **Deep-ageing training data.** No development kind goes below 72% health; NASA goes to 33%. Fetch KIT's 7 GB log (ages to 40–50%) or accept the range gap and report NASA in two bands?
- [ ] **Time grid.** 10 s is the finest grid all datasets support. Is losing sub-10 s driving dynamics acceptable? (See E-std below.)
- [ ] Confirm the thresholds in [docs/hypotheses.md](docs/hypotheses.md), in particular H2's "a quarter of the GPU-hours".

## Data

- [ ] XJTU: add Batches 1–4 and 6 (47 more cells of the same kind) to training?
- [ ] XJTU: fix the health label that exceeds 100% (see [docs/standardization.md](docs/standardization.md)).
- [ ] Stanford LG: check one cell's processed `.mat` against its raw Excel files.
- [ ] Confirm datasheet voltage limits for the kinds marked `verified: false` in `configs/datasets.yaml`.
- [ ] Get the Schreiber et al. PDF by hand (see [papers/README.md](papers/README.md)).
- Parked (not needed for the current split): KIT, TUM, Lyon, RWTH. Download notes are in [docs/datasets.md](docs/datasets.md).

## Standardization

- [ ] **E-std**: on validation kinds with 1 s data, compare a 1 s grid against the 10 s grid.
- [ ] Clip CMU VAH27's voltage spike, or drop clips with `voltage_norm` outside −0.1 to 1.1.
- [ ] Plot one standardized clip per kind side by side (a figure for the report).
- [ ] Decide whether temperature becomes a third input channel.

## Modelling

- [ ] E1: three seeds per mode and fold, and tune learning rate and steps on mean fold skill.
- [ ] Per-kind input scaling or other fixes for the K = 0 transfer gap found in E1 (see progress log).
- [ ] In-context head: a small transformer over clip embeddings and example labels, trained with mixed K (E2).
- [ ] Alternative to test: put the K example clips in the same Chronos-2 group as the query and let its group attention do the in-context step.
- [ ] Baselines: average guess, ridge on hand-built features, fine-tune on K, trained on the kind.
- [ ] `scripts/evaluate.py`: MAE, RMSE, skill, share over-reported by more than 5 points, 95% bootstrap intervals over cells.
- [ ] Context-swap experiment (E3).

## Housekeeping

- [ ] Save config, git commit and seed next to every result.
- [ ] Fix the reference list in the report (corrections in [papers/README.md](papers/README.md)).
