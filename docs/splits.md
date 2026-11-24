# Train, validation and test split

Status: **v2 (2026-11-18)**. Replaces the proposal's split of five training, two
validation and two test kinds. The split lives in
[configs/datasets.yaml](../configs/datasets.yaml) and is read by
[src/battery_icl/splits.py](../src/battery_icl/splits.py).

## Why the split had to change

The proposal planned nine kinds. Five are usable now:

| Kind | Cells | Use pattern | Lowest health reached per cell |
|---|---|---|---|
| `cmu` | 22 | eVTOL missions | 72% to 93% |
| `xjtu` | 8 | random-walk discharge | 79% to 85% |
| `stanford_lg` | 10 | UDDS driving | about 91% to 98% |
| `stanford_nca` | 92 | driving (8), synthetic (31), periodic (30), constant current (23) | 79% to 86% |
| `nasa` | 28 | random-walk current | 33% to 84% |

Added on 2026-11-23: `kit` (36 cells, driving profile at 0 to 40 °C, aged to about 40%; the
ageing logs are now parsed) is a development kind, and `tum` (three cells) is train-only,
never held out. Not used: `lyon` and `rwth` (not downloadable by script).

With five kinds, a fixed 5 / 2 / 2 split is impossible, and any fixed split leaves
one or two kinds to carry each conclusion.

## The split

```
test         nasa                                    locked until the method is frozen
development  kit, cmu, xjtu, stanford_lg, stanford_nca   leave one kind out, five folds
train only   tum
```

**Development: leave one kind out.** Each development kind is held out once. The
model trains on the other three and is scored on the held-out kind, which it has
never seen. This gives four measurements of transfer to an unseen kind instead of
one or two. Every choice (clip length, learning rate, LoRA rank, when to stop) is
made on the mean skill over the four folds.

**Test: NASA, once.** After the method is frozen, one model is trained on all four
development kinds and scored on NASA. NASA is the test kind because it is the
hardest and the least like the others: a chemistry no development kind has (LCO),
random current rather than a profile, and by far the widest health range.

**Cells inside a held-out kind.** One third of the cells supply in-context examples
and the other two thirds are scored. A cell is never both. The draw is fixed by a
seed, so K = 0, 1, 5 and 20 are all scored on the same cells.

**Balancing kinds.** Stanford NCA has more cells than the other three development
kinds together. Each training kind gets the same total weight, so the model cannot
do well by fitting NCA alone.

## What this costs, and what to watch

1. **Four trainings instead of one per experiment.** This is affordable only because
   fine-tuning is cheap, which is the professor's point. It makes a from-scratch
   comparison four times as expensive too; that cost is part of H2.
2. **The fold numbers are validation numbers.** They are used to choose settings, so
   they are slightly optimistic. Only the NASA number is a clean test result.
3. **One test kind.** The final claim rests on a single unseen kind. The four folds
   are the supporting evidence.
4. **Health range.** No development kind goes below 72% health; NASA goes to 33%.
   With no examples (K = 0) the model has never seen a label that low. NASA results
   will be reported in two bands: 70% and above, where development data exists, and
   below 70%, where the model must extrapolate. A failure in the lower band is a
   range problem, not evidence against transfer between kinds.
5. **Skill, not MAE, across folds.** Stanford LG spans under 10 points of health, so
   even a constant guess has a small MAE there. Skill puts the folds on one scale.

## Alternatives considered

| Option | Why not |
|---|---|
| Fixed split: train on 3 kinds, validate on 1, test on NASA | One validation kind decides every setting; three of four kinds never serve as unseen. |
| Stanford NCA as a second test kind, as in the proposal | Leaves three development kinds with 40 cells in total. |
| Leave one kind out over all five kinds, no locked test | Every number would have been used to choose settings. |
| Split by cell within kinds | Tests seen kinds, which is not the question. |

## Open choices

- **Stanford NCA when held out:** all 92 cells are scored for now; results for the
  eight recorded-driving cells are to be reported separately.
- **XJTU:** only Batch 5 (8 cells) is used. The dataset has 47 more cells of the same
  kind on constant and stepped currents that could be added to training.
- **KIT** would be the only development kind with deep ageing (to 40–50%). Its 7 GB
  log is the one extra download that would address point 4.
