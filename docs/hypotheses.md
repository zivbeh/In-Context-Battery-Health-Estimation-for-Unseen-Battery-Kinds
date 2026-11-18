# Hypotheses

Status: **draft v3 (2026-11-18)**, rewritten after the professor's feedback and the new split. The team
should confirm the thresholds marked *(to confirm)* before any test-set number is looked at.

## Setup in one paragraph

A query is one clip of current and voltage from random or driving use. The model
outputs the cell's health in percent. It may also see K labelled clips from *other*
cells of the same kind. Four development kinds rotate, each held out in turn while the model trains on
the other three; NASA is the one test kind, scored once at the end
([splits.md](splits.md)). "Each held-out kind" below means the four development
folds while we develop, and NASA for the final result. The main number is
**MAE in points of health**, averaged within each cell and then across cells.
**Skill** is `1 − MAE_model / MAE_average-guess`, so 0 means no better than guessing
the mean and 1 means perfect. A difference counts only if its 95% bootstrap interval
over scored cells excludes zero.

## What changed from the proposal

The proposal trained an S4D encoder from scratch. We now start from **Chronos-2**
(Amazon's pretrained time-series model, 120M parameters) and fine-tune it. This
adds two hypotheses about fine-tuning (H1, H2) ahead of the in-context ones. The old
H3 ("NASA is the harder kind") had only two data points behind it, so it is now a
descriptive question rather than a hypothesis.

## Hypotheses

| | Claim | Measured by | Holds if | Experiment |
|---|---|---|---|---|
| **H1** | A pretrained encoder transfers to unseen kinds better than one trained from scratch. | MAE at K = 0 on each held-out kind: fine-tuned Chronos-2 vs the same architecture with random initial weights, same data and steps. | Fine-tuned MAE is lower on NASA and on at least three of the four folds. | E1 |
| **H2** | Fine-tuning is cheaper. | GPU-hours to reach the from-scratch model's best validation MAE. | Fine-tuning gets there in at most a quarter of the GPU-hours *(to confirm)*. | E1 |
| **H3** | With no examples, the model still transfers. | Skill and MAE at K = 0 on each held-out kind. | Skill > 0 on NASA at 70% health and above, and on at least three of the four folds. | E2 |
| **H4** | A few labelled examples help, and reading them in context is as good as fine-tuning on them. | MAE at K = 0, 1, 5, 20; and against the K = 0 model fine-tuned on the same K examples. | MAE at K = 5 is below MAE at K = 0; in-context is no worse than fine-tuning at K = 1 and 5. | E2 |
| **H5** | The model reads its examples rather than recognising the kind from the query. | Change in MAE when the K examples come from a different kind. | MAE with swapped examples is above the K = 0 MAE. | E3 |

## Descriptive questions (reported, not tested)

- How large is the error on NASA below 70% health, where no development kind has data?
- Which held-out kinds are harder, and does the error follow how far its current rates,
  temperature and health range sit from the training kinds?
- Where does the error concentrate: which current-rate bands and which health bands?
- What share of cells is over-reported by more than 5 points (the 2029 California limit)?
- How much of the gain comes from each fine-tuning level: frozen encoder with a new
  head, LoRA adapters, full fine-tuning?

## Experiments

- **E1 — Fine-tune vs scratch.** Same data, same head, same step budget. Arms: frozen
  Chronos-2 + head, LoRA, full fine-tune, random-init. Log validation MAE against
  GPU-hours. Three seeds.
- **E2 — Transfer with K examples.** Score each held-out kind at K = 0, 1, 5, 20 against
  the baselines (average guess, ridge on hand-built features, fine-tune on K, trained
  on the kind). Five draws of example cells.
- **E3 — Context swap.** Replace the examples with ones from each other kind, K and
  queries fixed.

## Baselines

| Baseline | Role |
|---|---|
| Average guess (mean health of the K examples) | Floor |
| Ridge regression on hand-built clip features | Cheap non-neural reference |
| K = 0 model fine-tuned on the K examples | Alternative to in-context examples |
| Network trained on the held-out kind (cell-disjoint split) | Ceiling |
