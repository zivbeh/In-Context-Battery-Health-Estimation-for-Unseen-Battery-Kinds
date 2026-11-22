# Progress log

Newest week first. Each week: what we set out to do, what we did, what we found,
what is next. Numbers quoted here must point to a file in `results/` or a script
that reproduces them. Hypotheses are in [hypotheses.md](hypotheses.md).

---

## Week of 16 November 2026

### Direction from the professor

1. Start from Amazon's time-series foundation model and fine-tune it, to show the
   task can be done more easily and faster than with full training.
2. Standardize the data well.
3. Keep a week-by-week written record with cleaner hypotheses (this file).

### Goals for the week

- Set the repository up so the whole team can clone it and work.
- Get the data onto the GPU box and into one common format.
- Check that Chronos-2 can be used for our task at all.

### What was done

**Repository.** Layout, README, TODO, hypotheses, dataset and standardization docs,
are in place. 20 of the 21 cited papers can be fetched with `papers/download_papers.sh`.

**Hypotheses rewritten.** Five hypotheses, each with a metric and a pass condition.
H1 and H2 are new and test the professor's point (pretrained vs from scratch, and
cost). "NASA is the harder kind" moved to a descriptive question because two test
kinds cannot support it.

**Model choice.** Chronos-2 (`amazon/chronos-2`, 120M parameters, Apache-2.0). Its
encoder embeds a clip and a small head outputs health. One code path covers four
modes: frozen, LoRA, full fine-tune, and random-init as the from-scratch control.
All four run on a 16 GB GPU:

| Mode | Trainable parameters | GPU memory, batch of 8 |
|---|---|---|
| frozen | 0.39 M | 0.6 GB |
| LoRA (rank 8) | 0.98 M | 1.6 GB |
| full | 119.9 M | 2.0 GB |
| scratch | 119.9 M | 3.2 GB |

**Data.** Download status on the GPU machine:

| Dataset | Split | Status |
|---|---|---|
| cmu | train | downloaded, standardized, clips cut |
| nasa | test | downloaded, standardized, clips cut |
| stanford_nca | test | downloaded, standardized, clips cut |
| xjtu | train | downloading (slow host) |
| stanford_lg | validation | downloading (19.2 GB of processed files) |
| tum | train | downloading (32.5 GB) |
| kit | train | check-up results downloaded; the 7 GB log is rate-limited, retry loop running |
| lyon | train | repository host does not answer |
| rwth | validation | host blocks scripted downloads; needs a browser |

**Standardization.** One format for every kind: current as C-rate, voltage scaled by
datasheet limits, a 10 s grid, health relative to the cell's first reference test.
Details and reasons in [standardization.md](standardization.md).

| Dataset | Cells | Reference tests per cell | Lowest health reached per cell | Clips (30 min) |
|---|---|---|---|---|
| cmu | 22 | 6 to 47 | 72% to 93% | 11,000 |
| nasa | 28 | 10 to 80 | 33% to 84% | 14,000 |
| stanford_nca | 92 | median 16 to 24 by profile family | 79% to 86% | 46,000 |

Source: `data/processed/<dataset>/summary.csv`, written by `scripts/preprocess.py`.

### Findings

**F1. The datasets differ from what the proposal assumed.**
- TUM has 3 cells, not a fleet.
- Only 8 of Stanford's 92 NCA cells ran recorded driving; the rest are constant
  current (23), periodic (30) or synthetic (31) profiles.
- Stanford NCA cells reach 79–86% health; the proposal said 85%.
- NASA cells reach 33–84%, so many stop well above the proposal's "about 38%".
- KIT's small log is sampled every 30 s; the 2 s log is a single 70 GB archive.

**F2. Chronos-2 removes the signal level.** It rescales each input series by its own
mean and standard deviation, which would discard the mean voltage. We pass both
numbers to the head.

**F3. Automatic checks caught three labelling bugs** (aborted NASA reference tests,
a merged CMU test, a missed CMU discharge leg). All fixed; all 142 standardized
cells now pass.

**F4. Sanity check E0: frozen Chronos-2 embeddings carry health information.**
Frozen encoder, ridge regression on the embeddings, CMU only, scored on cells not
used for fitting (five-fold by cell).

| | MAE (points of health) | Skill |
|---|---|---|
| Average guess | 4.94 | 0 |
| Frozen Chronos-2 + ridge | 2.66 | 0.46 |

Source: `results/e0_frozen_sanity/cmu.json`, from `scripts/run_e0_frozen_sanity.py cmu`
(under one minute on the GPU). This is a seen-kind check on one training kind with
one run and no confidence interval. It shows the pipeline works; it says nothing yet
about unseen kinds. No test-set label has been used to fit or score a model.

**22 November 2026 — Same-cell x-shot pilot before E0.** We rebuilt the 500
standard 30-minute clips for CMU cell VAH07 from its downloaded CSV. Six reference
capacity tests provided the interpolated clip labels. We selected eight support
clips across the first half of elapsed time, then gave frozen Chronos-2 the latest
1, 2, 4 or 8 of those clips. Each run estimated SoH for the same 255 later clips.
The checkpoint received the query current and voltage after the clip was observed.

| Shots | Chronos-2 with current and voltage | Chronos-2 with SoH history only | Last known SoH |
|---:|---:|---:|---:|
| 1 | 1.423 | 1.413 | 1.413 |
| 2 | 1.471 | 1.334 | 1.413 |
| 4 | 1.330 | 1.170 | 1.413 |
| 8 | 1.377 | 1.146 | 1.413 |

Errors are mean absolute errors in SoH percentage points. On this cell the
electrical inputs increased error at every shot count relative to the history-only
forecast. The history-only forecast was constant across all queries at a given
shot count because every query used the same support and one-step horizon.
These are retrospective estimates using the cell's peak diagnostic capacity for
normalization. [Run summary](../results/pre_e0_same_cell_xshot/cmu/VAH07/summary.json)
and [query predictions](../results/pre_e0_same_cell_xshot/cmu/VAH07/predictions.csv).

### Open questions

See "Decisions the team needs to make" in [TODO.md](../TODO.md): what to do with TUM,
which Stanford NCA profiles count as test, the 10 s grid, and the KIT 2 s log.

### Next week

1. Finish downloads; write parsers for xjtu, stanford_lg, kit.
2. `scripts/train.py` and E1 (fine-tune vs scratch), scored on the validation kind.
3. First cross-kind number among training kinds (train on cmu + xjtu, score on kit).
