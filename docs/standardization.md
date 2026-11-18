# Standardization

The nine datasets come from different labs, testers, cell sizes and chemistries.
A model can only transfer between kinds if the same physical situation produces the
same numbers in every dataset. This page states each choice and why. The code is
[src/battery_icl/standardize.py](../src/battery_icl/standardize.py) and
[src/battery_icl/clips.py](../src/battery_icl/clips.py).

## The standardized table (one parquet per cell)

| Column | Meaning | How it is made |
|---|---|---|
| `time_s` | seconds since the start of the cell's test | uniform grid, 10 s |
| `current_c` | current as a C-rate, **charge positive** | amps ÷ nominal capacity (datasheet) |
| `voltage_norm` | voltage on a 0–1 scale | (V − V_min) ÷ (V_max − V_min), datasheet limits |
| `temperature_c` | cell temperature | as logged; NaN where the dataset has none |
| `health` | capacity ÷ the cell's first reference capacity | linear in time between reference tests |
| `block` | id of an uninterrupted stretch of ageing use | new block after a logging gap over 5 min |

## Choices

**1. Current as C-rate.** A 78 Ah pouch cell and a 2 Ah 18650 at "1C" are doing the
same thing. Dividing by the *nominal* capacity (a datasheet constant) keeps the
scale fixed over the cell's life. Dividing by the present capacity would leak the label.

**2. Voltage scaled by datasheet limits.** LFP lives between 2.0 and 3.65 V, NMC
between 2.5 and 4.2 V. Scaling by each kind's own limits puts "empty" at 0 and
"full" at 1 for every chemistry. The limits are constants per kind, never fitted to data.

**3. Sign convention.** Charge is positive. NASA logs the opposite and is flipped in its parser.

**4. One time grid, 10 s, by averaging.** Loggers differ: NASA and Stanford record
1 s during use, CMU records missions every 10 s, KIT's compact log is 30 s, and
several testers slow down when nothing changes. Chronos-2 has no notion of sample
spacing, so all kinds must share one grid. Signals are interpolated to 1 s and then
averaged over 10 s bins; averaging keeps the charge passed in each bin correct.
10 s is the finest grid every downloaded dataset supports. **Open question:** this
discards sub-10 s dynamics of driving profiles. To be tested on validation kinds
that have 1 s data (E-std in the TODO list).

**5. Health label.** Each reference test gives a capacity. Health is that capacity
divided by the *same cell's first* reference capacity, so every cell starts at 1.0
whatever its reference protocol. Between tests the label is interpolated linearly in
time. Data before the first or after the last test gets no label; we do not extrapolate.

**6. Reference tests are removed from the input.** A clip must look like ordinary
use. Each parser separates ageing use from reference tests, and the gap a test
leaves starts a new block, so no clip spans one.

**7. Clips.** Non-overlapping 30-minute windows (180 samples) of `current_c` and
`voltage_norm` inside one block. A window is dropped if the cell rests for more than
half of it. The label is the health at the middle of the window. At most 500 clips
per cell, spread evenly over its life, so long-lived cells do not dominate.

**8. What the model sees.** Chronos-2 rescales every input series by its own mean
and standard deviation. That removes exactly the levels we standardized (for
example mean voltage, which moves with health). The mean and standard deviation of
each channel are therefore passed to the prediction head next to the embedding
([model.py](../src/battery_icl/model.py)).

## Reference test used per dataset

Reference rates differ between labs. Because health is relative to the cell's own
first test at the same rate, this affects the label less than it affects absolute capacity.

| Dataset | Reference capacity | Found by |
|---|---|---|
| `cmu` | C/5 discharge, 4.2 → 2.5 V, every 50 missions | long C/5 runs in the current trace |
| `nasa` | 1 A discharge, 4.2 → 3.2 V | steps commented `reference discharge` that reach the cut-off |
| `stanford_nca` | C/2 constant-current discharge | authors' `diagnostic_capacities.pkl` |
| `xjtu` | 1C discharge, 4.2 → 2.5 V, every 10 random-walk cycles from cycle 21 | cycles described as `[test capacity]` |
| `stanford_lg` | C/20 discharge at each diagnostic | authors' `capacity_test.mat` |

## Automatic checks

`standardize.check` flags a cell when more than 1% of the voltage lies outside the
datasheet limits, when current exceeds 20C, when any health value lies outside
20–110%, or when no row has a label. `scripts/preprocess.py` writes the result per
cell to `data/processed/<dataset>/summary.csv`.

Problems these checks caught so far:

- NASA RW17–19: aborted reference discharges gave health near 1%. Fixed by keeping
  only reference discharges that reach the cut-off voltage.
- CMU VAH09: an aborted capacity test merged with its repeat (3.95 Ah on a 3 Ah
  cell). Fixed by rejecting capacities above 3.4 Ah.
- CMU: the capacity test has three discharge legs; the first version missed the
  one-hour leg and under-read capacity by 0.6 Ah.

## Known gaps

- **XJTU health goes above 100%** (up to 104%). Its first capacity test comes after 20
  cycles and capacity then rises for a while, so "relative to the first test" is not
  "relative to new" for this kind. Dividing by the largest early capacity would fix it.
- Stanford LG is read at 1 s by point sampling from 0.1 s data, and its campaigns are
  laid end to end because the files carry no shared clock.
- Temperature is not used as an input yet, and two of the five kinds do not log it.
- CMU VAH27 has an unphysical voltage spike (`voltage_norm` 1.65) that is not clipped.
- Stanford NCA voltage limits (2.8–4.2 V) are read off the data, not a datasheet.
- Reference rates are not harmonised across kinds (C/40 to 1C).
