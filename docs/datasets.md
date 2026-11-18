# Datasets

Nine public ageing datasets, one per kind of cell. **Five are in use** (cmu, xjtu,
stanford_lg, stanford_nca, nasa); the other four are documented here in case we come
back to them. `scripts/download_data.sh` fetches them into `data/raw/<name>/`. The numbers the code uses are in
[configs/datasets.yaml](../configs/datasets.yaml). Last checked 2026-11-18.

## Overview

| Name | Proposal split | Cells | Chemistry | Use pattern | Download | Format | Scripted? |
|---|---|---|---|---|---|---|---|
| `kit` | train | 36 of 228 | NMC/C-SiO 18650, 3.0 Ah | WLTP profile at 0/10/25/40 °C | 7.2 GB (30 s log) + 70 GB (2 s log) | 7z/zip of `;` CSV | yes |
| `tum` | train | **3** | NMC pouch, 78 Ah | highway, commuter, 1C/1C | 32.5 GB | BaSyTec text | yes (rsync) |
| `lyon` | train | 20 | LFP 18650, 1.1 Ah | WLTP at 50 °C | unknown | zip of Arbin text | **host unreachable** |
| `xjtu` | train | 8 of 55 | NCM523 18650, 2.0 Ah | random-walk discharge | 2.4 GB | `.mat` per cell | yes |
| `cmu` | train | 22 | VTC6 18650, 3.0 Ah | eVTOL missions | 1.5 GB | CSV per cell | yes |
| `rwth` | validation | 28 | NCA/C+Si 18650 | recorded drive cycle | about 9 GB | zip of CSV | **needs a browser** |
| `stanford_lg` | validation | 10 | NMC 21700, 4.85 Ah | UDDS | 19.2 GB processed (315 GB raw) | `.mat` | yes |
| `nasa` | test | 28 | LCO 18650, about 2.1 Ah | random-walk current | 1.1 GB | zip of zips of `.mat` | yes |
| `stanford_nca` | test | 92 | NCA/SiOx-graphite | driving, periodic, synthetic, constant | 17.1 GB | CSV per cell | yes |

The split now in force is in [splits.md](splits.md).

## Things that differ from the proposal

1. **TUM has three cells, not a fleet.** One cell per scenario. It fails our own
   inclusion rule of at least 8 cells. Either drop it or keep it as a small extra training kind.
2. **Only 8 of Stanford's 92 NCA cells use recorded driving profiles** (cells 089–096).
   The rest are constant-current, periodic or synthetic profiles. The test set is
   smaller than planned if we keep "driving" strict.
3. **KIT's convenient log is sampled every 30 s**, too coarse for a driving profile.
   The 2 s log is one 70 GB archive for all 228 cells.
4. **Stanford LG raw data is 315 GB.** We take the authors' processed `.mat` files (19.2 GB).
5. **Four datasets are not in use.** Lyon's repository does not answer. RWTH has a bot
   check and needs a browser. KIT's host rate-limits, so only its check-up results are
   on disk. TUM is 23 of 32.5 GB downloaded (commuter and 1C/1C cells, no highway cell).
6. **Stanford LG ages only to 91–98% health**, a span of under 10 points.
7. **XJTU's first capacity test comes after 20 cycles**, so its health label starts late.

## Per dataset

### kit — Luh & Blank 2024
- Source: RADAR4KIT, records `CoNQplSNoVeXExyV` (results) and `kww7jv8ajuvchcah` (logs). CC BY 4.0.
- Profile cells are parameter sets P065–P076, three replicates each.
- Reference capacity: `cap_aged_est_Ah` on check-up discharge rows of the `eoc` files (C/3, room temperature, about every three weeks).
- Do not use the `soh_cap` column: it is rescaled, not capacity over nominal.
- The host cannot resume downloads and rate-limits quick requests.

### tum — Schreiber et al. 2025
- Source: mediaTUM record 1755598 (the "Update" record; the older one is incomplete). CC BY 4.0.
- Check-ups (`_CU_`) and cycling (`_cyc_`) are separate files. Read as latin-1, 12 header lines starting with `~`.
- Sampling is event-based: 1 s to 30 s.
- Reference capacity: `cap_dis_measure`, apparently in ampere-seconds (unconfirmed).

### lyon — Wheeler et al. 2025
- Source: doi:10.57745/OLBXKT (Recherche Data Gouv). One zip per cell, Arbin text.
- Reference capacity: `Discharge_Capacity` at the end of step 11 in the characterisation files, every 100 cycles.
- Case 6 (cells 6a–c) is constant-current, not WLTP. No temperature channel.

### xjtu — Wang et al. 2024
- Source: Zenodo 10963339. CC BY 4.0. One zip, six batches; we use `Batch-5/RW_battery-*.mat`.
- 1 Hz. One struct per cycle: `relative_time_min, voltage_V, current_A, capacity_Ah, temperature_C, description`.
- Reference capacity: cycles whose description contains `test capacity` (one every 10 random-walk cycles, from cycle 21).

### cmu — Bills et al. 2023
- Source: CMU KiltHub 14226830. CC BY 4.0. One CSV per cell for its whole life; current is in **mA**.
- Reference capacity: C/5 discharge 4.2 → 2.5 V every 50 missions; no flag column, find it from `Ns` and the current level.
- Known faults per cell are listed in the dataset README (VAH05, 09, 10, 11, 13, 25, 26, 27, 28).

### rwth — Jöst et al. 2021
- Source: RWTH Publications record 815749. The 4TU record holds no files.
- Check-ups every 30 cycles sit in the same series. Column names not yet seen.

### stanford_lg — Pozzato et al. 2022
- Source: OSF `qsabn` with the view-only token printed in the paper. CC BY 4.0.
- Ageing and reference tests are separate trees; capacity test is a C/20 discharge every 25 or 50 cycles.
- Reference coverage is uneven: W3 and W7 have 3–4 diagnostics, W5/W8/W9/W10 have all 15.

### nasa — Randomized Battery Usage
- Source: NASA PCoE repository, entry 11. No formal licence; acknowledge NASA.
- One `.mat` per cell (RW1–RW28) inside seven nested zips, grouped by current distribution and temperature.
- Reference capacity: integrate current over the `reference discharge` steps.

### stanford_nca — Geslin et al. 2025
- Source: Stanford Digital Repository `td676xr4322`. CC BY 4.0.
- Current is already a C-rate and capacity is already divided by nominal. 1 s sampling.
- Reference capacity: `diagnostic_capacities.pkl` (C/40 and C/2), keyed by the cycle number of each diagnostic.
- `protocol_mapping_dic.json` says which profile family each cell ran.
