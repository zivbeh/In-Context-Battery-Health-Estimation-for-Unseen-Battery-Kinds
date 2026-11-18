# Papers

Numbers match the reference list in the proposal; 20 and 21 are additions.
The PDFs are not stored in git. Run `papers/download_papers.sh` to fetch them into
this folder; all sources are free copies (arXiv, open-access publishers, archives).

## Motivation and regulation

| # | Paper | File |
|---|---|---|
| 1 | CARB (2022). Durability Requirements in California Advanced Clean Cars II. UNECE EVE-57-13e. | `01_carb2022_advanced-clean-cars-ii-durability.pdf` |
| 2 | Park et al. (2026). Battery health reporting fails independent validation across manufacturers. arXiv:2603.21592 | `02_park2026_battery-health-reporting-fails-validation.pdf` |

## Health estimation: prior results we compare against

| # | Paper | Setting | File |
|---|---|---|---|
| 3 | Che et al. (2025). Diagnostic-free onboard battery health assessment. arXiv:2503.07383 | Seen kind, partial charging and driving | `03_che2025_diagnostic-free-onboard-health.pdf` |
| 4 | Feng et al. (2024). Cross domain Li-ion battery health estimation via physics-guided test-time training. arXiv:2402.00068 | New kind, steady charging | `04_feng2024_physics-guided-test-time-training.pdf` |
| 5 | Lanubile et al. (2024). Domain knowledge-guided ML framework for state of health estimation. Commun. Eng. | Seen kind, driving | `05_lanubile2024_domain-knowledge-guided-soh.pdf` |
| 6 | Richardson et al. (2019). Battery health prediction under generalized conditions using a GP transition model. J. Energy Storage (arXiv preprint) | Seen kind, random use (NASA) | `06_richardson2019_gp-transition-model.pdf` |
| 7 | Wang et al. (2024). Physics-informed neural network for battery degradation. Nat. Commun. 15 | Seen kind, steady charging; XJTU dataset | `07_wang2024_pinn-battery-degradation.pdf` |
| 8 | Tan et al. (2025). BatteryLife: dataset and benchmark for battery life prediction. KDD | Cycle-life prediction, new kinds | `08_tan2025_batterylife-dataset-benchmark.pdf` |
| 16 | Chen et al. (2026). Reference selection for cross-condition SOH estimation. World Electr. Veh. J. 17(11), 582 | Seen kind, random use (NASA) | `16_chen2026_reference-selection-cross-condition-soh.pdf` |

## Datasets

| # | Paper | Our split | File |
|---|---|---|---|
| 10 | Bole, Kulkarni, Daigle (2014). Electrochemistry-based model under randomized use. PHM Conf. (NASA Randomized Battery Usage) | Test | `10_bole2014_electrochemistry-model-randomized-use.pdf` |
| 11 | Geslin et al. (2025). Dynamic cycling enhances battery lifetime. Nat. Energy 10, 172–180 | Test | `11_geslin2025_dynamic-cycling-enhances-lifetime.pdf` |
| 12 | Pozzato, Allam, Onori (2022). Aging dataset based on EV real-driving profiles. Data in Brief 41, 107995 | Validation | `12_pozzato2022_ev-real-driving-aging-dataset.pdf` |
| 13 | Schreiber et al. (2025). Degradation in vehicle applications: VW ID.3 pouch cells. J. Energy Storage 112, 115357 | Train | **missing** — open access but blocks scripted download; get it in a browser from https://doi.org/10.1016/j.est.2025.115357 |
| 14 | Luh & Blank (2024). Comprehensive battery aging dataset (NMC/C-SiO). Sci. Data 11, 1004 | Train | `14_luh2024_comprehensive-battery-aging-dataset.pdf` |
| 17 | Wheeler et al. (2025). An ageing study of twenty 18650 Graphite/LFP cells in first and second life use. Sci. Data 12, 392 | Train | `17_wheeler2025_twenty-a123-graphite-lfp-cells.pdf` |
| 18 | Bills et al. (2023). A battery dataset for eVTOL aircraft. Sci. Data 10, 344 | Train | `18_bills2023_evtol-battery-dataset.pdf` |
| 19 | Jöst et al. (2021). Drive cycle aging test of 28 NCA/C+Si 18650 cells (dataset metadata sheet, not a paper) | Validation | `19_jost2021_drive-cycle-aging-28-nca-cells-dataset-metadata.pdf` |

## Methods

| # | Paper | File |
|---|---|---|
| 9 | Garg et al. (2022). What can transformers learn in-context? NeurIPS | `09_garg2022_what-can-transformers-learn-in-context.pdf` |
| 15 | Gu et al. (2022). Diagonal state space models (S4D). NeurIPS | `15_gu2022_diagonal-state-space-models.pdf` |
| 20 | Ansari et al. (2024). Chronos: Learning the language of time series. TMLR | `20_ansari2024_chronos-language-of-time-series.pdf` |
| 21 | Ansari et al. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821 | `21_ansari2025_chronos-2-univariate-to-universal.pdf` |

## Corrections to the proposal's reference list

- [1] is a 2022 slide deck titled "Durability Requirements in California Advanced Clean Cars II (ACC II) Program".
- [17]'s published title is "An ageing study of twenty 18650 lithium-ion Graphite/LFP cells in first and second life use".
- [18] is article number 344.
- [11] has an Author Correction dated 11 November 2026 (doi:10.1038/s41560-026-02182-8), not yet read.
