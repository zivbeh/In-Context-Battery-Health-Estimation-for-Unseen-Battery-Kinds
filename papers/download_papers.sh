#!/usr/bin/env bash
# Fetch the cited papers into this folder. All sources are free copies.
# Number 13 (Schreiber et al.) needs a browser: https://doi.org/10.1016/j.est.2025.115357
cd "$(dirname "$0")"

get() {  # get <file name> <url>
    [[ -s "$1" ]] || curl -fsSL -A "curl" -o "$1" "$2" || echo "FAILED: $1" >&2
}

get 01_carb2022_advanced-clean-cars-ii-durability.pdf "https://web.archive.org/web/20240616172105id_/https://wiki.unece.org/download/attachments/172852349/EVE-57-13e%20-%20CARB%20ACCII%20-%20updated.pdf"
get 02_park2026_battery-health-reporting-fails-validation.pdf https://arxiv.org/pdf/2603.21592
get 03_che2025_diagnostic-free-onboard-health.pdf https://arxiv.org/pdf/2503.07383
get 04_feng2024_physics-guided-test-time-training.pdf https://arxiv.org/pdf/2402.00068
get 05_lanubile2024_domain-knowledge-guided-soh.pdf https://www.nature.com/articles/s44172-024-00304-2.pdf
get 06_richardson2019_gp-transition-model.pdf https://arxiv.org/pdf/1807.06350
get 07_wang2024_pinn-battery-degradation.pdf https://www.nature.com/articles/s41467-024-48779-z.pdf
get 08_tan2025_batterylife-dataset-benchmark.pdf https://arxiv.org/pdf/2502.18807
get 09_garg2022_what-can-transformers-learn-in-context.pdf https://arxiv.org/pdf/2208.01066
get 10_bole2014_electrochemistry-model-randomized-use.pdf https://papers.phmsociety.org/index.php/phmconf/article/download/2490/1452
get 11_geslin2025_dynamic-cycling-enhances-lifetime.pdf https://www.nature.com/articles/s41560-024-01675-8.pdf
get 12_pozzato2022_ev-real-driving-aging-dataset.pdf https://pmc-oa-opendata.s3.amazonaws.com/PMC8892076.1/PMC8892076.1.pdf
get 14_luh2024_comprehensive-battery-aging-dataset.pdf https://www.nature.com/articles/s41597-024-03831-x.pdf
get 15_gu2022_diagonal-state-space-models.pdf https://arxiv.org/pdf/2206.11893
get 16_chen2026_reference-selection-cross-condition-soh.pdf https://mdpi-res.com/d_attachment/wevj/wevj-17-00582/article_deploy/wevj-17-00582.pdf
get 17_wheeler2025_twenty-a123-graphite-lfp-cells.pdf https://www.nature.com/articles/s41597-025-04712-7.pdf
get 18_bills2023_evtol-battery-dataset.pdf https://www.nature.com/articles/s41597-023-02180-5.pdf
get 19_jost2021_drive-cycle-aging-28-nca-cells-dataset-metadata.pdf "https://web.archive.org/web/20230514210623id_/https://publications.rwth-aachen.de/record/815749/files/Metadata.pdf"
get 20_ansari2024_chronos-language-of-time-series.pdf https://arxiv.org/pdf/2403.07815
get 21_ansari2025_chronos-2-univariate-to-universal.pdf https://arxiv.org/pdf/2510.15821
ls -1 *.pdf | wc -l
