#!/usr/bin/env bash
# Download the raw datasets into $BATTERY_DATA/raw/<dataset>/ (default: ./data/raw).
#
#   scripts/download_data.sh nasa xjtu      # named datasets
#   scripts/download_data.sh all            # everything scriptable (~82 GB)
#   scripts/download_data.sh kit_2s         # KIT 2-second logs (70 GB, not in "all")
#   scripts/download_data.sh samples        # one small file per dataset, for laptops
#
# Finished files are skipped and partial ones resume, so re-running is safe.
# Sources, sizes and quirks are documented in docs/datasets.md.
set -uo pipefail

RAW="${BATTERY_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}/raw"
HERE="$(cd "$(dirname "$0")" && pwd)"

# fetch <url> <output file>: skip finished files, resume partial ones where the
# host allows it, and retry on flaky or rate-limiting hosts.
fetch() {
    [[ -s "$2" ]] && return 0
    mkdir -p "$(dirname "$2")"
    wget -c -nv --tries=8 --waitretry=60 --retry-on-http-error=429,500,502,503 -O "$2.part" "$1" \
        && mv "$2.part" "$2" || echo "FAILED: $1" >&2
}

nasa() {  # 1.07 GB, zip of seven zips of .mat
    fetch "https://phm-datasets.s3.amazonaws.com/NASA/11.+Randomized+Battery+Usage+Data+Set.zip" "$RAW/nasa/randomized_battery_usage.zip"
}

xjtu() {  # 2.44 GB, one zip with all six batches
    fetch "https://zenodo.org/api/records/10963339/files/Battery%20Dataset.zip/content" "$RAW/xjtu/battery_dataset.zip"
}

CMU_FILES="README.txt:26855063 VAH01:26855147 VAH02:26855078 VAH05:26855114 VAH06:26855138
VAH07:26855069 VAH09:26855105 VAH10:26855099 VAH11:26855117 VAH12:26855090 VAH13:26855252
VAH15:26855075 VAH16:26855126 VAH17:26855144 VAH20:26855120 VAH22:26855072 VAH23:26855066
VAH24:26855129 VAH25:26855102 VAH26:26855258 VAH27:26855141 VAH28:26855111 VAH30:26855096"

cmu() {  # 1.54 GB, one CSV per cell
    for entry in $CMU_FILES; do
        name="${entry%%:*}"; [[ "$name" == README.txt ]] || name="$name.csv"
        fetch "https://ndownloader.figshare.com/files/${entry##*:}" "$RAW/cmu/$name"
    done
}

NCA_URL="https://stacks.stanford.edu/file/druid:td676xr4322"

stanford_nca() {  # 17.1 GB, one CSV per cell; cells 018 and 083 do not exist
    for f in README.md Publishing_data/diagnostic_capacities.pkl Publishing_data/protocol_mapping_dic.json; do
        fetch "$NCA_URL/$f" "$RAW/stanford_nca/$(basename "$f")"
    done
    for n in $(seq -f "%03g" 3 96); do
        [[ "$n" == 018 || "$n" == 083 ]] && continue
        for kind in raw_data aging_summary; do
            fetch "$NCA_URL/Publishing_data/${kind}_cell_$n.csv" "$RAW/stanford_nca/${kind}_cell_$n.csv"
        done
    done
}

LG_URL="https://files.osf.io/v1/resources/qsabn/providers/dropbox"
LG_TOKEN="view_only=2a03b6c78ef14922a3e244f3d549de78"  # public token printed in the paper

stanford_lg() {  # 19.2 GB: the processed .mat files only (the raw xlsx are 296 GB)
    while read -r path; do
        fetch "$LG_URL$path?$LG_TOKEN" "$RAW/stanford_lg$path"
    done < "$HERE/../configs/stanford_lg_files.txt"
}

tum() {  # 32.5 GB, BaSyTec text files; rsync access is published on the landing page
    mkdir -p "$RAW/tum"
    RSYNC_PASSWORD=m1755598 rsync -a --partial --info=progress2 \
        rsync://m1755598@dataserv.ub.tum.de/m1755598/ "$RAW/tum/"
}

KIT_URL="https://radar.kit.edu/radar-backend/archives"

kit() {  # 7.2 GB: check-up results, cell configs, docs and the 30-second ageing log.
         # The host answers 429 to quick successive requests, hence the pauses.
    fetch "$KIT_URL/CoNQplSNoVeXExyV/retrieveFile/15a3gtgdzdeffbr8" "$RAW/kit/cell_eocv2.zip"; sleep 60
    fetch "$KIT_URL/CoNQplSNoVeXExyV/retrieveFile/w049egbgr21pfcm2" "$RAW/kit/cfg.zip"; sleep 60
    fetch "$KIT_URL/CoNQplSNoVeXExyV/retrieveFile/47dn6826v0sgdd87" "$RAW/kit/data_structure.xlsx"; sleep 60
    fetch "$KIT_URL/CoNQplSNoVeXExyV/retrieveFile/0y6afbb5xwzvhk1h" "$RAW/kit/readme_v2.pdf"; sleep 60
    fetch "$KIT_URL/kww7jv8ajuvchcah/retrieveFile/n4wqt07cnxnkab9z" "$RAW/kit/cell_log_age.7z"
}

kit_2s() {  # 70 GB: the 2-second log of all 228 cells. The host cannot resume.
    fetch "$KIT_URL/kww7jv8ajuvchcah/retrieveFile/my5gchu2ufy4bsf9" "$RAW/kit/cell_logext.7z"
}

lyon() {  # Dataverse; one zip holding every file of the dataset
    fetch "https://entrepot.recherche.data.gouv.fr/api/access/dataset/:persistentId/?persistentId=doi:10.57745/OLBXKT" "$RAW/lyon/lyon_a123.zip"
}

rwth() {
    echo "RWTH needs a browser: its host blocks scripted downloads." >&2
    echo "Get TimeSeriesData.zip from https://publications.rwth-aachen.de/record/815749 and put it in $RAW/rwth/" >&2
}

samples() {  # about 240 MB in total
    fetch "https://ndownloader.figshare.com/files/26855069" "$RAW/cmu/VAH07.csv"
    fetch "$NCA_URL/Publishing_data/raw_data_cell_016.csv" "$RAW/stanford_nca/raw_data_cell_016.csv"
    fetch "$NCA_URL/Publishing_data/diagnostic_capacities.pkl" "$RAW/stanford_nca/diagnostic_capacities.pkl"
    fetch "https://dataserv.ub.tum.de/public.php/dav/files/m1755598/highway_scenario/Aging_highway_CU_17.txt" "$RAW/tum/highway_scenario/Aging_highway_CU_17.txt"
    fetch "$KIT_URL/CoNQplSNoVeXExyV/retrieveFile/15a3gtgdzdeffbr8" "$RAW/kit/cell_eocv2.zip"
    echo "NASA and XJTU ship as single archives (1.1 and 2.4 GB): run 'download_data.sh nasa xjtu' if you have room."
}

[[ $# -eq 0 ]] && { sed -n '2,10p' "$0"; exit 1; }
for target in "$@"; do
    if [[ "$target" == all ]]; then
        for d in nasa xjtu cmu lyon kit stanford_nca stanford_lg tum rwth; do echo "== $d"; "$d"; done
    else
        echo "== $target"; "$target"
    fi
done
du -sh "$RAW"/* 2>/dev/null
