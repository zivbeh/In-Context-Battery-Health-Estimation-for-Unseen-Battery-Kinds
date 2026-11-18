"""Cut every standardized cell into clips and save one file per dataset.

    python scripts/make_clips.py cmu nasa --minutes 30 --per-cell 500

Writes data/processed/clips/<dataset>.npz with
    clips   (n, 2, length)  current_c and voltage_norm
    health  (n,)            label in [0, 1]
    cell    (n,)            cell id of each clip
    time_s  (n,)            test time at the middle of each clip
"""

import argparse

import numpy as np
import pandas as pd

from battery_icl.clips import cut_clips, spread_evenly
from battery_icl.paths import PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--minutes", type=float, default=30, help="clip length")
    parser.add_argument("--per-cell", type=int, default=500, help="most clips kept per cell")
    args = parser.parse_args()

    out_dir = PROCESSED_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        parts = {"clips": [], "health": [], "cell": [], "time_s": []}
        for path in sorted((PROCESSED_DIR / name).glob("*.parquet")):
            table = pd.read_parquet(path)
            dt_s = table["time_s"].diff().median()
            clips, health, time_s = cut_clips(table, clip_len=int(args.minutes * 60 / dt_s))
            keep = spread_evenly(len(clips), args.per_cell)
            parts["clips"].append(clips[keep])
            parts["health"].append(health[keep])
            parts["time_s"].append(time_s[keep])
            parts["cell"].append(np.full(len(keep), path.stem))

        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        cells = len(set(arrays["cell"]))
        print(f"{name}: {len(arrays['health'])} clips from {cells} cells, "
              f"health {arrays['health'].min():.2f} to {arrays['health'].max():.2f}")


if __name__ == "__main__":
    main()
