"""Load clips and their frozen Chronos-2 embeddings, computing each only once."""

import numpy as np
import torch

from .model import HealthRegressor, load_encoder
from .paths import PROCESSED_DIR


def load_clips(name: str) -> dict[str, np.ndarray]:
    """The clip file of one dataset: clips, health, cell, time_s."""
    return dict(np.load(PROCESSED_DIR / "clips" / f"{name}.npz"))


@torch.no_grad()
def frozen_embeddings(name: str, batch_size: int = 128) -> np.ndarray:
    """Frozen-encoder features of every clip of a dataset, cached on disk."""
    cache = PROCESSED_DIR / "embeddings" / f"{name}.npy"
    clip_file = PROCESSED_DIR / "clips" / f"{name}.npz"
    if cache.exists() and cache.stat().st_mtime > clip_file.stat().st_mtime:
        return np.load(cache)

    clips = load_clips(name)["clips"]
    model = HealthRegressor(load_encoder("frozen")).cuda().eval()
    features = []
    for start in range(0, len(clips), batch_size):
        batch = torch.from_numpy(clips[start : start + batch_size]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features.append(model.embed(batch).cpu().numpy())

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.concatenate(features))
    return np.load(cache)
