"""Two Chronos-2 paths for state-of-health estimation.

ChronosHealthPredictor uses the native forecast head with labelled support clips
to predict the health of an unlabelled query clip.

The encoder experiments use load_encoder to configure the model and
HealthRegressor to extract clip embeddings or train a separate regression head.
count_trainable reports the parameter counts for those training comparisons.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from .clips import CHANNELS

if TYPE_CHECKING:
    from chronos import Chronos2Pipeline

CHRONOS_ID = "amazon/chronos-2"
FINETUNE_MODES = ("frozen", "lora", "full", "scratch")


# Native forecast head with labelled support clips.


class ChronosHealthPredictor:
    """Use a pretrained or adapted Chronos-2 pipeline to predict query health.

    Each step represents one clip. The observed steps contain support features
    and health labels. The next step contains query features, and the native
    forecast head estimates its health.
    """

    FEATURE_NAMES = tuple(
        f"{channel}_{segment}"
        for channel in CHANNELS
        for segment in range(1, 7)
    )

    def __init__(self, pipeline: "Chronos2Pipeline") -> None:
        """Use a complete Chronos-2 pipeline in evaluation mode."""
        self.pipeline = pipeline
        self.pipeline.model.eval()

    @classmethod
    def from_pretrained(
        cls, checkpoint: str = CHRONOS_ID, *, device: str = "cpu"
    ) -> "ChronosHealthPredictor":
        """Load the stock checkpoint or a compatible saved model."""
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(checkpoint, device_map=device)
        return cls(pipeline)

    @staticmethod
    def _clip_features(clips: np.ndarray) -> np.ndarray:
        """Average each channel in six five-minute segments on the 10 s grid."""
        clips = np.asarray(clips, dtype=np.float32)
        if clips.ndim != 3 or clips.shape[1:] != (2, 180):
            raise ValueError("Expected clips with shape (n_clips, 2, 180).")
        if not np.isfinite(clips).all():
            raise ValueError("Clips must contain finite current and voltage values.")

        segments = clips.reshape(len(clips), 2, 6, 30)
        return segments.mean(axis=-1).reshape(len(clips), 12)

    @torch.inference_mode()
    def predict(
        self,
        support_clips: np.ndarray,
        support_health: np.ndarray,
        query_clips: np.ndarray,
    ) -> np.ndarray:
        """Return one median health estimate per query, in fraction units.

        The caller selects support and query cells. For one shot on a held-out
        kind, the support can include a source bank and one clip of that kind.
        """
        support_features = self._clip_features(support_clips)
        query_features = self._clip_features(query_clips)
        health = np.asarray(support_health, dtype=np.float32)
        if len(support_features) == 0 or health.shape != (len(support_features),):
            raise ValueError("Provide one health label per support clip.")
        if not np.isfinite(health).all():
            raise ValueError("Support health labels must be finite.")
        if len(query_features) == 0:
            return np.empty(0, dtype=np.float32)

        past_covariates = dict(zip(self.FEATURE_NAMES, support_features.T))
        inputs = [
            {
                "target": health,
                "past_covariates": past_covariates,
                "future_covariates": {
                    name: row[index : index + 1]
                    for index, name in enumerate(self.FEATURE_NAMES)
                },
            }
            for row in query_features
        ]

        quantiles, _ = self.pipeline.predict_quantiles(
            inputs, prediction_length=1, quantile_levels=[0.5]
        )
        # A single target, forecast step and quantile are returned for each query.
        return np.asarray([forecast[0, 0, 0].item() for forecast in quantiles], dtype=np.float32)


# Encoder features and a separate regression head.


def load_encoder(mode: str = "lora", lora_rank: int = 8) -> nn.Module:
    """Prepare the encoder for the embedding and custom-head experiments.

    Frozen keeps the pretrained encoder fixed. LoRA trains attention adapters.
    Full updates the pretrained weights. Scratch starts from random weights as
    the control for pretraining.
    """
    from chronos import Chronos2Pipeline

    if mode not in FINETUNE_MODES:
        raise ValueError(f"mode must be one of {FINETUNE_MODES}, got {mode!r}")

    encoder = Chronos2Pipeline.from_pretrained(CHRONOS_ID).model
    if mode == "scratch":
        encoder = type(encoder)(encoder.config)  # fresh random weights
    elif mode == "frozen":
        encoder.requires_grad_(False)
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model

        # Adapters on the attention projections; everything else stays frozen.
        config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules=["q", "v"])
        encoder = get_peft_model(encoder, config)
    return encoder


class HealthRegressor(nn.Module):
    """Use Chronos-2 embeddings with a separately trained health readout.

    The frozen ridge experiments use embed() to get features. The training
    script uses forward() to predict health fractions with the PyTorch head.
    """

    def __init__(self, encoder: nn.Module, n_channels: int = 2, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.config.d_model
        # Per channel: pooled embedding plus that channel's mean and std.
        self.head = nn.Sequential(
            nn.Linear(n_channels * (d_model + 2), hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def embed(self, clips: torch.Tensor) -> torch.Tensor:
        """Pool patch embeddings and retain each channel's center and scale."""
        batch, channels, _ = clips.shape
        rows = clips.flatten(0, 1)  # (batch * channels, time)
        # Current and voltage in the same clip can attend to each other.
        groups = torch.arange(batch, device=clips.device).repeat_interleave(channels)

        outputs, (loc, scale), *_ = self.encoder.encode(context=rows, group_ids=groups)
        pooled = outputs[0].mean(dim=1)  # average over patches
        per_row = torch.cat([pooled, loc.view(-1, 1), scale.view(-1, 1)], dim=-1)
        return per_row.view(batch, -1).float()

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Apply the custom regression head and return one estimate per clip."""
        return self.head(self.embed(clips)).squeeze(-1)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts for comparing training modes."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
