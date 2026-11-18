"""Chronos-2 encoder with a small head that outputs health.

Chronos-2 treats every channel as its own univariate series and lets series
that share a group id attend to each other. A clip is therefore fed as two
rows (current, voltage) in one group.

Chronos-2 also rescales each row by its own mean and standard deviation, which
throws away the level of the signal (for example the mean voltage, which moves
with health). We hand those two numbers to the head so that nothing is lost.
"""

import torch
from torch import nn

CHRONOS_ID = "amazon/chronos-2"
FINETUNE_MODES = ("frozen", "lora", "full", "scratch")


def load_encoder(mode: str = "lora", lora_rank: int = 8) -> nn.Module:
    """Return the Chronos-2 network prepared for one of the fine-tuning modes.

    frozen  : pretrained weights, not trained (only the head learns)
    lora    : pretrained weights frozen, small low-rank adapters trained
    full    : pretrained weights, all trained
    scratch : same architecture, random weights, all trained (control for H1)
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
    """Map clips of shape (batch, channels, time) to health in [0, 1]."""

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
        """One feature vector per clip, shape (batch, channels * (d_model + 2))."""
        batch, channels, _ = clips.shape
        rows = clips.flatten(0, 1)  # (batch * channels, time)
        groups = torch.arange(batch, device=clips.device).repeat_interleave(channels)

        outputs, (loc, scale), *_ = self.encoder.encode(context=rows, group_ids=groups)
        pooled = outputs[0].mean(dim=1)  # average over patches
        per_row = torch.cat([pooled, loc.view(-1, 1), scale.view(-1, 1)], dim=-1)
        return per_row.view(batch, -1).float()

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(clips)).squeeze(-1)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts, for the cost comparison in H2."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
