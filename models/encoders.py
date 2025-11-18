"""Temporal encoders for each modality."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.utils import TemporalConvBlock, TemporalTransformerBlock


@dataclass
class EncoderConfig:
    input_dim: int
    model_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    use_transformer: bool = True


class BaseEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        blocks = []
        for _ in range(config.num_layers):
            if config.use_transformer:
                blocks.append(TemporalTransformerBlock(config.model_dim, dropout=config.dropout))
            else:
                blocks.append(TemporalConvBlock(config.model_dim, config.model_dim))
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            if isinstance(block, TemporalTransformerBlock):
                x = block(x)
            else:
                x = block(x.transpose(1, 2)).transpose(1, 2)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        return pooled


class HandEncoder(BaseEncoder):
    def __init__(self, input_dim: int = 21 * 3 * 2, **kwargs):
        super().__init__(EncoderConfig(input_dim=input_dim, **kwargs))


class FaceEncoder(BaseEncoder):
    def __init__(self, input_dim: int = 468 * 3, **kwargs):
        super().__init__(EncoderConfig(input_dim=input_dim, **kwargs))


class PoseEncoder(BaseEncoder):
    def __init__(self, input_dim: int = 33 * 3, **kwargs):
        super().__init__(EncoderConfig(input_dim=input_dim, **kwargs))
