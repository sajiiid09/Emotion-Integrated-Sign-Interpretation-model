"""Fusion networks for multimodal features."""
from __future__ import annotations

from torch import nn


class FusionMLP(nn.Module):
    """Simple multilayer perceptron for feature fusion."""

    def __init__(self, input_dim: int = 768, hidden_dims: tuple[int, int] = (256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
