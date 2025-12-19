"""Fusion networks for multimodal features."""
from __future__ import annotations

import torch
from torch import nn

from models.classifier import MultiTaskHead
from models.encoders import FaceEncoder, HandEncoder, PoseEncoder


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


class FusionModel(nn.Module):
    """Canonical fusion architecture combining hand, face, and pose encoders."""

    def __init__(self):
        super().__init__()
        self.hand_encoder = HandEncoder()
        self.face_encoder = FaceEncoder()
        self.pose_encoder = PoseEncoder()
        fusion_dim = (
            self.hand_encoder.config.model_dim
            + self.face_encoder.config.model_dim
            + self.pose_encoder.config.model_dim
        )
        self.fusion = FusionMLP(input_dim=fusion_dim)
        self.head = MultiTaskHead(128)

    def forward(self, batch):
        hand = torch.cat((batch["hand_left"], batch["hand_right"]), dim=-1)
        hand_feat = self.hand_encoder(hand.view(hand.size(0), hand.size(1), -1))
        face_feat = self.face_encoder(batch["face"].view(batch["face"].size(0), batch["face"].size(1), -1))
        pose_feat = self.pose_encoder(batch["pose"].view(batch["pose"].size(0), batch["pose"].size(1), -1))
        fused = torch.cat([hand_feat, face_feat, pose_feat], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)
