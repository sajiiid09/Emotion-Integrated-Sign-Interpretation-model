"""Shared configuration for modality encoders and tensor extraction."""
from __future__ import annotations

import torch

from models.encoders import FaceEncoder, HandEncoder, PoseEncoder

MODALITY_TENSORS = {
    "hands": lambda batch: torch.cat((batch["hand_left"], batch["hand_right"]), dim=-1),
    "face": lambda batch: batch["face"],
    "pose": lambda batch: batch["pose"],
}

MODALITY_ENCODERS = {
    "hands": lambda: HandEncoder(input_dim=21 * 3 * 2),
    "face": lambda: FaceEncoder(input_dim=468 * 3),
    "pose": lambda: PoseEncoder(input_dim=33 * 3),
}
