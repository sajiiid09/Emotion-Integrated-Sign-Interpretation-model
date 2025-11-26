"""Shared configuration for modality encoders and tensor extraction."""
from __future__ import annotations

import torch

from models.constants import FACE_IN_DIM, HAND_IN_DIM, POSE_IN_DIM
from models.encoders import FaceEncoder, HandEncoder, PoseEncoder

MODALITY_TENSORS = {
    "hands": lambda batch: torch.cat((batch["hand_left"], batch["hand_right"]), dim=-1),
    "face": lambda batch: batch["face"],
    "pose": lambda batch: batch["pose"],
}

MODALITY_ENCODERS = {
    "hands": lambda: HandEncoder(input_dim=HAND_IN_DIM),
    "face": lambda: FaceEncoder(input_dim=FACE_IN_DIM),
    "pose": lambda: PoseEncoder(input_dim=POSE_IN_DIM),
}
