"""Shared configuration for modality encoders and tensor extraction."""
from __future__ import annotations

import torch

from models.constants import FACE_IN_DIM, FACE_PCA_DIM, HAND_IN_DIM, POSE_IN_DIM
from models.encoders import FaceEncoder, HandEncoder, PoseEncoder

# Keep this flag aligned with the preprocessing pipeline. If face PCA was applied during
# normalization (NormalizationConfig.use_face_pca=True), this flag must also be True so the
# FaceEncoder input dimension matches the reduced features.
USE_FACE_PCA = False

MODALITY_TENSORS = {
    "hands": lambda batch: torch.cat((batch["hand_left"], batch["hand_right"]), dim=-1),
    "face": lambda batch: batch["face"],
    "pose": lambda batch: batch["pose"],
}

MODALITY_ENCODERS = {
    "hands": lambda: HandEncoder(input_dim=HAND_IN_DIM),
    "face": lambda: FaceEncoder(input_dim=FACE_PCA_DIM if USE_FACE_PCA else FACE_IN_DIM),
    "pose": lambda: PoseEncoder(input_dim=POSE_IN_DIM),
}
