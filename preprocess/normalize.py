"""Normalization utilities for landmark sequences."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


NECK_POSE_INDEX = 11  # MediaPipe pose landmark for left shoulder? we use neck derived from shoulders
RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11


@dataclass
class NormalizationConfig:
    """Configuration for landmark normalization."""

    face_pca_components: int = 128
    sequence_length: int = 48


class FacePCAReducer:
    """Utility that lazily fits PCA on face landmarks and applies reduction."""

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.pca: PCA | None = None

    def fit(self, face_array: np.ndarray) -> None:
        LOGGER.info("Fitting PCA on %s samples", face_array.shape)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(face_array)

    def transform(self, face_array: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA is not fitted.")
        return self.pca.transform(face_array)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"components": self.n_components, "mean": self.pca.mean_.tolist(), "components_matrix": self.pca.components_.tolist()}
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "FacePCAReducer":
        data = json.loads(path.read_text())
        reducer = cls(n_components=data["components"])
        reducer.pca = PCA(n_components=reducer.n_components)
        reducer.pca.mean_ = np.array(data["mean"])
        reducer.pca.components_ = np.array(data["components_matrix"])
        reducer.pca.n_features_in_ = reducer.pca.components_.shape[1]
        return reducer


def center_and_scale(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center landmarks around the neck (average shoulder) and scale by shoulder width."""
    left = landmarks[:, LEFT_SHOULDER, :3]
    right = landmarks[:, RIGHT_SHOULDER, :3]
    neck = (left + right) / 2.0
    centered = landmarks - neck[:, None, :]
    shoulder_width = np.linalg.norm(left - right, axis=-1)
    scale = np.maximum(shoulder_width.mean(), 1e-6)
    normalized = centered / scale
    return normalized, scale


def pad_or_crop(sequence: np.ndarray, target_length: int = 48) -> np.ndarray:
    """Crop or pad a temporal sequence along axis 0."""
    length = sequence.shape[0]
    if length == target_length:
        return sequence
    if length > target_length:
        start = (length - target_length) // 2
        return sequence[start : start + target_length]
    pad_width = target_length - length
    padding = np.zeros((pad_width, *sequence.shape[1:]), dtype=sequence.dtype)
    return np.concatenate([sequence, padding], axis=0)


def normalize_sample(sample: Dict[str, np.ndarray], config: NormalizationConfig) -> Dict[str, np.ndarray]:
    """Normalize a holistic landmark sample."""
    pose = sample["pose"]
    pose_norm, _ = center_and_scale(pose)
    hands_left = sample["hand_left"]
    hands_right = sample["hand_right"]
    face = sample["face"]

    def process(stream: np.ndarray) -> np.ndarray:
        centered = stream - pose[:, None, :3].mean(axis=1, keepdims=True)
        return pad_or_crop(centered, config.sequence_length)

    output = {
        "pose": pad_or_crop(pose_norm, config.sequence_length),
        "hand_left": process(hands_left),
        "hand_right": process(hands_right),
        "face": pad_or_crop(face, config.sequence_length),
    }
    return output
