"""Normalization utilities for landmark sequences."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from mediapipe.solutions.holistic import PoseLandmark
from sklearn.decomposition import PCA


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# MediaPipe pose landmark indices for reference points
LEFT_SHOULDER = PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER.value
NECK_POSE_INDEX = LEFT_SHOULDER  # retained for backward compatibility in comments
MIN_SCALE = 0.1


@dataclass
class NormalizationConfig:
    """Configuration for landmark normalization.

    ``use_face_pca`` should mirror the model-side flag (``models.config.USE_FACE_PCA``)
    to keep feature dimensions consistent with the FaceEncoder input.
    """

    face_pca_components: int = 128
    face_pca_path: Optional[Path] = None
    use_face_pca: bool = False
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


def _compute_neck_and_scale(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute neck reference (average shoulders) and robust scale.

    A minimum scale is enforced to avoid exploding coordinates when shoulders are missing or
    extremely close together. Existing datasets should be regenerated after this change so all
    modalities share a common origin and scale.
    """

    left = landmarks[:, LEFT_SHOULDER, :3]
    right = landmarks[:, RIGHT_SHOULDER, :3]
    neck = (left + right) / 2.0
    shoulder_width = np.linalg.norm(left - right, axis=-1)
    valid_width = np.isfinite(shoulder_width) & (shoulder_width > 0)
    if not np.any(valid_width):
        LOGGER.warning("Invalid shoulder widths detected; falling back to minimum scale %s", MIN_SCALE)
        scale = MIN_SCALE
    else:
        scale = max(float(shoulder_width[valid_width].mean()), MIN_SCALE)

    valid_neck = np.isfinite(neck).all(axis=1)
    if not np.any(valid_neck):
        LOGGER.warning("Neck landmarks invalid; defaulting to zeros for centering.")
        neck[:] = 0.0
    else:
        last_valid = np.where(valid_neck)[0][-1]
        neck[~valid_neck] = neck[last_valid]

    return neck, scale


def center_and_scale(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center landmarks around the neck (average shoulder) and scale by shoulder width."""

    neck, scale = _compute_neck_and_scale(landmarks)
    centered = landmarks - neck[:, None, :]
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


_FACE_PCA_CACHE: Dict[Path, FacePCAReducer] = {}


def _get_face_reducer(config: NormalizationConfig) -> Optional[FacePCAReducer]:
    if not config.use_face_pca:
        return None
    if config.face_pca_path is None:
        LOGGER.warning("Face PCA requested but no reducer path was provided; skipping PCA.")
        return None
    if config.face_pca_path in _FACE_PCA_CACHE:
        return _FACE_PCA_CACHE[config.face_pca_path]
    if not config.face_pca_path.exists():
        LOGGER.warning("Face PCA reducer path %s not found; using raw face landmarks.", config.face_pca_path)
        return None
    reducer = FacePCAReducer.load(config.face_pca_path)
    _FACE_PCA_CACHE[config.face_pca_path] = reducer
    return reducer


def normalize_sample(sample: Dict[str, np.ndarray], config: NormalizationConfig) -> Dict[str, np.ndarray]:
    """Normalize a holistic landmark sample.

    All modalities are centered on the neck (mid-shoulder) reference and scaled by shoulder width
    so pose, hands, and face share a common origin/scale. This behavior differs from earlier
    versions; regenerate stored .npz landmarks after updating normalization.
    """

    pose = sample["pose"]
    neck, scale = _compute_neck_and_scale(pose)

    def process_stream(stream: np.ndarray) -> np.ndarray:
        centered = stream - neck[:, None, :]
        scaled = centered / scale
        return pad_or_crop(scaled, config.sequence_length).astype(np.float32)

    pose_norm = process_stream(pose)
    hand_left_norm = process_stream(sample["hand_left"])
    hand_right_norm = process_stream(sample["hand_right"])
    face_norm = process_stream(sample["face"])

    reducer = _get_face_reducer(config)
    if reducer is not None:
        face_flat = face_norm.reshape(face_norm.shape[0], -1)
        face_norm = reducer.transform(face_flat).astype(np.float32)

    return {
        "pose": pose_norm,
        "hand_left": hand_left_norm,
        "hand_right": hand_right_norm,
        "face": face_norm,
    }
