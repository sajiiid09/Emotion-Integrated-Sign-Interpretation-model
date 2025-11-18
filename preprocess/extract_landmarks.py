"""Extract MediaPipe Holistic landmarks from recorded videos."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np

from preprocess.normalize import NormalizationConfig, normalize_sample, pad_or_crop


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("extract")


mp_holistic = mp.solutions.holistic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Holistic landmarks from videos.")
    parser.add_argument("video_dir", type=Path, help="Directory containing videos.")
    parser.add_argument("output_dir", type=Path, help="Where to store .npz landmarks.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="CSV manifest containing metadata for each sample.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=48,
        help="Number of frames per sequence after padding/cropping.",
    )
    return parser.parse_args()


def extract_video(video_path: Path, holistic: mp_holistic.Holistic, seq_len: int) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames_hand_left: List[np.ndarray] = []
    frames_hand_right: List[np.ndarray] = []
    frames_face: List[np.ndarray] = []
    frames_pose: List[np.ndarray] = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)
        frames_hand_left.append(_landmark_array(result.left_hand_landmarks, 21))
        frames_hand_right.append(_landmark_array(result.right_hand_landmarks, 21))
        frames_face.append(_landmark_array(result.face_landmarks, 468))
        frames_pose.append(_landmark_array(result.pose_landmarks, 33))

    cap.release()

    sample = {
        "hand_left": pad_or_crop(np.stack(frames_hand_left), seq_len),
        "hand_right": pad_or_crop(np.stack(frames_hand_right), seq_len),
        "face": pad_or_crop(np.stack(frames_face), seq_len),
        "pose": pad_or_crop(np.stack(frames_pose), seq_len),
    }
    return sample


def _landmark_array(landmarks, size: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((size, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = NormalizationConfig(sequence_length=args.sequence_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        for video_path in args.video_dir.glob("*.mp4"):
            LOGGER.info("Processing %s", video_path.name)
            sample = extract_video(video_path, holistic, args.sequence_length)
            normalized = normalize_sample(sample, config)
            output_path = args.output_dir / f"{video_path.stem}.npz"
            np.savez_compressed(output_path, **normalized)
            metadata = {
                "source": video_path.as_posix(),
                "sequence_length": args.sequence_length,
            }
            (output_path.with_suffix(".json")).write_text(json.dumps(metadata))
            LOGGER.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
