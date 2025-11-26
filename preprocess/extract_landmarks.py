"""Extract MediaPipe Holistic landmarks from recorded videos."""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

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
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers to use.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Safety limit on frames processed per clip to avoid excessive memory use.",
    )
    return parser.parse_args()


def extract_video(video_path: Path, seq_len: int, max_frames: int = 300) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames_hand_left: List[np.ndarray] = []
    frames_hand_right: List[np.ndarray] = []
    frames_face: List[np.ndarray] = []
    frames_pose: List[np.ndarray] = []
    frame_count = 0

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
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
            frame_count += 1
            if frame_count >= max_frames:
                LOGGER.warning("Reached frame limit (%s) for %s; truncating clip", max_frames, video_path.name)
                break

    cap.release()

    if not frames_hand_left:
        raise ValueError(f"No frames decoded from {video_path}")

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


def _process_video_file(
    video_path: Path,
    seq_len: int,
    max_frames: int,
    output_dir: Path,
    config: NormalizationConfig,
) -> Optional[Path]:
    try:
        sample = extract_video(video_path, seq_len, max_frames)
        normalized = normalize_sample(sample, config)
        output_path = output_dir / f"{video_path.stem}.npz"
        np.savez_compressed(output_path, **normalized)
        metadata = {
            "source": video_path.as_posix(),
            "sequence_length": seq_len,
        }
        (output_path.with_suffix(".json")).write_text(json.dumps(metadata))
        return output_path
    except Exception:
        LOGGER.exception("Failed to process %s", video_path)
        return None


def main() -> None:
    args = parse_args()
    config = NormalizationConfig(sequence_length=args.sequence_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(args.video_dir.glob("*.mp4"))
    if args.num_workers <= 1:
        for video_path in video_paths:
            LOGGER.info("Processing %s", video_path.name)
            output_path = _process_video_file(video_path, args.sequence_length, args.max_frames, args.output_dir, config)
            if output_path:
                LOGGER.info("Saved %s", output_path)
    else:
        LOGGER.info("Processing %d videos with %d workers", len(video_paths), args.num_workers)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    _process_video_file, video_path, args.sequence_length, args.max_frames, args.output_dir, config
                )
                for video_path in video_paths
            ]
            for future in futures:
                output_path = future.result()
                if output_path:
                    LOGGER.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
