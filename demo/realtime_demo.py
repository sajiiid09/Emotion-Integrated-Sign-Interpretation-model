"""Real-time BdSL recognition demo using webcam input."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from demo.ui_helpers import draw_landmarks, overlay_text
from models.fusion import FusionModel
from models.constants import FACE_POINTS, HAND_POINTS, POSE_POINTS
from preprocess.normalize import NormalizationConfig, normalize_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time BdSL demo.")
    parser.add_argument("checkpoint", type=Path, help="Path to trained fusion model weights.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--buffer", type=int, default=48)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    holistic = mp.solutions.holistic.Holistic()
    cap = cv2.VideoCapture(0)
    buffers = _init_buffers(args.buffer)
    config = NormalizationConfig(sequence_length=args.buffer)
    ema_sign = None
    ema_grammar = None
    alpha = 0.6

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)
        sample = {
            "hand_left": _landmark_array(result.left_hand_landmarks, HAND_POINTS),
            "hand_right": _landmark_array(result.right_hand_landmarks, HAND_POINTS),
            "face": _landmark_array(result.face_landmarks, FACE_POINTS),
            "pose": _landmark_array(result.pose_landmarks, POSE_POINTS),
        }
        _append_sample(buffers, sample)

        if _is_full(buffers):
            ordered = _stack_window(buffers)
            normalized = normalize_sample(ordered, config)
            tensor_sample = {k: torch.from_numpy(v).unsqueeze(0).to(device).float() for k, v in normalized.items()}
            with torch.no_grad():
                sign_logits, grammar_logits = model(tensor_sample)
            sign_prob = torch.softmax(sign_logits, dim=1)
            grammar_prob = torch.softmax(grammar_logits, dim=1)
            ema_sign = sign_prob if ema_sign is None else alpha * sign_prob + (1 - alpha) * ema_sign
            ema_grammar = (
                grammar_prob if ema_grammar is None else alpha * grammar_prob + (1 - alpha) * ema_grammar
            )
            sign_pred = int(torch.argmax(ema_sign))
            grammar_pred = int(torch.argmax(ema_grammar))
        else:
            sign_pred = grammar_pred = -1

        overlay = overlay_text(
            frame.copy(),
            sign=f"#{sign_pred}" if sign_pred >= 0 else "...",
            grammar=f"#{grammar_pred}" if grammar_pred >= 0 else "...",
            fps=1.0 / (time.time() - start),
        )
        cv2.imshow("BdSL Demo", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


def _landmark_array(landmarks, size: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((size, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _init_buffers(size: int) -> dict[str, dict[str, np.ndarray | int]]:
    """Preallocate circular buffers to minimize per-frame allocations."""

    return {
        "hand_left": {"data": np.zeros((size, HAND_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
        "hand_right": {"data": np.zeros((size, HAND_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
        "face": {"data": np.zeros((size, FACE_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
        "pose": {"data": np.zeros((size, POSE_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
    }


def _append_sample(buffers: dict[str, dict[str, np.ndarray | int]], sample: dict[str, np.ndarray]) -> None:
    for key, buffer in buffers.items():
        buffer["data"][buffer["write_idx"]] = sample[key]
    # All buffers share the same write index/fill; update once using one of them
    first = next(iter(buffers.values()))
    first["write_idx"] = (first["write_idx"] + 1) % first["data"].shape[0]
    first["filled"] = min(first["filled"] + 1, first["data"].shape[0])
    # Propagate updated indices to keep buffers aligned
    for buffer in buffers.values():
        buffer["write_idx"] = first["write_idx"]
        buffer["filled"] = first["filled"]


def _is_full(buffers: dict[str, dict[str, np.ndarray | int]]) -> bool:
    meta = next(iter(buffers.values()))
    return meta["filled"] == meta["data"].shape[0]


def _stack_window(buffers: dict[str, dict[str, np.ndarray | int]]) -> dict[str, np.ndarray]:
    """Return the current window in chronological order without per-frame stacking."""

    stacked = {}
    sample_meta = next(iter(buffers.values()))
    size = sample_meta["data"].shape[0]
    write_idx = sample_meta["write_idx"]
    for key, buffer in buffers.items():
        if buffer["filled"] < size:
            stacked[key] = buffer["data"][: buffer["filled"]]
            continue
        if write_idx == 0:
            stacked[key] = buffer["data"]
        else:
            stacked[key] = np.concatenate(
                (buffer["data"][write_idx:], buffer["data"][:write_idx]), axis=0
            )
    return stacked


if __name__ == "__main__":
    main()
