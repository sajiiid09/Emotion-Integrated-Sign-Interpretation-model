"""Helper functions for demo visualizations."""
from __future__ import annotations

import cv2
import numpy as np


COLORS = {
    "hands": (0, 255, 0),
    "pose": (255, 0, 0),
    "face": (0, 165, 255),
}


def draw_landmarks(frame: np.ndarray, landmarks: dict) -> np.ndarray:
    output = frame.copy()
    for key, color in COLORS.items():
        pts = landmarks.get(key)
        if pts is None:
            continue
        for x, y in pts:
            cv2.circle(output, (int(x), int(y)), 2, color, -1)
    return output


def overlay_text(frame: np.ndarray, sign: str, grammar: str, fps: float) -> np.ndarray:
    cv2.putText(frame, f"Sign: {sign}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Grammar: {grammar}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame
