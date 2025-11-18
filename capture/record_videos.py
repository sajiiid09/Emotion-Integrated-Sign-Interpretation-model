"""Video capture utility for BdSL dataset collection."""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
LOGGER = logging.getLogger("capture")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Record BdSL dataset videos.")
    parser.add_argument("output", type=Path, help="Directory to save videos.")
    parser.add_argument("word", type=str, help="Target BdSL word.")
    parser.add_argument("signer_id", type=str, help="Signer identifier (e.g., S01).")
    parser.add_argument("session_id", type=int, help="Session id (1-based).")
    parser.add_argument("repetition", type=int, help="Repetition id (1-based).")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (OpenCV VideoCapture)."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Clip duration in seconds. Recording stops automatically after this duration.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional CSV file where capture metadata will be appended.",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        choices=["neutral", "question", "negation"],
        default="neutral",
        help="Grammar label captured in this clip.",
    )
    return parser.parse_args()


def build_filename(word: str, signer_id: str, session_id: int, repetition: int) -> str:
    """Create standardized filename."""
    session_str = f"sess{session_id:02d}"
    rep_str = f"rep{repetition:02d}"
    return f"{word}__{signer_id}__{session_str}__{rep_str}.mp4"


def get_video_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create OpenCV writer for MP4 output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def record_clip(args: argparse.Namespace) -> None:
    """Capture a single clip from webcam."""
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = build_filename(args.word, args.signer_id, args.session_id, args.repetition)
    video_path = output_dir / filename

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = get_video_writer(video_path, fps, (width, height))
    LOGGER.info("Recording to %s at %sx%s@%.1f", video_path, width, height, fps)

    frame_count = 0
    max_frames = int(args.duration * fps)
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            LOGGER.warning("Failed to read frame %d", frame_count)
            break
        writer.write(frame)
        cv2.imshow("BdSL Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            LOGGER.info("Recording interrupted by user.")
            break
        frame_count += 1

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    LOGGER.info("Saved %d frames to %s", frame_count, video_path)

    if args.metadata:
        append_metadata(
            args.metadata,
            video_path,
            args.word,
            args.signer_id,
            args.session_id,
            args.repetition,
            args.grammar,
            fps,
            width,
            height,
        )


def append_metadata(
    csv_path: Path,
    filepath: Path,
    word: str,
    signer_id: str,
    session_id: int,
    repetition: int,
    grammar: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Append a metadata row describing the recorded sample."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                [
                    "filepath",
                    "word",
                    "signer_id",
                    "session",
                    "rep",
                    "grammar_label",
                    "fps",
                    "width",
                    "height",
                    "timestamp",
                ]
            )
        writer.writerow(
            [
                filepath.as_posix(),
                word,
                signer_id,
                session_id,
                repetition,
                grammar,
                f"{fps:.1f}",
                width,
                height,
                datetime.utcnow().isoformat(),
            ]
        )
    LOGGER.info("Metadata appended to %s", csv_path)


if __name__ == "__main__":
    arguments = parse_args()
    record_clip(arguments)
