"""Create a CSV manifest enumerating the dataset samples."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest for dataset.")
    parser.add_argument("data_dir", type=Path, help="Directory containing video files.")
    parser.add_argument("output", type=Path, help="Path to manifest CSV.")
    return parser.parse_args()


def parse_filename(filename: str) -> tuple[str, str, str, str]:
    parts = filename.split("__")
    word, signer, session, repetition = parts[:4]
    session_id = session.replace("sess", "")
    rep_id = repetition.replace("rep", "").replace(".mp4", "")
    return word, signer, session_id, rep_id


def main() -> None:
    args = parse_args()
    rows = []
    for video in args.data_dir.glob("*.mp4"):
        word, signer, session, rep = parse_filename(video.stem)
        rows.append(
            [video.as_posix(), word, signer, session, rep, "neutral", 30, 1920, 1080, "auto"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
                "notes",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
