"""Create a CSV manifest enumerating the dataset samples.

Filenames are expected to follow the schema ``<word>__<signer>__sessXX__repYY__<grammar>.mp4``
so grammar labels are preserved for the secondary task.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest for dataset.")
    parser.add_argument("data_dir", type=Path, help="Directory containing video files.")
    parser.add_argument("output", type=Path, help="Path to manifest CSV.")
    return parser.parse_args()


def parse_filename(filename: str) -> tuple[str, str, str, str, str] | None:
    """Parse filename into components with validation.

    Supports words that may contain the separator by assuming the last four tokens are
    signer, session, repetition, and grammar. Returns ``None`` when parsing fails.
    """

    stem = Path(filename).stem
    parts = stem.split("__")
    if len(parts) < 5:
        LOGGER.warning("Skipping %s: expected at least 5 parts in filename.", filename)
        return None

    word = "__".join(parts[:-4])
    signer, session, repetition, grammar = parts[-4:]
    if grammar not in {"neutral", "question", "negation", "happy", "sad"}:
        LOGGER.warning("Unrecognized grammar '%s' in %s; skipping.", grammar, filename)
        return None

    session_id = session.replace("sess", "")
    rep_id = repetition.replace("rep", "")
    return word, signer, session_id, rep_id, grammar


def main() -> None:
    args = parse_args()
    rows = []
    for video in args.data_dir.glob("*.mp4"):
        parsed = parse_filename(video.name)
        if parsed is None:
            continue
        word, signer, session, rep, grammar = parsed
        rows.append(
            [video.as_posix(), word, signer, session, rep, grammar, 30, 1920, 1080, "auto"]
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
