"""Vocabulary utilities to keep label indices consistent across splits."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Vocabulary:
    label_to_idx: Dict[str, int]
    idx_to_label: List[str]


@lru_cache(maxsize=None)
def build_vocab_from_manifest(manifest_path: Path) -> Vocabulary:
    """Build a global vocabulary from the full manifest file.

    This ensures train/val/test splits share identical label indices.
    """

    path = Path(manifest_path)
    words = set()
    with path.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            _, word, *_ = line.strip().split(",")
            words.add(word)
    idx_to_label = sorted(words)
    label_to_idx = {word: idx for idx, word in enumerate(idx_to_label)}
    return Vocabulary(label_to_idx=label_to_idx, idx_to_label=idx_to_label)

