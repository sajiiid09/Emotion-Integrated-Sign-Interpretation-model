"""Dataset and dataloader utilities for BdSL landmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from train.vocab import Vocabulary, build_vocab_from_manifest


@dataclass
class SampleMetadata:
    filepath: Path
    word: str
    signer_id: str
    grammar: str


@dataclass(frozen=True)
class SignerSplits:
    """Explicit signer partitioning for train/val/test."""

    train: Sequence[str]
    val: Sequence[str]
    test: Sequence[str]

    def __post_init__(self) -> None:
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)
        if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
            raise ValueError("Signer splits must be disjoint across train/val/test.")

    def for_split(self, split: str) -> set[str]:
        if split == "train":
            return set(self.train)
        if split == "val":
            return set(self.val)
        if split == "test":
            return set(self.test)
        raise ValueError(f"Unknown split '{split}'")


class BdSLDataset(Dataset):
    """Dataset that loads normalized landmark npz files."""

    def __init__(
        self,
        manifest_path: Path,
        landmarks_dir: Path,
        signer_splits: SignerSplits,
        split: str,
        transform: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]] | None = None,
        vocab: Vocabulary | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.landmarks_dir = landmarks_dir
        self.signer_splits = signer_splits
        self.split = split
        self.transform = transform
        self.samples = self._load_manifest()
        self.vocab = vocab if vocab is not None else build_vocab_from_manifest(self.manifest_path)
        self.label_to_idx = self.vocab.label_to_idx
        self.grammar_to_idx = {"neutral": 0, "question": 1, "negation": 2}

    def _load_manifest(self) -> List[SampleMetadata]:
        rows: List[SampleMetadata] = []
        allowed_signers = self.signer_splits.for_split(self.split)
        with self.manifest_path.open("r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                filepath, word, signer_id, session, rep, grammar, *_ = line.strip().split(",")
                if signer_id in allowed_signers:
                    rows.append(
                        SampleMetadata(
                            filepath=Path(filepath),
                            word=word,
                            signer_id=signer_id,
                            grammar=grammar,
                        )
                    )
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.samples[idx]
        npz_path = self.landmarks_dir / (meta.filepath.stem + ".npz")
        arrays = dict(np.load(npz_path))
        if self.transform:
            arrays = self.transform(arrays)
        sample = {
            "hand_left": torch.from_numpy(arrays["hand_left"]).float(),
            "hand_right": torch.from_numpy(arrays["hand_right"]).float(),
            "face": torch.from_numpy(arrays["face"]).float(),
            "pose": torch.from_numpy(arrays["pose"]).float(),
            "sign_label": torch.tensor(self.label_to_idx[meta.word], dtype=torch.long),
            "grammar_label": torch.tensor(self.grammar_to_idx[meta.grammar], dtype=torch.long),
        }
        return sample
