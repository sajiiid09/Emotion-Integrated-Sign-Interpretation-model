"""Dataset and dataloader utilities for BdSL landmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleMetadata:
    filepath: Path
    word: str
    signer_id: str
    grammar: str


class BdSLDataset(Dataset):
    """Dataset that loads normalized landmark npz files."""

    def __init__(
        self,
        manifest_path: Path,
        landmarks_dir: Path,
        split_signers: List[str],
        split: str,
        transform: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.landmarks_dir = landmarks_dir
        self.split_signers = set(split_signers)
        self.split = split
        self.transform = transform
        self.samples = self._load_manifest()
        self.label_to_idx = {word: idx for idx, word in enumerate(sorted({s.word for s in self.samples}))}
        self.grammar_to_idx = {"neutral": 0, "question": 1, "negation": 2}

    def _load_manifest(self) -> List[SampleMetadata]:
        rows: List[SampleMetadata] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                filepath, word, signer_id, session, rep, grammar, *_ = line.strip().split(",")
                in_train = signer_id in self.split_signers
                if (self.split == "train" and in_train) or (self.split != "train" and not in_train):
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
