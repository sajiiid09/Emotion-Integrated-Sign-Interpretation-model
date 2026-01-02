"""Phrase segmentation for Bangla live token streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

if __package__:
    from .dataset_loader import normalize_token_bn
    from .lexicon import INTERACTION
else:
    import importlib.util
    import sys
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent

    dataset_module_name = "brain.lang.dataset_loader"
    dataset_spec = importlib.util.spec_from_file_location(
        dataset_module_name, module_dir / "dataset_loader.py"
    )
    dataset_loader = importlib.util.module_from_spec(dataset_spec)
    assert dataset_spec and dataset_spec.loader
    sys.modules[dataset_module_name] = dataset_loader
    dataset_spec.loader.exec_module(dataset_loader)  # type: ignore[misc]
    normalize_token_bn = dataset_loader.normalize_token_bn

    lexicon_module_name = "brain.lang.lexicon"
    lexicon_spec = importlib.util.spec_from_file_location(
        lexicon_module_name, module_dir / "lexicon.py"
    )
    lexicon = importlib.util.module_from_spec(lexicon_spec)
    assert lexicon_spec and lexicon_spec.loader
    sys.modules[lexicon_module_name] = lexicon
    lexicon_spec.loader.exec_module(lexicon)  # type: ignore[misc]
    INTERACTION = lexicon.INTERACTION


@dataclass
class Phrase:
    tokens: List[str]
    detected_emotion: str
    start_ts_ms: int
    end_ts_ms: int
    reason: str


@dataclass
class SegmenterConfig:
    phrase_pause_ms: int = 1000
    max_tokens: int = 12
    immediate_question_finalize: bool = False


class PhraseSegmenter:
    """Incrementally segment a stream of Bangla tokens into phrases."""

    def __init__(self, config: Optional[SegmenterConfig] = None) -> None:
        self.config = config or SegmenterConfig()
        self.current_tokens: List[str] = []
        self.current_emotion: str = ""
        self.first_token_ts: Optional[int] = None
        self.last_token_ts: Optional[int] = None
        self._question_pending: bool = False

    def add_token(self, token: str, detected_emotion: str, ts_ms: int) -> List[Phrase]:
        normalized = normalize_token_bn(token)
        if not normalized:
            return []

        phrases: List[Phrase] = []

        if (
            self.last_token_ts is not None
            and self.current_tokens
            and ts_ms - self.last_token_ts >= self.config.phrase_pause_ms
        ):
            finalized = self._finalize("pause", self.last_token_ts)
            if finalized:
                phrases.append(finalized)

        if not self.current_tokens:
            self.first_token_ts = ts_ms

        self.current_tokens.append(normalized)
        self.current_emotion = detected_emotion
        self.last_token_ts = ts_ms

        if detected_emotion == "question" and self.current_tokens:
            self._question_pending = True
            if self.config.immediate_question_finalize:
                finalized = self._finalize("question_boundary", ts_ms)
                if finalized:
                    phrases.append(finalized)
                return phrases

        if normalized in INTERACTION and normalized in {"বিদায়", "হ্যালো"}:
            finalized = self._finalize("interaction_boundary", ts_ms)
            if finalized:
                phrases.append(finalized)
            return phrases

        if len(self.current_tokens) >= self.config.max_tokens:
            finalized = self._finalize("max_len", ts_ms)
            if finalized:
                phrases.append(finalized)

        return phrases

    def flush(self, ts_ms: int) -> List[Phrase]:
        if not self.current_tokens:
            return []
        finalized = self._finalize("pause", ts_ms)
        return [finalized] if finalized else []

    def _finalize(self, reason: str, end_ts_ms: int) -> Optional[Phrase]:
        if not self.current_tokens:
            return None

        final_reason = "question_boundary" if reason == "pause" and self._question_pending else reason
        phrase = Phrase(
            tokens=list(self.current_tokens),
            detected_emotion=self.current_emotion,
            start_ts_ms=self.first_token_ts if self.first_token_ts is not None else end_ts_ms,
            end_ts_ms=end_ts_ms,
            reason=final_reason,
        )

        self.current_tokens = []
        self.current_emotion = ""
        self.first_token_ts = None
        self.last_token_ts = None
        self._question_pending = False

        return phrase


__all__ = ["Phrase", "SegmenterConfig", "PhraseSegmenter"]


if __name__ == "__main__":
    segmenter = PhraseSegmenter()
    events = [
        ("হ্যালো", "neutral", 0),
        ("আমি", "neutral", 200),
        ("বিজ্ঞান", "neutral", 400),
        ("কি", "question", 600),
        ("অর্থ", "question", 1800),
        ("বিদায়", "neutral", 2200),
    ]

    for token, emotion, ts in events:
        phrases = segmenter.add_token(token, emotion, ts)
        for phrase in phrases:
            print(phrase)

    for phrase in segmenter.flush(3000):
        print(phrase)
