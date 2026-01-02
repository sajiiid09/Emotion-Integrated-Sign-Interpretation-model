"""Phrase segmentation utilities for Bangla live token streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .dataset_loader import normalize_token_bn
from .lexicon import INTERACTION


@dataclass
class Phrase:
    """A finalized phrase from the live token stream."""

    tokens: List[str]
    detected_emotion: str
    start_ts_ms: int
    end_ts_ms: int
    reason: str


@dataclass
class SegmenterConfig:
    """Configuration controlling phrase segmentation boundaries."""

    phrase_pause_ms: int = 1000
    max_tokens: int = 12
    immediate_question_finalize: bool = False


class PhraseSegmenter:
    """Segment live tokens into phrases based on timing and content boundaries."""

    def __init__(self, config: Optional[SegmenterConfig] = None):
        self.config = config or SegmenterConfig()
        self.current_tokens: List[str] = []
        self.current_emotion: str = ""
        self.last_token_ts: Optional[int] = None
        self.start_ts_ms: Optional[int] = None

    def _finalize(self, end_ts_ms: int, reason: str) -> Phrase:
        phrase = Phrase(
            tokens=self.current_tokens.copy(),
            detected_emotion=self.current_emotion,
            start_ts_ms=self.start_ts_ms or end_ts_ms,
            end_ts_ms=end_ts_ms,
            reason=reason,
        )
        self.current_tokens = []
        self.current_emotion = ""
        self.last_token_ts = None
        self.start_ts_ms = None
        return phrase

    def add_token(self, token: str, detected_emotion: str, ts_ms: int) -> List[Phrase]:
        """Add a token with timestamp, returning any finalized phrases."""

        normalized = normalize_token_bn(token)
        if not normalized:
            return []

        finalized: List[Phrase] = []

        if (
            self.current_tokens
            and self.last_token_ts is not None
            and ts_ms - self.last_token_ts >= self.config.phrase_pause_ms
        ):
            reason = "question_boundary" if self.current_emotion == "question" else "pause"
            finalized.append(self._finalize(self.last_token_ts, reason))

        if not self.current_tokens:
            self.start_ts_ms = ts_ms

        self.current_tokens.append(normalized)
        self.current_emotion = detected_emotion or self.current_emotion
        self.last_token_ts = ts_ms

        if (
            self.config.immediate_question_finalize
            and self.current_emotion == "question"
            and len(self.current_tokens) >= 1
        ):
            finalized.append(self._finalize(ts_ms, "question_boundary"))
            return finalized

        if normalized in INTERACTION and normalized in {"বিদায়", "হ্যালো"}:
            finalized.append(self._finalize(ts_ms, "interaction_boundary"))
            return finalized

        if len(self.current_tokens) >= self.config.max_tokens:
            finalized.append(self._finalize(ts_ms, "max_len"))

        return finalized

    def flush(self, ts_ms: int) -> List[Phrase]:
        """Force-finalize any pending tokens using pause semantics."""

        if not self.current_tokens:
            return []
        reason = "question_boundary" if self.current_emotion == "question" else "pause"
        return [self._finalize(ts_ms, reason)]


if __name__ == "__main__":
    demo_tokens = [
        ("হ্যালো", "neutral", 0),
        ("আমি", "neutral", 200),
        ("খুশি", "emotion", 450),
        ("কি", "question", 1700),
        ("তুমি", "neutral", 1900),
        ("বিদায়", "neutral", 4000),
    ]

    segmenter = PhraseSegmenter()
    phrases: List[Phrase] = []
    for token, emotion, ts in demo_tokens:
        phrases.extend(segmenter.add_token(token, emotion, ts))
    phrases.extend(segmenter.flush(5000))

    for idx, phrase in enumerate(phrases, start=1):
        print(
            f"Phrase {idx}: tokens={phrase.tokens}, emotion={phrase.detected_emotion}, "
            f"start={phrase.start_ts_ms}, end={phrase.end_ts_ms}, reason={phrase.reason}"
        )
