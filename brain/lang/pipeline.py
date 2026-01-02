from __future__ import annotations

"""Lightweight language pipeline entry point for Bangla phrases."""

from typing import Iterable, Optional

from .dataset_loader import normalize_token_bn
from .segmenter import Phrase, PhraseSegmenter, SegmenterConfig
from .shaper import ShapedSentence, shape_phrase


def _normalize_tokens(tokens: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for tok in tokens:
        cleaned = normalize_token_bn(tok)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def run_language_pipeline(
    tokens: list[str],
    detected_emotion: str,
    *,
    ts_ms: Optional[list[int]] = None,
    segmenter: Optional[PhraseSegmenter] = None,
    segmenter_config: Optional[SegmenterConfig] = None,
) -> ShapedSentence:
    """Run normalization -> segmentation -> shaping on a token stream.

    If timestamp information is provided, a :class:`PhraseSegmenter` is used to
    respect pause/interaction boundaries. Otherwise the tokens are treated as a
    single phrase.
    """

    normalized_tokens = _normalize_tokens(tokens)
    if segmenter is None:
        segmenter = PhraseSegmenter(segmenter_config)

    phrases = []
    if ts_ms and len(ts_ms) == len(tokens):
        for tok, ts in zip(tokens, ts_ms):
            phrases.extend(segmenter.add_token(tok, detected_emotion, ts))
        if ts_ms:
            phrases.extend(segmenter.flush(ts_ms[-1] + segmenter.config.phrase_pause_ms))
    elif normalized_tokens:
        phrases.append(
            Phrase(
                tokens=normalized_tokens,
                detected_emotion=detected_emotion,
                start_ts_ms=0,
                end_ts_ms=0,
                reason="manual",
            )
        )
    else:
        phrases.append(
            Phrase(
                tokens=[],
                detected_emotion=detected_emotion,
                start_ts_ms=0,
                end_ts_ms=0,
                reason="empty",
            )
        )

    # Use the most recent phrase for shaping (last finalized phrase wins)
    phrase = phrases[-1]
    return shape_phrase(phrase)


__all__ = ["run_language_pipeline", "SegmenterConfig", "PhraseSegmenter"]
