"""Intent representation for the Brain module."""

from __future__ import annotations

from dataclasses import dataclass

from .types import EmotionTag


@dataclass(frozen=True)
class Intent:
    """Parsed intent after normalization.

    Attributes:
        keywords: Cleaned Bangla tokens only.
        raw_keywords: Tokens before cleaning/normalization.
        detected_emotion: Emotion determined from the input parsing.
        meta: Optional metadata forwarded from the input.
        flags: Parsing flags (e.g., truncated, had_unknowns, tag_in_keywords).
        notes: Short notes for debug and telemetry.
    """

    keywords: list[str]
    raw_keywords: list[str]
    detected_emotion: EmotionTag
    meta: dict[str, str] | None
    flags: dict[str, bool]
    notes: list[str]


def intent_to_debug(intent: Intent) -> dict[str, object]:
    """Return a compact debug representation of an :class:`Intent`."""

    return {
        "keywords": intent.keywords,
        "raw_keywords": intent.raw_keywords,
        "detected_emotion": intent.detected_emotion,
        "meta_keys": list(intent.meta.keys()) if intent.meta else [],
        "flags": intent.flags,
        "notes": intent.notes,
    }

