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


@dataclass(frozen=True)
class ResolvedIntent:
    """Intent after contradiction resolution via the rule engine."""

    keywords: list[str]
    detected_emotion: EmotionTag
    resolved_emotion: EmotionTag
    meta: dict[str, str] | None
    flags: dict[str, bool]
    notes: list[str]
    rule_trace: list[dict[str, object]]


def resolved_intent_to_debug(resolved_intent: ResolvedIntent) -> dict[str, object]:
    """Return a compact debug representation of a :class:`ResolvedIntent`."""

    return {
        "keywords": resolved_intent.keywords,
        "detected_emotion": resolved_intent.detected_emotion,
        "resolved_emotion": resolved_intent.resolved_emotion,
        "meta_keys": list(resolved_intent.meta.keys()) if resolved_intent.meta else [],
        "flags": resolved_intent.flags,
        "notes": resolved_intent.notes,
        "rule_trace": resolved_intent.rule_trace,
    }

