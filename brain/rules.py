"""Rule engine for resolving intent contradictions (Phase 3)."""

from __future__ import annotations

from typing import Iterable

from .constants import EMOTION_KEYWORDS_MAP
from .intent import Intent, ResolvedIntent


def detect_emotion_keywords(keywords: Iterable[str]) -> set[str]:
    """Return canonical emotion keyword states detected in the provided keywords."""

    detected: set[str] = set()
    for token in keywords:
        canonical = EMOTION_KEYWORDS_MAP.get(token)
        if canonical:
            detected.add(canonical)
    return detected


def _trace_entry(rule: str, reason: str, before_emotion: str, after_emotion: str, keywords: list[str]) -> dict[str, object]:
    return {
        "rule": rule,
        "reason": reason,
        "before": {"resolved_emotion": before_emotion, "keywords": keywords},
        "after": {"resolved_emotion": after_emotion, "keywords": keywords},
    }


def resolve_emotion(intent: Intent) -> ResolvedIntent:
    """Resolve a parsed :class:`Intent` into a :class:`ResolvedIntent` using rules."""

    resolved_emotion = intent.detected_emotion
    flags = dict(intent.flags)
    notes = list(intent.notes)
    rule_trace: list[dict[str, object]] = []
    emotion_keywords = detect_emotion_keywords(intent.keywords)

    # Rule 1: Negation overrides happy cues
    if intent.detected_emotion == "negation" and "happy_word" in emotion_keywords:
        rule_trace.append(
            _trace_entry(
                "negation_overrides_happy",
                "face_tag=negation with happy keyword",
                resolved_emotion,
                "negation",
                intent.keywords,
            )
        )
        resolved_emotion = "negation"

    # Rule 2: Negation cancels negative keyword emotions
    negative_keywords = {"sad_word", "angry_word", "bad_word", "sick_word"}
    if intent.detected_emotion == "negation" and emotion_keywords.intersection(negative_keywords):
        before = resolved_emotion
        resolved_emotion = "neutral"
        flags["negated_state"] = True
        notes.append("negation_canceled_negative_state")
        rule_trace.append(
            _trace_entry(
                "negation_cancels_negative_state",
                "negation with negative emotion keyword",
                before,
                resolved_emotion,
                intent.keywords,
            )
        )

    # Rule 3: Question priority
    if intent.detected_emotion == "question":
        before = resolved_emotion
        resolved_emotion = "question"
        rule_trace.append(
            _trace_entry(
                "question_priority",
                "face_tag=question holds priority",
                before,
                resolved_emotion,
                intent.keywords,
            )
        )

    # Rule 4: Keyword-only emotion fallback when neutral face tag
    if intent.detected_emotion == "neutral":
        if "happy_word" in emotion_keywords:
            before = resolved_emotion
            resolved_emotion = "happy"
            notes.append("keyword_emotion_inference_happy")
            rule_trace.append(
                _trace_entry(
                    "keyword_emotion_inference",
                    "neutral tag with happy keyword",
                    before,
                    resolved_emotion,
                    intent.keywords,
                )
            )
        elif emotion_keywords.intersection({"sad_word", "bad_word", "sick_word"}):
            before = resolved_emotion
            resolved_emotion = "sad"
            notes.append("keyword_emotion_inference_sad")
            rule_trace.append(
                _trace_entry(
                    "keyword_emotion_inference",
                    "neutral tag with negative keyword",
                    before,
                    resolved_emotion,
                    intent.keywords,
                )
            )

    return ResolvedIntent(
        keywords=intent.keywords,
        detected_emotion=intent.detected_emotion,
        resolved_emotion=resolved_emotion,
        meta=intent.meta,
        flags=flags,
        notes=notes,
        rule_trace=rule_trace,
    )

