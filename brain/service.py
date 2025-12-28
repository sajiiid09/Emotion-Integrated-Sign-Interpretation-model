"""Core Brain service stub for Phase 1.

This module provides deterministic placeholder responses while establishing a
stable, testable interface for future phases.
"""

from __future__ import annotations

import time
from typing import cast

from .config import BrainConfig, load_config
from .constants import ALLOWED_TAGS, DEFAULT_BN, FALLBACK_BN
from .types import BrainInput, BrainOutput, EmotionTag


def normalize_keywords(keywords: list[str]) -> list[str]:
    """Normalize keyword tokens with light preprocessing.

    - Strips whitespace from each token.
    - Drops empty tokens.
    - Collapses consecutive duplicates while preserving order.
    """

    normalized: list[str] = []
    previous: str | None = None
    for token in keywords:
        cleaned = token.strip()
        if not cleaned:
            continue
        if previous is not None and cleaned == previous:
            continue
        normalized.append(cleaned)
        previous = cleaned
    return normalized


def validate_emotion(tag: str) -> EmotionTag:
    """Return a valid emotion tag, defaulting to neutral."""

    if tag in ALLOWED_TAGS:
        return cast(EmotionTag, tag)
    return "neutral"


def _enforce_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return f"{truncated}…"


def respond(brain_input: BrainInput, cfg: BrainConfig | None = None) -> BrainOutput:
    """Generate a deterministic stub response based on keywords and emotion."""

    config = cfg or load_config()
    start = time.perf_counter()
    try:
        normalized = normalize_keywords(brain_input.keywords)
        emotion = brain_input.emotion

        if not normalized:
            response = DEFAULT_BN
        elif emotion == "question":
            response = "আপনি " + " ".join(normalized) + " সম্পর্কে জানতে চাচ্ছেন। সংক্ষেপে বলি।"
        elif emotion == "happy":
            response = "দারুণ! চলুন " + " ".join(normalized) + " নিয়ে শিখি!"
        elif emotion == "sad":
            response = "চিন্তা করবেন না। " + " ".join(normalized) + " বিষয়টা ধীরে ধীরে শিখে ফেলবেন।"
        elif emotion == "negation":
            response = "ঠিক আছে, এটা নয়। আপনি কোনটা বোঝাতে চাচ্ছেন?"
        else:
            response = "আপনি বললেন: " + " ".join(normalized) + "। আমি সাহায্য করছি।"

        response = _enforce_word_limit(response, config.max_response_words)
        status = "ready"
        error: str | None = None
    except Exception as exc:  # pragma: no cover - Phase 1 has no tests
        response = FALLBACK_BN
        emotion = "neutral"
        status = "error"
        error = str(exc)
        normalized = []
    end = time.perf_counter()

    latency_ms = int((end - start) * 1000)

    debug = {
        "raw_keywords": brain_input.keywords,
        "normalized_keywords": normalized,
        "input_emotion": brain_input.emotion,
    }

    return BrainOutput(
        response_bn=response,
        resolved_emotion=emotion,
        status=status,
        error=error,
        latency_ms=latency_ms,
        debug=debug,
    )


def respond_from_list(tokens: list[str], cfg: BrainConfig | None = None) -> BrainOutput:
    """Helper to respond from a positional token list.

    The final token may be an emotion tag; otherwise neutral is assumed.
    """

    if not tokens:
        brain_input = BrainInput(keywords=[], emotion="neutral")
    elif tokens[-1] in ALLOWED_TAGS:
        emotion = validate_emotion(tokens[-1])
        brain_input = BrainInput(keywords=tokens[:-1], emotion=emotion)
    else:
        brain_input = BrainInput(keywords=tokens, emotion="neutral")

    return respond(brain_input, cfg=cfg)
