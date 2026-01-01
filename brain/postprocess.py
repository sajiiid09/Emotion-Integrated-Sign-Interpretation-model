"""Response post-processing utilities for Bangla outputs."""

from __future__ import annotations

import re

from .config import BrainConfig
from .constants import FALLBACK_BN


_MARKDOWN_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
_ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"([।?!])")


def estimate_word_count_bn(text: str) -> int:
    """Approximate word count for Bangla text."""

    return len(text.split())


def _truncate_sentences(text: str, max_terminators: int = 3) -> str:
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    if not parts:
        return text
    collected: list[str] = []
    terminators = 0
    for chunk in parts:
        collected.append(chunk)
        if chunk in {"।", "?", "!"}:
            terminators += 1
        if terminators >= max_terminators:
            break
    return "".join(collected).strip()


def _strip_markdown(text: str) -> str:
    cleaned = re.sub(_MARKDOWN_BLOCK_PATTERN, "", text)
    cleaned_lines = []
    for line in cleaned.splitlines():
        line = line.lstrip("-*# ")
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def postprocess_response_bn(text: str, cfg: BrainConfig, mode: str | None = None) -> str:
    """Clean and constrain Bangla responses for UI stability.
    
    Args:
        text: Raw response text
        cfg: Configuration with word limits
        mode: Optional mode ('tutor' or 'realtime') to apply appropriate word limit
    """

    if not text:
        return FALLBACK_BN

    cleaned = _strip_markdown(text).strip()
    cleaned = _ENGLISH_WORD_PATTERN.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    
    # Mode-aware sentence truncation
    # Tutor mode: allow up to 10 sentences (supports 6-8 target)
    # Realtime mode: limit to 3 sentences
    max_sentences = 10 if mode == "tutor" else 3
    cleaned = _truncate_sentences(cleaned, max_terminators=max_sentences)

    # Use mode-aware word limit
    max_words = cfg.tutor_max_response_words if mode == "tutor" else cfg.max_response_words
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]) + "…"

    if not cleaned:
        return FALLBACK_BN
    return cleaned


__all__ = [
    "postprocess_response_bn",
    "estimate_word_count_bn",
]
