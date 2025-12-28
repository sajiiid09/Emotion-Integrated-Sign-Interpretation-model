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


def postprocess_response_bn(text: str, cfg: BrainConfig) -> str:
    """Clean and constrain Bangla responses for UI stability."""

    if not text:
        return FALLBACK_BN

    cleaned = _strip_markdown(text).strip()
    cleaned = _ENGLISH_WORD_PATTERN.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    cleaned = _truncate_sentences(cleaned)

    words = cleaned.split()
    if len(words) > cfg.max_response_words:
        cleaned = " ".join(words[: cfg.max_response_words]) + "…"

    if not cleaned:
        return FALLBACK_BN
    return cleaned


__all__ = [
    "postprocess_response_bn",
    "estimate_word_count_bn",
]
