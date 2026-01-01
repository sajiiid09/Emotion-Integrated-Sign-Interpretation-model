"""Core Brain service logic (Phase 6).

Provides deterministic placeholder responses with robust normalization,
intent parsing, contradiction resolution, prompt construction, and
optional Gemini-backed generation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import cast

from .cache import ResponseCache
from .config import BrainConfig, load_config
from .constants import (
    ALLOWED_TAGS,
    DEFAULT_BN,
    DEDUPE_WINDOW,
    FALLBACK_BN,
    MAX_KEYWORDS,
    PUNCT_STRIP_CHARS,
    UNKNOWN_TOKEN_PATTERNS,
)
from .intent import Intent, ResolvedIntent, intent_to_debug, resolved_intent_to_debug
from .gemini_client import GeminiClient
from .postprocess import postprocess_response_bn
from .prompt_builder import build_prompt
from .rules import resolve_emotion
from .types import BrainInput, BrainOutput, BrainStatus, EmotionTag


_CACHE: ResponseCache | None = None
_CACHE_CFG: tuple[float, float, bool] | None = None


# Phase 2: Continuation classifiers
def is_affirmative_bn(tokens: list[str]) -> bool:
    """Check if tokens indicate affirmative intent (yes)."""
    affirmative = {"হ্যাঁ", "জি", "আচ্ছা", "ঠিক", "হবে", "চাই"}
    for token in tokens:
        if token in affirmative:
            return True
    return False


def is_negative_bn(tokens: list[str]) -> bool:
    """Check if tokens indicate negative intent (no)."""
    negative = {"না", "চাইনা", "দরকার নেই", "থাক", "বন্ধ"}
    for token in tokens:
        if token in negative:
            return True
    return False


@dataclass(frozen=True)
class _NormalizationResult:
    cleaned: list[str]
    removed_unknowns: list[str]
    removed_tags: list[str]
    deduped: bool
    truncated: bool
    truncated_count: int


def clean_token(tok: str) -> str:
    """Trim whitespace and surrounding punctuation, collapsing internal spaces."""

    stripped = tok.strip().strip(PUNCT_STRIP_CHARS)
    collapsed = " ".join(stripped.split())
    return collapsed


def is_unknown_token(tok: str) -> bool:
    """Return True if the token matches common unknown patterns or is empty."""

    return not tok or tok.lower() in UNKNOWN_TOKEN_PATTERNS


def split_keywords_text(text: str) -> list[str]:
    """Split a whitespace-separated keyword string for CLI usage."""

    return text.split()


def _normalize_keywords_detailed(keywords: list[str]) -> _NormalizationResult:
    """Normalize keyword tokens with Phase 2 rules.

    Steps:
    - Clean punctuation/whitespace.
    - Drop unknown tokens.
    - Drop tokens that are emotion tags.
    - Collapse duplicates within a small jitter window.
    - Truncate to MAX_KEYWORDS, keeping the most recent items.
    """

    cleaned_tokens: list[str] = []
    removed_unknowns: list[str] = []
    removed_tags: list[str] = []

    for token in keywords:
        cleaned = clean_token(token)
        if is_unknown_token(cleaned):
            removed_unknowns.append(token)
            continue
        if cleaned in ALLOWED_TAGS:
            removed_tags.append(cleaned)
            continue
        if cleaned:
            cleaned_tokens.append(cleaned)

    deduped_tokens: list[str] = []
    deduped = False
    for token in cleaned_tokens:
        window = deduped_tokens[-(DEDUPE_WINDOW - 1) :] if DEDUPE_WINDOW > 1 else []
        if token in window:
            deduped = True
            continue
        deduped_tokens.append(token)

    truncated = False
    truncated_count = 0
    if len(deduped_tokens) > MAX_KEYWORDS:
        truncated_count = len(deduped_tokens) - MAX_KEYWORDS
        deduped_tokens = deduped_tokens[-MAX_KEYWORDS:]
        truncated = True

    return _NormalizationResult(
        cleaned=deduped_tokens,
        removed_unknowns=removed_unknowns,
        removed_tags=removed_tags,
        deduped=deduped,
        truncated=truncated,
        truncated_count=truncated_count,
    )


def normalize_keywords(keywords: list[str]) -> list[str]:
    """Backward-compatible normalization wrapper returning only cleaned tokens."""

    return _normalize_keywords_detailed(keywords).cleaned


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


def parse_intent_from_input(brain_input: BrainInput) -> tuple[Intent, dict[str, object], dict[str, int]]:
    """Parse an :class:`Intent` from a :class:`BrainInput`.

    Returns the intent, normalization debug, and token statistics.
    """

    norm_result = _normalize_keywords_detailed(brain_input.keywords)
    flags = {
        "had_unknowns": bool(norm_result.removed_unknowns),
        "had_duplicates": norm_result.deduped,
        "truncated": norm_result.truncated,
        "tag_in_keywords": bool(norm_result.removed_tags),
    }

    notes: list[str] = []
    if norm_result.removed_unknowns:
        notes.append("dropped_unknown_tokens")
    if norm_result.deduped:
        notes.append("collapsed_duplicates")
    if norm_result.truncated:
        notes.append("truncated_keywords")
    if norm_result.removed_tags:
        notes.append("removed_emotion_tags_from_keywords")

    intent = Intent(
        keywords=norm_result.cleaned,
        raw_keywords=list(brain_input.keywords),
        detected_emotion=brain_input.emotion,
        meta=brain_input.meta,
        flags=flags,
        notes=notes,
    )

    normalization_debug = {
        "removed_unknowns": norm_result.removed_unknowns,
        "removed_tags": norm_result.removed_tags,
        "deduped": norm_result.deduped,
        "truncated": norm_result.truncated,
        "truncated_count": norm_result.truncated_count,
    }

    token_stats = {
        "input_count": len(brain_input.keywords),
        "cleaned_count": len(norm_result.cleaned),
        "dropped_unknowns": len(norm_result.removed_unknowns),
        "dropped_tags": len(norm_result.removed_tags),
        "truncated_count": norm_result.truncated_count,
    }

    return intent, normalization_debug, token_stats


def parse_intent_from_tokens(tokens: list[str]) -> tuple[Intent, dict[str, object], dict[str, int]]:
    """Parse an intent from a positional token list."""

    if not tokens:
        brain_input = BrainInput(keywords=[], emotion="neutral")
        return parse_intent_from_input(brain_input)

    last_cleaned = clean_token(tokens[-1]) if tokens else ""
    if last_cleaned in ALLOWED_TAGS:
        emotion = validate_emotion(last_cleaned)
        keywords = tokens[:-1]
    else:
        emotion = "neutral"
        keywords = tokens

    brain_input = BrainInput(keywords=keywords, emotion=emotion)
    return parse_intent_from_input(brain_input)


def _generate_response_text(keywords: list[str], emotion: EmotionTag) -> str:
    if not keywords:
        return DEFAULT_BN
    if emotion == "question":
        return "আপনি " + " ".join(keywords) + " সম্পর্কে জানতে চাচ্ছেন। সংক্ষেপে বলি।"
    if emotion == "happy":
        return "দারুণ! চলুন " + " ".join(keywords) + " নিয়ে শিখি!"
    if emotion == "sad":
        return "চিন্তা করবেন না। " + " ".join(keywords) + " বিষয়টা ধীরে ধীরে শিখে ফেলবেন।"
    if emotion == "negation":
        return "ঠিক আছে, এটা নয়। আপনি কোনটা বোঝাতে চাচ্ছেন?"
    return "আপনি বললেন: " + " ".join(keywords) + "। আমি সাহায্য করছি।"


def _respond_with_intent(
    intent: Intent,
    normalization_debug: dict[str, object] | None,
    token_stats: dict[str, int] | None,
    cfg: BrainConfig | None = None,
) -> BrainOutput:
    config = cfg or load_config()
    start = time.perf_counter()
    resolved: ResolvedIntent | None = None
    response = FALLBACK_BN
    emotion: EmotionTag = intent.detected_emotion
    status: BrainStatus = "error"
    error: str | None = None
    prompt = None
    prompt_hash: str | None = None
    gemini_meta: dict[str, object] = {"enabled": config.use_gemini}
    cache_hit = False

    try:
        resolved = resolve_emotion(intent)
        emotion = resolved.resolved_emotion
        prompt = build_prompt(resolved, cfg=config)
        if config.use_gemini:
            cache_key = None
            global _CACHE, _CACHE_CFG
            if config.cache_enabled:
                desired_cfg = (config.cache_ttl_s, float(config.max_response_words), config.cache_enabled)
                if _CACHE is None or _CACHE_CFG != desired_cfg:
                    _CACHE = ResponseCache(ttl_s=config.cache_ttl_s)
                    _CACHE_CFG = desired_cfg
                cache_key = hashlib.sha1(prompt.as_text.encode("utf-8")).hexdigest()
                cached = _CACHE.get(cache_key) if _CACHE else None
                if cached:
                    response = cached
                    status = "ready"
                    cache_hit = True
            if not cache_hit:
                client = GeminiClient(config)
                response, gemini_meta = client.generate(prompt)
                if gemini_meta.get("error"):
                    status = "error"
                    error = str(gemini_meta.get("error"))
                else:
                    status = "ready"
                if cache_key and _CACHE and not gemini_meta.get("error"):
                    _CACHE.set(cache_key, response)
        else:
            response = _generate_response_text(resolved.keywords, emotion)
            status = "ready"
    except Exception as exc:  # pragma: no cover - Phase 1 has no tests
        response = FALLBACK_BN
        emotion = "neutral"
        status = "error"
        error = str(exc)
    end = time.perf_counter()

    latency_ms = int((end - start) * 1000)

    prompt_debug: dict[str, object | None] = {}
    if prompt:
        prompt_hash = hashlib.sha1(prompt.as_text.encode("utf-8")).hexdigest()
        prompt_debug["preview"] = prompt.as_text[:200]
        prompt_debug["user_preview"] = prompt.user
        prompt_debug["mode"] = prompt.mode
        prompt_debug["metadata"] = prompt.debug
        if config.debug:
            prompt_debug.update(
                {
                    "system": prompt.system,
                    "user": prompt.user,
                    "as_text": prompt.as_text,
                    "hash": prompt_hash,
                }
            )
        else:
            prompt_debug["hash"] = prompt_hash

    debug = {
        "raw_keywords": intent.raw_keywords,
        "normalized_keywords": intent.keywords,
        "input_emotion": intent.detected_emotion,
        "intent": intent_to_debug(intent),
        "resolved_intent": resolved_intent_to_debug(resolved) if resolved else None,
        "rule_trace": resolved.rule_trace if resolved else None,
        "normalization": normalization_debug,
        "token_stats": token_stats,
        "prompt": prompt_debug if prompt_debug else None,
        "gemini": gemini_meta,
        "cache_hit": cache_hit,
    }

    # Use mode-aware postprocessing
    mode = prompt.mode if prompt else None
    final_response = postprocess_response_bn(response, config, mode=mode)

    return BrainOutput(
        response_bn=final_response,
        resolved_emotion=emotion,
        status=status,
        error=error,
        latency_ms=latency_ms,
        debug=debug,
    )


def respond(brain_input: BrainInput, cfg: BrainConfig | None = None) -> BrainOutput:
    """Generate a deterministic stub response based on parsed intent."""

    return build_output(brain_input, cfg=cfg)


def build_output(brain_input: BrainInput, cfg: BrainConfig | None = None) -> BrainOutput:
    """Full pipeline hook used by both synchronous and async paths."""

    intent, normalization_debug, token_stats = parse_intent_from_input(brain_input)
    return _respond_with_intent(intent, normalization_debug, token_stats, cfg=cfg)


def respond_from_list(tokens: list[str], cfg: BrainConfig | None = None) -> BrainOutput:
    """Helper to respond from a positional token list."""

    intent, normalization_debug, token_stats = parse_intent_from_tokens(tokens)
    return _respond_with_intent(intent, normalization_debug, token_stats, cfg=cfg)
