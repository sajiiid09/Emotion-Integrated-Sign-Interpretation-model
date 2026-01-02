"""Prompt builder for Phase 4.

Transforms a resolved intent into a structured prompt for downstream
providers (e.g., Gemini in Phase 5). The builder stays deterministic and
lightweight, exposing debug metadata for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import BrainConfig, load_config
from .constants import (
    BASE_SYSTEM_INSTRUCTION_BN,
    DYNAMIC_RULES_BY_TAG,
    HARD_OUTPUT_RULE_BN,
    MODE_REALTIME,
    MODE_TUTOR,
    MODE_RULES_BY_MODE,
    OUTPUT_CONSTRAINTS_BN,
    TUTOR_TOPIC_KEYWORDS_BN,
)
from .lang.lexicon import INSTRUCTION_WORDS, TUTOR_TOPICS
from .intent import ResolvedIntent
from .types import EmotionTag
from .lang.shaper import ShapedSentence


@dataclass(frozen=True)
class BuiltPrompt:
    """Structured prompt payload."""

    system: str
    user: str
    resolved_emotion: EmotionTag
    keywords: list[str]
    mode: str
    debug: dict[str, object]
    as_text: str


def summarize_rule_trace(trace: list[dict[str, object]], max_items: int = 3) -> list[str]:
    """Return a list of rule names that fired, capped to ``max_items``."""

    names: list[str] = []
    for entry in trace:
        rule_name = entry.get("rule")
        if isinstance(rule_name, str):
            names.append(rule_name)
        if len(names) >= max_items:
            break
    return names


def infer_mode(resolved: ResolvedIntent, shaped: ShapedSentence | None = None) -> str:
    """Infer tutor or realtime mode based on keywords, lexicon, and emotion."""

    tokens = set(resolved.keywords)
    if shaped:
        tokens.update(shaped.canonical_tokens)

    has_tutor_topic = bool(tokens & (TUTOR_TOPICS | set(TUTOR_TOPIC_KEYWORDS_BN)))
    has_instruction = bool(tokens & INSTRUCTION_WORDS)

    if has_tutor_topic and (resolved.resolved_emotion == "question" or has_instruction):
        return MODE_TUTOR

    # Fallback to previous heuristic
    if resolved.resolved_emotion == "question":
        for keyword in resolved.keywords:
            if keyword in TUTOR_TOPIC_KEYWORDS_BN:
                return MODE_TUTOR

    return MODE_REALTIME


def build_user_payload(resolved: ResolvedIntent, mode: str, shaped: ShapedSentence | None = None) -> str:
    """Build a concise user payload block in Bangla, mode-aware."""

    canonical_tokens = shaped.canonical_tokens if shaped else resolved.keywords
    keywords_joined = " ".join(canonical_tokens)
    keywords_line = keywords_joined if keywords_joined else "কিছু নেই"
    gloss = resolved.gloss_bn or keywords_joined or " "
    proto = shaped.proto_bn if shaped else gloss

    lines = [
        f"প্রাথমিক_বাক্য: {proto}",
        f"কীওয়ার্ড_ক্রম: {keywords_line}",
        f"ট্যাগ: {resolved.resolved_emotion}",
        f"সম্ভাব্য_অর্থ: {gloss}",
        f"মোড: {mode}",
        "নির্দেশনা: এই অর্থ ধরে সাহায্য করুন। টিউটর মোড হলে বিস্তারিত ব্যাখ্যা দিন।",
    ]
    return "\n".join(lines)


def build_prompt(
    resolved: ResolvedIntent, *, shaped: ShapedSentence | None = None, cfg: BrainConfig | None = None
) -> BuiltPrompt:
    """Construct the full prompt for the LLM provider with mode support.
    
    Order of system instruction blocks:
    1. Base system instruction
    2. Output constraints
    3. Hard output rule
    4. Mode-specific rule (tutor or realtime)
    5. Tag-specific rule (tone only, no length constraints)
    """

    config = cfg or load_config()
    mode = infer_mode(resolved, shaped)
    
    # Build system instruction in deterministic order
    system = f"{BASE_SYSTEM_INSTRUCTION_BN}\n{OUTPUT_CONSTRAINTS_BN}\n{HARD_OUTPUT_RULE_BN}"
    
    # Append mode-specific rule (before tag rule to take priority)
    mode_rule = MODE_RULES_BY_MODE.get(mode, "")
    if mode_rule:
        system = f"{system}\n{mode_rule}"
    
    # Append tag-specific rule (tone only, cannot override mode)
    tag_rule = DYNAMIC_RULES_BY_TAG.get(resolved.resolved_emotion, "")
    if tag_rule:
        system = f"{system}\n{tag_rule}"
    
    system_final = system
    user = build_user_payload(resolved, mode, shaped)
    as_text = f"{system_final}\n\n{user}"

    # Use mode-aware word limit in debug metadata
    max_words = config.tutor_max_response_words if mode == MODE_TUTOR else config.max_response_words
    
    debug = {
        "keyword_count": len(resolved.keywords),
        "flags": resolved.flags,
        "notes": resolved.notes,
        "rule_trace_count": len(resolved.rule_trace),
        "rule_trace_summary": summarize_rule_trace(resolved.rule_trace),
        "detected_vs_resolved": {
            "detected": resolved.detected_emotion,
            "resolved": resolved.resolved_emotion,
        },
        "mode": mode,
        "max_response_words": max_words,
        "proto_bn": shaped.proto_bn if shaped else None,
        "shaped_intent_type": shaped.intent_type if shaped else None,
    }

    return BuiltPrompt(
        system=system_final,
        user=user,
        resolved_emotion=resolved.resolved_emotion,
        keywords=list(resolved.keywords),
        mode=mode,
        debug=debug,
        as_text=as_text,
    )


def _iterable_preview(items: Iterable[str], *, max_items: int = 5) -> str:
    """Return a short preview of an iterable for debug purposes."""

    values = []
    for idx, item in enumerate(items):
        if idx >= max_items:
            values.append("…")
            break
        values.append(str(item))
    return ", ".join(values)

__all__ = [
    "BuiltPrompt",
    "build_prompt",
    "build_user_payload",
    "summarize_rule_trace",
]
