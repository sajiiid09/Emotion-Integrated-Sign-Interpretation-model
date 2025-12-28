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
    OUTPUT_CONSTRAINTS_BN,
)
from .intent import ResolvedIntent
from .types import EmotionTag


@dataclass(frozen=True)
class BuiltPrompt:
    """Structured prompt payload."""

    system: str
    user: str
    resolved_emotion: EmotionTag
    keywords: list[str]
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


def build_user_payload(resolved: ResolvedIntent) -> str:
    """Build a concise user payload block in Bangla."""

    keywords_joined = " ".join(resolved.keywords)
    lines = [f"ইনপুট কীওয়ার্ড: {keywords_joined if keywords_joined else 'কিছু নেই'}"]
    lines.append(f"ট্যাগ: {resolved.resolved_emotion}")
    if resolved.detected_emotion != resolved.resolved_emotion:
        lines.append("মুখের অভিব্যক্তি/ব্যাকরণ অনুযায়ী টোন ঠিক করা হয়েছে।")
    lines.append("কাজ: উদ্দেশ্য বুঝে ২-৩ বাক্যে সাহায্য করুন।")
    return "\n".join(lines)


def build_prompt(resolved: ResolvedIntent, *, cfg: BrainConfig | None = None) -> BuiltPrompt:
    """Construct the full prompt for the LLM provider."""

    config = cfg or load_config()
    format_hint = "উত্তর ২-৩টি ছোট বাক্য, কোনো তালিকা নয়, কোনো শিরোনাম নয়।"
    system = f"{BASE_SYSTEM_INSTRUCTION_BN}\n{OUTPUT_CONSTRAINTS_BN}\n{HARD_OUTPUT_RULE_BN}\n{format_hint}"
    dynamic = DYNAMIC_RULES_BY_TAG.get(resolved.resolved_emotion, "")
    system_final = f"{system}\n{dynamic}" if dynamic else system
    user = build_user_payload(resolved)
    as_text = f"{system_final}\n\n{user}"

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
        "max_response_words": config.max_response_words,
    }

    return BuiltPrompt(
        system=system_final,
        user=user,
        resolved_emotion=resolved.resolved_emotion,
        keywords=list(resolved.keywords),
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
