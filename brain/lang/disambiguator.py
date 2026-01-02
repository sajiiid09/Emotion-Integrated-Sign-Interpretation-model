"""Ambiguity detection and clarification prompt generation for shaped phrases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

if __package__:
    from .dataset_loader import normalize_token_bn
    from .lexicon import (
        INSTRUCTION_WORDS,
        PRONOUNS,
        TUTOR_TOPICS,
        VERBS,
        get_category,
    )
else:
    import importlib.util
    import sys
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent

    dataset_module_name = "brain.lang.dataset_loader"
    dataset_spec = importlib.util.spec_from_file_location(
        dataset_module_name, module_dir / "dataset_loader.py"
    )
    dataset_loader = importlib.util.module_from_spec(dataset_spec)
    assert dataset_spec and dataset_spec.loader
    sys.modules[dataset_module_name] = dataset_loader
    dataset_spec.loader.exec_module(dataset_loader)  # type: ignore[misc]
    normalize_token_bn = dataset_loader.normalize_token_bn

    lexicon_module_name = "brain.lang.lexicon"
    lexicon_spec = importlib.util.spec_from_file_location(
        lexicon_module_name, module_dir / "lexicon.py"
    )
    lexicon = importlib.util.module_from_spec(lexicon_spec)
    assert lexicon_spec and lexicon_spec.loader
    sys.modules[lexicon_module_name] = lexicon
    lexicon_spec.loader.exec_module(lexicon)  # type: ignore[misc]
    INSTRUCTION_WORDS = lexicon.INSTRUCTION_WORDS
    PRONOUNS = lexicon.PRONOUNS
    TUTOR_TOPICS = lexicon.TUTOR_TOPICS
    VERBS = lexicon.VERBS
    get_category = lexicon.get_category


@dataclass
class DisambiguationResult:
    needs_context: bool
    clarification_bn: Optional[str]
    suggested_options: List[str]
    confidence_adjustment: float = -0.1


def _categorize(tokens: List[str]) -> dict[str, List[str]]:
    categorized: dict[str, List[str]] = {
        "pronoun": [],
        "verb": [],
        "object": [],
        "subject": [],
        "concept": [],
        "instruction": [],
    }

    for token in tokens:
        category = get_category(token)
        if category == "pronoun" or token in PRONOUNS:
            categorized["pronoun"].append(token)
        if category == "verb" or token in VERBS:
            categorized["verb"].append(token)
        if category in {"object", "noun", "person"}:
            categorized["object"].append(token)
        if category == "subject":
            categorized["subject"].append(token)
        if category == "concept":
            categorized["concept"].append(token)
        if category == "instruction" or token in INSTRUCTION_WORDS:
            categorized["instruction"].append(token)
    return categorized


def _append_options(text: str, options: List[str]) -> str:
    if not options:
        return text
    return f"{text} উদাহরণ: {', '.join(options)}"


def disambiguate(tokens: List[str]) -> DisambiguationResult:
    normalized_tokens = [normalize_token_bn(t) for t in tokens]
    clean_tokens = [t for t in normalized_tokens if t]
    tokens_set = set(clean_tokens)

    categories = _categorize(clean_tokens)
    verb = categories["verb"][0] if categories["verb"] else None
    has_topic = bool(categories["object"] or categories["subject"] or categories["concept"])

    # Rule 1: Verb without object/topic
    if verb and not has_topic and not (tokens_set & TUTOR_TOPICS):
        options: List[str] = []
        if verb in {"পড়া", "লেখা"}:
            options = ["বই", "গণিত", "বিজ্ঞান", "ভাষা"]
        clarification = _append_options(f"আপনি কী {verb} করতে চান?", options)
        return DisambiguationResult(True, clarification, options, -0.2)

    # Rule 2: Instruction word without topic
    instruction_tokens = set(categories["instruction"])
    if instruction_tokens and len(clean_tokens) == len(instruction_tokens):
        word = sorted(instruction_tokens)[0]
        options = ["গণিত", "বিজ্ঞান", "ইতিহাস", "ভূগোল", "কম্পিউটার", "বাংলাদেশ"]
        clarification = _append_options(
            f"আপনি কোন বিষয়ের {word} চান?", options
        )
        return DisambiguationResult(True, clarification, options, -0.15)

    # Rule 3: Conflicting tokens
    if "সঠিক" in tokens_set and "ভুল" in tokens_set:
        clarification = "আপনি কোনটা সঠিক আর কোনটা ভুল বোঝাতে চান?"
        return DisambiguationResult(True, clarification, [], -0.1)
    if "ভালো" in tokens_set and "খারাপ" in tokens_set:
        clarification = "আপনি ভালো বলছেন নাকি খারাপ? একটু নিশ্চিত করুন।"
        return DisambiguationResult(True, clarification, [], -0.1)

    # Rule 4: Emotion-only without subject
    emotion_only = tokens_set & {"খুশি", "দুঃখ", "রাগ", "অবাক"}
    if emotion_only and not categories["pronoun"]:
        options = ["আমি", "তুমি", "আমরা"]
        clarification = _append_options(
            "কার কথা বলছেন? আপনি, নাকি অন্য কেউ?", options
        )
        return DisambiguationResult(True, clarification, options, -0.2)

    return DisambiguationResult(False, None, [], 0.0)


__all__ = ["DisambiguationResult", "disambiguate"]


if __name__ == "__main__":
    from pprint import pprint

    demo_tokens: List[List[str]] = [
        ["খাওয়া"],
        ["ব্যাখ্যা"],
        ["সঠিক", "ভুল"],
        ["খুশি"],
        ["আমি", "খুশি"],
        ["পড়া"],
    ]

    for token_list in demo_tokens:
        result = disambiguate(token_list)
        pprint({"tokens": token_list, "result": result})
