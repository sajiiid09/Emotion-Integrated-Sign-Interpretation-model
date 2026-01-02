"""Rule-based shaper converting phrases into proto Bangla sentences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

if __package__:
    from .dataset_loader import normalize_token_bn
    from .lexicon import (
        EMOTION_ADJ,
        INTERACTION,
        PRONOUNS,
        TIME_WORDS,
        TUTOR_TOPICS,
        VERBS,
        WH_WORDS,
        get_category,
    )
    from .segmenter import Phrase
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
    EMOTION_ADJ = lexicon.EMOTION_ADJ
    INTERACTION = lexicon.INTERACTION
    PRONOUNS = lexicon.PRONOUNS
    TIME_WORDS = lexicon.TIME_WORDS
    TUTOR_TOPICS = lexicon.TUTOR_TOPICS
    VERBS = lexicon.VERBS
    WH_WORDS = lexicon.WH_WORDS
    get_category = lexicon.get_category

    segmenter_module_name = "brain.lang.segmenter"
    segmenter_spec = importlib.util.spec_from_file_location(
        segmenter_module_name, module_dir / "segmenter.py"
    )
    segmenter = importlib.util.module_from_spec(segmenter_spec)
    assert segmenter_spec and segmenter_spec.loader
    sys.modules[segmenter_module_name] = segmenter
    segmenter_spec.loader.exec_module(segmenter)  # type: ignore[misc]
    Phrase = segmenter.Phrase


@dataclass
class ShapedSentence:
    canonical_tokens: List[str]
    proto_bn: str
    intent_type: str
    confidence: float
    flags: Dict[str, bool]
    needs_gemini: bool


def _apply_punctuation(proto: str, is_question: bool) -> str:
    if not proto:
        return proto
    if proto[-1] in {"?", "।", "!"}:
        return proto
    return f"{proto}?" if is_question else f"{proto}।"


def _pick_first(tokens: List[str], candidates: set[str]) -> Optional[str]:
    for token in tokens:
        if token in candidates:
            return token
    return None


def _categorize_tokens(tokens: List[str]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "pronoun": [],
        "verb": [],
        "object": [],
        "place": [],
        "time": [],
        "wh": [],
        "emotion_adj": [],
    }
    for token in tokens:
        category = get_category(token)
        if category == "pronoun" or token in PRONOUNS:
            categories["pronoun"].append(token)
        if category == "verb" or token in VERBS:
            categories["verb"].append(token)
        if category in {"object", "subject", "concept", "noun", "person"}:
            categories["object"].append(token)
        if category == "place":
            categories["place"].append(token)
        if category == "time" or token in TIME_WORDS:
            categories["time"].append(token)
        if category == "wh" or token in WH_WORDS:
            categories["wh"].append(token)
        if category == "emotion_adj" or token in EMOTION_ADJ:
            categories["emotion_adj"].append(token)
    return categories


def shape_phrase(phrase: Phrase) -> ShapedSentence:
    normalized_tokens = [normalize_token_bn(t) for t in phrase.tokens]
    tokens = [t for t in normalized_tokens if t]
    tokens_set = set(tokens)

    flags: Dict[str, bool] = {}
    intent_type = "statement"
    confidence = 0.4
    needs_gemini = False
    proto_parts: List[str] = []

    if not tokens:
        return ShapedSentence([], "", "unclear", 0.4, {"empty": True}, True)

    # Interaction singletons
    if "বিদায়" in tokens_set:
        proto = "বিদায়। আবার দেখা হবে।"
        return ShapedSentence(tokens, proto, "interaction", 0.9, flags, False)
    if "হ্যালো" in tokens_set:
        proto = "হ্যালো। আপনি কী শিখতে চান?"
        return ShapedSentence(tokens, proto, "interaction", 0.9, flags, False)
    if "ধন্যবাদ" in tokens_set:
        proto = "ধন্যবাদ। আর কী জানতে চান?"
        return ShapedSentence(tokens, proto, "interaction", 0.9, flags, False)
    if tokens == ["হ্যাঁ"]:
        proto = "ঠিক আছে। কোন দিকটা জানতে চান?"
        return ShapedSentence(tokens, proto, "interaction", 0.9, flags, False)

    categories = _categorize_tokens(tokens)

    # Tutor topic question
    topic_token = _pick_first(tokens, TUTOR_TOPICS)
    if topic_token and phrase.detected_emotion == "question":
        proto = f"{topic_token} কী?"
        return ShapedSentence([topic_token], proto, "question", 0.9, flags, True)

    # WH questions
    wh_token = _pick_first(tokens, WH_WORDS)
    if wh_token:
        intent_type = "question"
        needs_gemini = True
        confidence = 0.7

        if wh_token == "কি":
            topic = _pick_first(tokens, set(categories["object"]) | TUTOR_TOPICS)
            if topic:
                proto = _apply_punctuation(f"{topic} কী", True)
                return ShapedSentence([topic, "কি"], proto, intent_type, confidence, flags, needs_gemini)
            proto = _apply_punctuation("কি", True)
            return ShapedSentence(["কি"], proto, intent_type, confidence, flags, needs_gemini)

        if wh_token == "কোথায়":
            place = _pick_first(tokens, set(categories["place"]))
            if place:
                proto = _apply_punctuation(f"{place} কোথায়", True)
                return ShapedSentence([place, "কোথায়"], proto, intent_type, confidence, flags, needs_gemini)
            proto = _apply_punctuation("কোথায়", True)
            return ShapedSentence(["কোথায়"], proto, intent_type, confidence, flags, needs_gemini)

        if wh_token in {"কবে", "কালকে"}:
            time_token = _pick_first(tokens, set(categories["time"])) or wh_token
            proto = _apply_punctuation(time_token, True)
            return ShapedSentence([time_token], proto, intent_type, confidence, flags, needs_gemini)

    # SOV shaping
    pronoun = _pick_first(tokens, set(categories["pronoun"]))
    verb = _pick_first(tokens, set(categories["verb"]))
    objects: List[str] = categories["object"]
    time_tokens = categories["time"]
    place_tokens = categories["place"]

    if pronoun or verb or objects:
        canonical_tokens: List[str] = []
        if pronoun:
            canonical_tokens.append(pronoun)
        canonical_tokens.extend(objects)
        if verb:
            canonical_tokens.append(verb)
        canonical_tokens.extend(time_tokens)
        canonical_tokens.extend(place_tokens)

        proto_parts.extend(canonical_tokens)
        intent_type = "question" if phrase.detected_emotion == "question" else "statement"
        confidence = 0.6
        flags["missing_subject"] = not bool(pronoun) and bool(verb)
        flags["missing_object"] = not bool(objects) and bool(verb)
        needs_gemini = intent_type == "question"

    if not proto_parts:
        proto_parts = tokens
        intent_type = "unclear"
        needs_gemini = True

    proto = " ".join(proto_parts)

    # Negation handling
    if phrase.detected_emotion == "negation":
        if "খুশি" in tokens_set or "খুশি" in proto:
            if pronoun:
                proto = f"{pronoun} খুশি নই"
            else:
                proto = "খুশি নই"
        elif verb:
            proto = f"{proto} না".strip()
        else:
            proto = f"{proto} না".strip()
        intent_type = "statement"
        confidence = max(confidence, 0.6)

    proto = _apply_punctuation(proto, intent_type == "question")

    return ShapedSentence(proto_parts, proto, intent_type, confidence, flags, needs_gemini)


__all__ = ["ShapedSentence", "shape_phrase"]


if __name__ == "__main__":
    from pprint import pprint

    sample_phrases = [
        Phrase(["হ্যালো"], "neutral", 0, 0, "interaction"),
        Phrase(["বিজ্ঞান", "কি"], "question", 0, 0, "question_boundary"),
        Phrase(["আমি", "খুশি"], "negation", 0, 0, "pause"),
        Phrase(["গণিত"], "question", 0, 0, "pause"),
        Phrase(["আমি", "বিজ্ঞান", "শিখছি"], "neutral", 0, 0, "pause"),
    ]

    for phrase in sample_phrases:
        shaped = shape_phrase(phrase)
        pprint(shaped)
