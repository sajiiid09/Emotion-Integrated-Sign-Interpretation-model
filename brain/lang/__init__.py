"""Bangla lexicon utilities."""

from .dataset_loader import LexiconEntry, load_dataset, map_category, normalize_token_bn, parse_emotion_list
from .lexicon import (
    LEXICON_ENTRIES,
    LEXICON_SETS,
    PRONOUNS,
    WH_WORDS,
    INSTRUCTION_WORDS,
    TIME_WORDS,
    VERBS,
    INTERACTION,
    EMOTION_ADJ,
    TUTOR_TOPICS,
    get_entry,
    get_category,
    is_category,
)
from .pipeline import run_language_pipeline

__all__ = [
    "LexiconEntry",
    "load_dataset",
    "map_category",
    "normalize_token_bn",
    "parse_emotion_list",
    "LEXICON_ENTRIES",
    "LEXICON_SETS",
    "PRONOUNS",
    "WH_WORDS",
    "INSTRUCTION_WORDS",
    "TIME_WORDS",
    "VERBS",
    "INTERACTION",
    "EMOTION_ADJ",
    "TUTOR_TOPICS",
    "get_entry",
    "get_category",
    "is_category",
    "run_language_pipeline",
]
