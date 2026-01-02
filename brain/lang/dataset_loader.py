"""Dataset loading and normalization helpers for Bangla lexicon generation."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

__all__ = ["LexiconEntry", "normalize_token_bn", "parse_emotion_list", "map_category", "load_dataset"]


@dataclass
class LexiconEntry:
    """Normalized lexicon entry for runtime lookups."""

    bangla: str
    english: str
    category_raw: str
    category: str
    emotions_to_record: List[str]


_SPACE_PATTERN = re.compile(r"\s+")


def normalize_token_bn(token: str | None) -> str:
    """Normalize Bangla token spacing and unicode.

    - Strip leading and trailing whitespace.
    - Collapse multiple internal whitespace to single spaces.
    - Normalize unicode to NFC form.
    - Return an empty string if the token is empty after normalization.
    """

    if token is None:
        return ""

    normalized = str(token)
    normalized = normalized.strip()
    normalized = _SPACE_PATTERN.sub(" ", normalized)
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = normalized.strip()
    return normalized if normalized else ""


_CATEGORY_MAP: Dict[str, str] = {
    "pronoun": "pronoun",
    "wh": "wh",
    "question": "wh",
    "instruction": "instruction",
    "command": "instruction",
    "time": "time",
    "temporal": "time",
    "emotion": "emotion_adj",
    "emotion_adj": "emotion_adj",
    "emotion/adj": "emotion_adj",
    "adj_emotion": "emotion_adj",
    "verb": "verb",
    "verb_emotion": "verb",
    "verb/emotion": "verb",
    "action": "verb",
    "noun": "noun",
    "noun_relation": "noun",
    "noun/relation": "noun",
    "subject": "subject",
    "topic": "subject",
    "concept": "concept",
    "idea": "concept",
    "place": "place",
    "location": "place",
    "object": "object",
    "thing": "object",
    "person": "person",
    "name": "person",
    "interaction": "interaction",
    "greeting": "interaction",
}


def map_category(category_raw: str | None) -> str:
    """Map raw CSV category into a normalized internal category."""

    if category_raw is None:
        return "concept"
    normalized = category_raw.strip().lower()
    normalized = normalized.replace("/", "_").replace("-", "_").replace(" ", "_")
    if not normalized:
        return "concept"
    return _CATEGORY_MAP.get(normalized, normalized)


def parse_emotion_list(value: str | None) -> List[str]:
    """Parse the emotions column into a normalized list."""

    if value is None:
        return []
    parts = re.split(r"[;,]", value)
    normalized_parts: List[str] = []
    for part in parts:
        token = normalize_token_bn(part)
        if token:
            normalized_parts.append(token)
    return normalized_parts


def load_dataset(csv_path: str | Path) -> List[LexiconEntry]:
    """Load and normalize the tutor dataset into lexicon entries."""

    path = Path(csv_path)
    entries: List[LexiconEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bangla_raw = row.get("Bangla")
            bangla = normalize_token_bn(bangla_raw)
            if not bangla:
                continue
            english = row.get("English", "").strip()
            category_raw = row.get("Category", "")
            category = map_category(category_raw)
            emotions = parse_emotion_list(row.get("Emotions to Record"))
            entries.append(
                LexiconEntry(
                    bangla=bangla,
                    english=english,
                    category_raw=category_raw.strip() if category_raw else "",
                    category=category,
                    emotions_to_record=emotions,
                )
            )
    return entries
