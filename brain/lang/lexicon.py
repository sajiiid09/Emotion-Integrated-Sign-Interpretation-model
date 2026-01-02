"""Runtime Bangla lexicon utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Set

from .dataset_loader import LexiconEntry, normalize_token_bn


def _normalize_token_set(values: Set[str]) -> Set[str]:
    return {normalize_token_bn(token) for token in values if normalize_token_bn(token)}


PRONOUNS: Set[str] = _normalize_token_set({"আমি", "তুমি", "আমরা"})
WH_WORDS: Set[str] = _normalize_token_set({"কি", "কোথায়", "কেন", "কবে", "কেমন"})
INSTRUCTION_WORDS: Set[str] = _normalize_token_set({"ব্যাখ্যা", "উদাহরণ", "অর্থ", "সঠিক", "ভুল"})
TIME_WORDS: Set[str] = _normalize_token_set({"সকাল", "কালকে", "সময়"})
VERBS: Set[str] = _normalize_token_set({"খাওয়া", "পড়া", "লেখা", "শোনা", "বলা", "দেখা", "চিন্তা", "কাজ", "থামা", "সাহায্য"})
INTERACTION: Set[str] = _normalize_token_set({"হ্যালো", "বিদায়", "ধন্যবাদ", "হ্যাঁ"})
EMOTION_ADJ: Set[str] = _normalize_token_set({"ভালো", "খারাপ", "খুশি", "দুঃখ", "রাগ", "অবাক", "গরম", "ঠান্ডা", "সুন্দর", "অসুস্থ", "পছন্দ", "বন্ধু"})
TUTOR_TOPICS: Set[str] = _normalize_token_set({"গণিত", "বিজ্ঞান", "ইতিহাস", "ভূগোল", "ভাষা", "মহাবিশ্ব", "পৃথিবী", "পরিবেশ", "শরীর", "বাংলাদেশ", "কম্পিউটার"})

LEXICON_JSON_PATH = Path(__file__).with_name("lexicon_bn.json")


def _load_lexicon_json() -> Dict:
    if not LEXICON_JSON_PATH.exists():
        raise FileNotFoundError(
            "Lexicon file missing. Run scripts/build_lexicon.py to generate brain/lang/lexicon_bn.json."
        )
    with LEXICON_JSON_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


_raw_data = _load_lexicon_json()
LEXICON_SETS: Dict[str, Set[str]] = {name: set(values) for name, values in _raw_data.get("sets", {}).items()}
LEXICON_ENTRIES: Dict[str, LexiconEntry] = {}
for bangla, payload in _raw_data.get("entries", {}).items():
    LEXICON_ENTRIES[bangla] = LexiconEntry(
        bangla=bangla,
        english=payload.get("english", ""),
        category_raw=payload.get("category_raw", ""),
        category=payload.get("category", ""),
        emotions_to_record=payload.get("emotions", []),
    )


def get_entry(token: Optional[str]) -> Optional[LexiconEntry]:
    """Return normalized lexicon entry for the given Bangla token."""

    normalized = normalize_token_bn(token)
    if not normalized:
        return None
    return LEXICON_ENTRIES.get(normalized)


def get_category(token: Optional[str]) -> Optional[str]:
    entry = get_entry(token)
    return entry.category if entry else None


def is_category(token: Optional[str], category: str) -> bool:
    return get_category(token) == category


__all__ = [
    "PRONOUNS",
    "WH_WORDS",
    "INSTRUCTION_WORDS",
    "TIME_WORDS",
    "VERBS",
    "INTERACTION",
    "EMOTION_ADJ",
    "TUTOR_TOPICS",
    "LEXICON_ENTRIES",
    "LEXICON_SETS",
    "get_entry",
    "get_category",
    "is_category",
]
