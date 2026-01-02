"""Build Bangla lexicon JSON artifact from the tutor dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util

_DATASET_LOADER_PATH = REPO_ROOT / "brain" / "lang" / "dataset_loader.py"
_spec = importlib.util.spec_from_file_location("brain.lang.dataset_loader", _DATASET_LOADER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load dataset_loader module")
dataset_loader = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dataset_loader
_spec.loader.exec_module(dataset_loader)

LexiconEntry = dataset_loader.LexiconEntry
load_dataset = dataset_loader.load_dataset
normalize_token_bn = dataset_loader.normalize_token_bn

def _normalize_set(values):
    return {normalize_token_bn(token) for token in values if normalize_token_bn(token)}


PRONOUNS = _normalize_set({"আমি", "তুমি", "আমরা"})
WH_WORDS = _normalize_set({"কি", "কোথায়", "কেন", "কবে", "কেমন"})
INSTRUCTION_WORDS = _normalize_set({"ব্যাখ্যা", "উদাহরণ", "অর্থ", "সঠিক", "ভুল"})
TIME_WORDS = _normalize_set({"সকাল", "কালকে", "সময়"})
VERBS = _normalize_set({"খাওয়া", "পড়া", "লেখা", "শোনা", "বলা", "দেখা", "চিন্তা", "কাজ", "থামা", "সাহায্য"})
INTERACTION = _normalize_set({"হ্যালো", "বিদায়", "ধন্যবাদ", "হ্যাঁ"})
EMOTION_ADJ = _normalize_set({"ভালো", "খারাপ", "খুশি", "দুঃখ", "রাগ", "অবাক", "গরম", "ঠান্ডা", "সুন্দর", "অসুস্থ", "পছন্দ", "বন্ধু"})
TUTOR_TOPICS = _normalize_set({"গণিত", "বিজ্ঞান", "ইতিহাস", "ভূগোল", "ভাষা", "মহাবিশ্ব", "পৃথিবী", "পরিবেশ", "শরীর", "বাংলাদেশ", "কম্পিউটার"})

DEFAULT_DATASET_PATHS = [Path("AI_Tutor_Dataset.csv"), Path("data/AI_Tutor_Dataset.csv")]
DEFAULT_OUTPUT_PATH = Path("brain/lang/lexicon_bn.json")


def _resolve_dataset_path(arg_path: str | None) -> Path:
    if arg_path:
        candidate = Path(arg_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset not found at provided path: {candidate}")
        return candidate

    for candidate in DEFAULT_DATASET_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "AI_Tutor_Dataset.csv not found. Place it in repo root or data/ and rerun."
    )


def detect_normalization_fixes(dataset_path: Path) -> List[Tuple[str, str]]:
    fixes: List[Tuple[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("Bangla", "")
            normalized = normalize_token_bn(raw)
            if raw is not None and raw != normalized and normalized:
                fixes.append((raw, normalized))
    return fixes


def build_payload(entries: List[LexiconEntry]) -> Dict:
    payload_entries: Dict[str, Dict] = {}
    for entry in entries:
        payload_entries[entry.bangla] = {
            "english": entry.english,
            "category": entry.category,
            "category_raw": entry.category_raw,
            "emotions": entry.emotions_to_record,
        }

    sets = {
        "PRONOUNS": sorted(PRONOUNS),
        "WH_WORDS": sorted(WH_WORDS),
        "INSTRUCTION_WORDS": sorted(INSTRUCTION_WORDS),
        "TIME_WORDS": sorted(TIME_WORDS),
        "VERBS": sorted(VERBS),
        "INTERACTION": sorted(INTERACTION),
        "EMOTION_ADJ": sorted(EMOTION_ADJ),
        "TUTOR_TOPICS": sorted(TUTOR_TOPICS),
    }

    return {"version": 1, "entries": payload_entries, "sets": sets}


def write_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bangla lexicon JSON from dataset")
    parser.add_argument("--dataset", help="Path to AI_Tutor_Dataset.csv")
    parser.add_argument("--output", help="Path to write lexicon JSON", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    output_path = Path(args.output)

    entries = load_dataset(dataset_path)
    payload = build_payload(entries)
    write_json(payload, output_path)

    fixes = detect_normalization_fixes(dataset_path)
    print(f"Dataset path: {dataset_path}")
    print(f"Lexicon written: {output_path}")
    print(f"Total entries: {len(entries)}")
    if len(entries) != 60:
        print("WARNING: Expected 60 entries based on the current dataset contract.")
    if fixes:
        print("Tokens normalized (trailing/extra spaces fixed):")
        for raw, normalized in fixes:
            print(f"  '{raw}' -> '{normalized}'")
    else:
        print("No whitespace fixes were required.")


if __name__ == "__main__":
    main()
