"""Module-wide constants for the Brain system."""

FALLBACK_BN = "সংযোগ সমস্যা, আবার চেষ্টা করুন"
DEFAULT_BN = "আমি প্রস্তুত। আপনি কোন বিষয়ে জানতে চান?"
MAX_RESPONSE_WORDS = 40
ALLOWED_TAGS = ("neutral", "question", "negation", "happy", "sad")
DEFAULT_MODEL_NAME = "gemini-1.5-flash"
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_DEBUG = False

# Phase 2 parsing helpers
UNKNOWN_TOKEN_PATTERNS = ("???", "unknown", "null", "none")
PUNCT_STRIP_CHARS = ".,!?\"'()[]{}<>|/\\"
MAX_KEYWORDS = 12
DEDUPE_WINDOW = 2
