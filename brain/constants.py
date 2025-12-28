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

# Phase 3 rule helpers
EMOTION_KEYWORDS_MAP: dict[str, str] = {
    "খুশি": "happy_word",
    "দুঃখ": "sad_word",
    "রাগ": "angry_word",
    "ভালো": "good_word",
    "খারাপ": "bad_word",
    "অবাক": "surprised_word",
    "অসুস্থ": "sick_word",
    "অসু9": "sick_word",
    "সুন্দর": "beautiful_word",
}

RULE_PRIORITY = ["negation", "question", "sad", "happy", "neutral"]

# Optional templates for future stub adjustments
NEGATION_HINT_BN = "ঠিক আছে, এটা নয়।"

# Phase 4 prompt builder constants
BASE_SYSTEM_INSTRUCTION_BN = (
    "আপনি বাংলাদেশি বধির শিক্ষার্থীদের জন্য একজন বন্ধুসুলভ এআই টিউটর। "
    "ব্যবহারকারী বিচ্ছিন্ন কিওয়ার্ড দিয়ে কথা বলেন; আপনি উদ্দেশ্য বুঝে সাহায্য করবেন। "
    "উত্তর বাংলায়, ২-৩ বাক্য, সর্বোচ্চ ৪০ শব্দের মধ্যে রাখুন। "
    "ব্যবহারকারী দুঃখিত হলে সহানুভূতিশীল, খুশি হলে প্রাণবন্ত, প্রশ্ন হলে সরাসরি উত্তর দিন। "
    "ইনপুটে ভুল থাকলে সম্ভাব্য উদ্দেশ্য ধরে সহায়তা করুন।"
)

DYNAMIC_RULES_BY_TAG: dict[str, str] = {
    "question": "প্রশ্নের উত্তর সরাসরি ও সহজভাবে দিন।",
    "happy": "উচ্ছ্বাস মেলে ধরুন এবং ইতিবাচক টোন রাখুন।",
    "sad": "নরম সুরে আশ্বাস দিন এবং সহমর্মী হোন।",
    "negation": "অস্বীকৃতি ধরে নিয়ে স্পষ্টীকরণ চেয়ে নিন।",
    "neutral": "সাধারণ সহায়ক টিউটর টোন বজায় রাখুন।",
}

OUTPUT_CONSTRAINTS_BN = (
    "শুধু বাংলায় লিখুন, সর্বোচ্চ ৪০ শব্দ, ২-৩ বাক্যে, কোনো বুলেট লিস্ট নয়।"
)
