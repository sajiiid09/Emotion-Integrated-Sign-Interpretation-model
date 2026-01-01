"""Module-wide constants for the Brain system."""

FALLBACK_BN = "সংযোগ সমস্যা, আবার চেষ্টা করুন"
DEFAULT_BN = "আমি প্রস্তুত। আপনি কোন বিষয়ে জানতে চান?"
MAX_RESPONSE_WORDS = 40
TUTOR_MAX_RESPONSE_WORDS = 260
ALLOWED_TAGS = ("neutral", "question", "negation", "happy", "sad")
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_TIMEOUT_S = 12.0
DEFAULT_DEBUG = False
# Phase 1: Tutor mode constants
MODE_REALTIME = "realtime"
MODE_TUTOR = "tutor"

TUTOR_TOPIC_KEYWORDS_BN = {
    "বিজ্ঞান", "গণিত", "ইতিহাস", "ভূগোল", "মহাবিশ্ব", "কম্পিউটার",
    "বাংলাদেশ", "ঢাকা", "প্রযুক্তি", "জীববিজ্ঞান", "পদার্থবিজ্ঞান",
    "রসায়ন", "ভাষা", "সাহিত্য", "শিল্প", "ধর্ম", "অর্থনীতি"
}

TUTOR_MIN_WORDS = 120
TUTOR_TARGET_WORDS_MIN = 160
TUTOR_TARGET_WORDS_MAX = 240
TUTOR_MAX_SENTENCES = 4
FOLLOWUP_QUESTION_BN = "আপনি কি এই বিষয়ের আরও গভীরে যেতে চান?"
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
TUTOR_EXAMPLE_OUTPUT_BN = (
    "উদাহরণ (টিউটর মোডে উত্তর কীভাবে হবে): "
    "কম্পিউটার একটি শক্তিশালী যন্ত্র যা হাজার হাজার গণনা দ্রুত করে। এটি তথ্য সংরক্ষণ, প্রক্রিয়াকরণ এবং প্রদর্শন করে। আমরা দৈনন্দিন জীবনে কম্পিউটার ব্যবহার করি। স্মার্টফোনও একটি ধরনের কম্পিউটার। কম্পিউটারের তিনটি প্রধান অংশ রয়েছে: হার্ডওয়্যার, সফটওয়্যার এবং ডেটা। প্রতিটি অংশ একসাথে কাজ করে দরকারি ফলাফল দেয়। আপনার স্কুলের ল্যাবেও কম্পিউটার আছে যেখানে আমরা শিখি। "
    "পরের বিষয়: হার্ডওয়্যার, সফটওয়্যার, ডিজিটাল নিরাপত্তা "
    "আরও জানতে চান?"
)

BASE_SYSTEM_INSTRUCTION_BN = (
    "আপনি বাংলাদেশি বধির শিক্ষার্থীদের জন্য একজন বন্ধুসুলভ এআই টিউটর। "
    "ব্যবহারকারী বিচ্ছিন্ন কিওয়ার্ড দিয়ে কথা বলেন; আপনি উদ্দেশ্য বুঝে সাহায্য করবেন। "
    "উত্তর শুধু বাংলায়, মার্কডাউন ছাড়াই, কোনো তালিকা বা শিরোনাম নয়। "
    "উত্তরে কীওয়ার্ড, ট্যাগ, মোড, ইনপুট এসব শব্দ উল্লেখ করবেন না। "
    "টিউটর মোডে এই নিয়ম মেনে চলুন: "
    "- একশত ষাট থেকে দুইশত বিশটি বাংলা শব্দ লিখুন। "
    "- ছয় থেকে আটটি ছোট বাক্য লিখুন। "
    "- সর্বদা একটি সহজ উদাহরণ বা উপমা দিন। "
    "- শেষে ঠিক এই দুই লাইন দিয়ে শেষ করুন (অন্য কিছু নয়): "
    "পরের বিষয়: [বিষয় এক, বিষয় দুই, বিষয় তিন] "
    "আরও জানতে চান? "
    f"\n\n{TUTOR_EXAMPLE_OUTPUT_BN}"
)

DYNAMIC_RULES_BY_TAG: dict[str, str] = {
    "question": "প্রশ্নের উত্তর দিন এবং সহজভাবে বিস্তারিত ব্যাখ্যা করুন।",
    "happy": "উৎসাহী এবং উৎসাহী টোন বজায় রাখুন।",
    "sad": "সহানুভূতিশীল এবং আশ্বাসী টোন।",
    "negation": "ভুল ধারণা স্পষ্ট করুন। প্রয়োজনে একটি হ্যাঁ/না প্রশ্ন করুন।",
    "neutral": "স্বাভাবিক সহায়ক টোন।",
}

OUTPUT_CONSTRAINTS_BN = (
    "রিয়েলটাইম মোডে: ২-৩টি ছোট বাক্য, সর্বোচ্চ ৪০ শব্দ। "
    "টিউটর মোডে অবশ্যই এই ফরম্যাট অনুসরণ করুন: "
    "১) ব্যাখ্যা ব্লক (ন্যূনতম ১২০ শব্দ, প্রাকৃতিক অনুচ্ছেদ), "
    "২) 'পরের বিষয়: X, Y, Z' লাইন (৩-৬টি সম্পর্কিত বিষয়), "
    "৩) 'আরও জানতে চান' দিয়ে শেষ লাইন।"
)

# Mode-specific rules (applied before tag rules in prompt_builder.py)
MODE_RULES_BY_MODE: dict[str, str] = {
    "tutor": (
        "টিউটর মোডে এই নির্দেশাবলী অনুসরণ করুন: "
        "ক. একশত ষাট থেকে দুইশত বিশটি বাংলা শব্দ লিখুন। "
        "খ. ছয় থেকে আটটি ছোট বাক্য লিখুন। "
        "গ. অবশ্যই একটি সহজ উদাহরণ বা উপমা অন্তর্ভুক্ত করুন। "
        "ঘ. শেষে ঠিক এই দুই লাইন দিয়ে শেষ করুন (অন্য লাইন নয়): "
        "পরের বিষয়: [বিষয় এক], [বিষয় দুই], [বিষয় তিন] "
        "আরও জানতে চান? "
        "ঙ. শেষ করার আগে যাচাই করুন উত্তর একশত ষাট শব্দের কম নয়।"
    ),
    "realtime": "সংক্ষিপ্ত এবং সরাসরি উত্তর দিন।",
}
# Phase 5 Gemini client constants
GEMINI_API_KEY_ENV_CANDIDATES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
BRAIN_USE_GEMINI_ENV = "BRAIN_USE_GEMINI"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_OUTPUT_TOKENS = 500
DEFAULT_TUTOR_MAX_OUTPUT_TOKENS = 800
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_BACKOFF_S = 0.6
DEFAULT_STREAMING = False
HARD_OUTPUT_RULE_BN = "শুধু বাংলায় লিখুন, মার্কডাউন ছাড়া, তালিকা নেই, শিরোনাম নেই।"

# Phase 2: Request minimization and caching
DEFAULT_TRIGGER_POLICY = "smart"
DEFAULT_PHRASE_PAUSE_MS = 900
DEFAULT_MIN_DELTA_WORDS_FOR_NEW_CALL = 1
DEFAULT_CACHE_MAX_ITEMS = 256

# Phase 6 executor defaults
DEFAULT_DEBOUNCE_MS = 350
DEFAULT_COOLDOWN_MS = 1200
DEFAULT_QUEUE_MAXSIZE = 2

# Phase 8 logging/cache defaults
DEFAULT_LOG_PATH = "logs/brain_events.jsonl"
DEFAULT_LOG_MAX_BYTES = 5_000_000
DEFAULT_CACHE_TTL_S = 30.0
DEFAULT_CACHE_ENABLED = True
DEFAULT_LOG_ENABLED = False
