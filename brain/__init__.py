"""Public API for the Brain module."""

from .config import BrainConfig, load_config
from .intent import Intent, ResolvedIntent
from .prompt_builder import BuiltPrompt, build_prompt
from .gemini_client import GeminiClient
from .rules import detect_emotion_keywords, resolve_emotion
from .service import (
    normalize_keywords,
    parse_intent_from_input,
    parse_intent_from_tokens,
    build_output,
    respond,
    respond_from_list,
    validate_emotion,
)
from .types import BrainInput, BrainOutput, BrainStatus, EmotionTag, ExecutorSnapshot
from .postprocess import estimate_word_count_bn, postprocess_response_bn
from .cache import ResponseCache
from .logging_utils import append_event, ensure_log_dir
from .executor import BrainExecutor

__all__ = [
    "BrainConfig",
    "BrainInput",
    "BrainOutput",
    "BrainStatus",
    "EmotionTag",
    "ExecutorSnapshot",
    "Intent",
    "ResolvedIntent",
    "BrainExecutor",
    "GeminiClient",
    "load_config",
    "build_prompt",
    "BuiltPrompt",
    "resolve_emotion",
    "detect_emotion_keywords",
    "build_output",
    "normalize_keywords",
    "parse_intent_from_input",
    "parse_intent_from_tokens",
    "respond",
    "respond_from_list",
    "validate_emotion",
    "postprocess_response_bn",
    "estimate_word_count_bn",
    "ResponseCache",
    "append_event",
    "ensure_log_dir",
]
