"""Configuration loader for the Brain module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .constants import (
    BRAIN_USE_GEMINI_ENV,
    DEFAULT_DEBUG,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RETRY_COUNT,
    DEFAULT_DEBOUNCE_MS,
    DEFAULT_COOLDOWN_MS,
    DEFAULT_QUEUE_MAXSIZE,
    DEFAULT_STREAMING,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    GEMINI_API_KEY_ENV_CANDIDATES,
    MAX_RESPONSE_WORDS,
)


@dataclass(frozen=True)
class BrainConfig:
    model_name: str
    timeout_s: float
    debug: bool
    max_response_words: int
    use_gemini: bool
    api_key: str | None
    temperature: float
    max_output_tokens: int
    retries: int
    streaming: bool
    debounce_ms: int
    cooldown_ms: int
    queue_maxsize: int


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(env: Mapping[str, str] | None = None) -> BrainConfig:
    """Load configuration from environment variables.

    Args:
        env: Optional mapping of environment variables for easier testing.

    Returns:
        A fully populated :class:`BrainConfig` with safe defaults.
    """

    environment = env if env is not None else os.environ

    model_name = environment.get("BRAIN_MODEL_NAME", DEFAULT_MODEL_NAME)
    timeout_s = _parse_float(environment.get("BRAIN_TIMEOUT_S"), DEFAULT_TIMEOUT_S)
    debug = _parse_bool(environment.get("BRAIN_DEBUG"), DEFAULT_DEBUG)
    max_response_words = _parse_int(
        environment.get("BRAIN_MAX_WORDS"), MAX_RESPONSE_WORDS
    )
    use_gemini = _parse_bool(environment.get(BRAIN_USE_GEMINI_ENV), False)

    api_key: str | None = None
    for key_name in GEMINI_API_KEY_ENV_CANDIDATES:
        candidate = environment.get(key_name)
        if candidate:
            api_key = candidate
            break

    temperature = _parse_float(
        environment.get("BRAIN_TEMPERATURE"), DEFAULT_TEMPERATURE
    )
    max_output_tokens = _parse_int(
        environment.get("BRAIN_MAX_OUTPUT_TOKENS"), DEFAULT_MAX_OUTPUT_TOKENS
    )
    retries = _parse_int(environment.get("BRAIN_RETRIES"), DEFAULT_RETRY_COUNT)
    streaming = _parse_bool(environment.get("BRAIN_STREAMING"), DEFAULT_STREAMING)
    debounce_ms = _parse_int(
        environment.get("BRAIN_DEBOUNCE_MS"), DEFAULT_DEBOUNCE_MS
    )
    cooldown_ms = _parse_int(
        environment.get("BRAIN_COOLDOWN_MS"), DEFAULT_COOLDOWN_MS
    )
    queue_maxsize = _parse_int(
        environment.get("BRAIN_QUEUE_MAXSIZE"), DEFAULT_QUEUE_MAXSIZE
    )

    return BrainConfig(
        model_name=model_name,
        timeout_s=timeout_s,
        debug=debug,
        max_response_words=max_response_words,
        use_gemini=use_gemini,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
        streaming=streaming,
        debounce_ms=debounce_ms,
        cooldown_ms=cooldown_ms,
        queue_maxsize=queue_maxsize,
    )
