"""Configuration loader for the Brain module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .constants import (
    DEFAULT_DEBUG,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_S,
    MAX_RESPONSE_WORDS,
)


@dataclass(frozen=True)
class BrainConfig:
    model_name: str
    timeout_s: float
    debug: bool
    max_response_words: int


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

    return BrainConfig(
        model_name=model_name,
        timeout_s=timeout_s,
        debug=debug,
        max_response_words=max_response_words,
    )
