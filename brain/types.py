"""Data contract definitions for the Brain module.

These types are designed to be stable across phases; future phases will extend
optional fields rather than change existing ones.
"""

from dataclasses import dataclass
from typing import Literal

EmotionTag = Literal["neutral", "question", "negation", "happy", "sad"]
BrainStatus = Literal["idle", "listening", "thinking", "ready", "error"]


@dataclass(frozen=True)
class BrainInput:
    """Structured input passed to the Brain service.

    ``keywords`` may contain noisy tokens from CV/ASR; they will be cleaned and
    normalized internally before intent parsing and contradiction resolution.
    """

    keywords: list[str]
    emotion: EmotionTag
    meta: dict[str, str] | None = None


@dataclass(frozen=True)
class BrainOutput:
    """Structured response returned by the Brain service.

    ``debug`` can include nested objects such as the parsed intent,
    normalization details, resolved intent, rule traces, and token
    statistics for later analysis.
    """

    response_bn: str
    resolved_emotion: EmotionTag
    status: BrainStatus
    error: str | None
    latency_ms: int | None
    debug: dict[str, object]


@dataclass(frozen=True)
class ExecutorSnapshot:
    """Lightweight snapshot of executor state for realtime HUDs."""

    status: BrainStatus
    last_output: BrainOutput | None
    last_update_ts: float
    request_id: int
    in_flight: bool
    last_error: str | None
    debug: dict[str, object]
