"""Data contract definitions for the Brain module.

These types are designed to be stable across phases; future phases will extend
optional fields rather than change existing ones.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Dict, List

EmotionTag = Literal["neutral", "question", "negation", "happy", "sad"]
BrainStatus = Literal["idle", "listening", "thinking", "ready", "error"]


@dataclass(frozen=True)
class BrainInput:
    """Structured input passed to the Brain service.

    ``keywords`` may contain noisy tokens from CV/ASR; they will be cleaned and
    normalized internally before intent parsing and contradiction resolution.
    """

    keywords: List[str]
    emotion: EmotionTag
    meta: Optional[Dict[str, str]] = None


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
    error: Optional[str]
    latency_ms: Optional[int]
    debug: Dict[str, object]


@dataclass(frozen=True)
class ExecutorSnapshot:
    """Lightweight snapshot of executor state for realtime HUDs."""

    status: BrainStatus
    last_output: Optional["BrainOutput"]
    last_update_ts: float
    request_id: int
    in_flight: bool
    last_error: Optional[str]
    debug: Dict[str, object]
