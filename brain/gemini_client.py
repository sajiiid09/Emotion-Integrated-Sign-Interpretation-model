"""Gemini client wrapper with retries and safe fallbacks."""

from __future__ import annotations

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import random
import re
import time
from dataclasses import dataclass
from typing import Iterator, Tuple

from .config import BrainConfig
from .constants import (
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TOP_P,
    FALLBACK_BN,
    HARD_OUTPUT_RULE_BN,
    MODE_TUTOR,
)
from .postprocess import estimate_word_count_bn, postprocess_response_bn
from .prompt_builder import BuiltPrompt

try:  # pragma: no cover - runtime import guard
    from google import genai
    from google.genai import types as genai_types

    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - keep runtime safe
    genai = None
    genai_types = None
    _GENAI_AVAILABLE = False


@dataclass(frozen=True)
class GeminiResult:
    text: str
    meta: dict[str, object]


def extract_next_topics_bn(text: str) -> list[str]:
    """Extract next topics from tutor response.
    
    Looks for line starting with "পরের বিষয়:" (or spelling variants with optional spacing)
    and extracts comma-separated topics.
    Handles variants:
    - পরের বিষয়:
    - পরের বিষয়:
    - পরের বিষয় : (with spaces around colon)
    Removes "আরও জানতে চান" if present on same line.
    Returns 3-6 topics, trimmed.
    """
    topics: list[str] = []
    for line in text.split("\n"):
        line_stripped = line.strip()
        # Match both "পরের বিষয়:" and "পরের বিষয়:" with optional spaces around colon
        if re.match(r"পরের\s*বিষয়?\s*:", line_stripped):
            # Extract everything after the colon
            match = re.search(r":\s*(.+)", line_stripped)
            if match:
                topic_part = match.group(1)
                # Remove "আরও জানতে চান" from the end if present
                topic_part = re.sub(r'\s*আরও\s*জানতে\s*চান\??.*$', '', topic_part).strip()
                raw_topics = [t.strip() for t in topic_part.split(",") if t.strip()]
                # Keep 3-6 topics
                topics = raw_topics[:6] if len(raw_topics) >= 3 else raw_topics
                break
    return topics


class GeminiClient:
    """Minimal Gemini wrapper with retries and defensive error handling."""

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self._available = _GENAI_AVAILABLE
        self._client = genai.Client(api_key=cfg.api_key) if _GENAI_AVAILABLE and cfg.api_key else None

    def generate(self, prompt: BuiltPrompt) -> Tuple[str, dict[str, object]]:
        meta: dict[str, object] = {
            "provider": "google-genai",
            "model": self.cfg.model_name,
            "attempt_count": 0,
            "enabled": True,
        }

        if not self._available:
            meta["error"] = "missing_sdk"
            return FALLBACK_BN, meta

        if not self.cfg.api_key:
            meta["error"] = "missing_api_key"
            return FALLBACK_BN, meta

        attempts = max(1, self.cfg.retries + 1)
        content = f"{prompt.as_text}\n\n{HARD_OUTPUT_RULE_BN}" if HARD_OUTPUT_RULE_BN else prompt.as_text

        # Use mode-specific token limits
        max_tokens = self.cfg.tutor_max_output_tokens if prompt.mode == MODE_TUTOR else self.cfg.max_output_tokens

        last_error: str | None = None
        start = time.perf_counter()

        for attempt in range(attempts):
            meta["attempt_count"] = attempt + 1
            try:
                response = self._client.models.generate_content(
                    model=self.cfg.model_name,
                    contents=content,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=self.cfg.temperature,
                        top_p=DEFAULT_TOP_P,
                        http_options=genai_types.HttpOptions(
                            timeout=int(self.cfg.timeout_s * 1000)
                        ),
                    ),
                )
                raw_text = self._extract_text(response)
                
                # Extract next topics from raw text BEFORE postprocessing (Phase 2)
                # This ensures topics aren't lost if postprocessing truncates
                next_topics = extract_next_topics_bn(raw_text) if prompt.mode == MODE_TUTOR else []
                
                text = postprocess_response_bn(raw_text, self.cfg, mode=prompt.mode)
                
                end = time.perf_counter()
                meta.pop("error", None)
                meta.update(
                    {
                        "latency_ms": int((end - start) * 1000),
                        "raw_length_chars": len(raw_text),
                        "final_word_count": estimate_word_count_bn(text),
                        "mode": prompt.mode,
                        "max_output_tokens_used": max_tokens,
                        "next_topics": next_topics,
                    }
                )
                return text, meta
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = str(exc)
                meta["error"] = last_error
                if attempt < attempts - 1:
                    backoff = DEFAULT_RETRY_BACKOFF_S * (2 ** attempt)
                    backoff += random.random() * 0.2
                    time.sleep(backoff)

        end = time.perf_counter()
        meta["latency_ms"] = int((end - start) * 1000)
        if last_error:
            meta["error"] = last_error
        return FALLBACK_BN, meta

    def stream(self, prompt: BuiltPrompt) -> Iterator[str]:
        if not self._available or not self.cfg.api_key:
            return

        content = f"{prompt.as_text}\n\n{HARD_OUTPUT_RULE_BN}" if HARD_OUTPUT_RULE_BN else prompt.as_text
        
        # Use mode-specific token limits
        max_tokens = self.cfg.tutor_max_output_tokens if prompt.mode == MODE_TUTOR else self.cfg.max_output_tokens
        
        try:
            stream = self._client.models.generate_content_stream(
                model=self.cfg.model_name,
                contents=content,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=self.cfg.temperature,
                    top_p=DEFAULT_TOP_P,
                    http_options=genai_types.HttpOptions(
                        timeout=int(self.cfg.timeout_s * 1000)
                    ),
                ),
            )
            chunks: list[str] = []
            for chunk in stream:
                part = self._extract_text(chunk)
                if part:
                    chunks.append(part)
            if chunks:
                yield postprocess_response_bn(" ".join(chunks), self.cfg, mode=prompt.mode)
        except Exception:
            return

    @staticmethod
    def _extract_text(response: object) -> str:
        """Robustly extract text from Gemini response, handling warnings/blocks."""
        if response is None:
            return ""

        # Strategy 1: Iterating parts (The safest way)
        try:
            # Check for candidates list
            candidates = getattr(response, "candidates", [])
            if candidates:
                # Usually we want the first candidate
                first_candidate = candidates[0]
                content = getattr(first_candidate, "content", None)
                if content and hasattr(content, "parts"):
                    # Join all text parts, ignoring 'thought' or other types
                    text_parts = []
                    for part in content.parts:
                        # Check if part has 'text' attribute and it is not empty
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return " ".join(text_parts)
        except Exception:
            pass

        # Strategy 2: Fallback to the property (New SDKs)
        try:
            text = getattr(response, "text", None)
            if isinstance(text, str) and text:
                return text
        except Exception:
            pass

        # Strategy 3: Fallback to string representation if all else fails
        return str(response)

    
