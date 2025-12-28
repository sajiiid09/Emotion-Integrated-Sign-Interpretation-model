"""Gemini client wrapper with retries and safe fallbacks."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterator, Tuple

from .config import BrainConfig
from .constants import (
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TOP_P,
    FALLBACK_BN,
    HARD_OUTPUT_RULE_BN,
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

        last_error: str | None = None
        start = time.perf_counter()

        for attempt in range(attempts):
            meta["attempt_count"] = attempt + 1
            try:
                response = self._client.models.generate_content(
                    model=self.cfg.model_name,
                    contents=content,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=self.cfg.max_output_tokens,
                        temperature=self.cfg.temperature,
                        top_p=DEFAULT_TOP_P,
                    ),
                    timeout=self.cfg.timeout_s,
                )
                raw_text = self._extract_text(response)
                text = postprocess_response_bn(raw_text, self.cfg)
                end = time.perf_counter()
                meta.update(
                    {
                        "latency_ms": int((end - start) * 1000),
                        "raw_length_chars": len(raw_text),
                        "final_word_count": estimate_word_count_bn(text),
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
        try:
            stream = self._client.models.generate_content_stream(
                model=self.cfg.model_name,
                contents=content,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=self.cfg.max_output_tokens,
                    temperature=self.cfg.temperature,
                    top_p=DEFAULT_TOP_P,
                ),
                timeout=self.cfg.timeout_s,
            )
            chunks: list[str] = []
            for chunk in stream:
                part = self._extract_text(chunk)
                if part:
                    chunks.append(part)
            if chunks:
                yield postprocess_response_bn(" ".join(chunks), self.cfg)
        except Exception:
            return

    @staticmethod
    def _extract_text(response: object) -> str:
        if response is None:
            return ""
        # google-genai responses often expose .text
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
        if hasattr(response, "candidates"):
            try:
                for cand in response.candidates:
                    part_text = getattr(cand, "text", "")
                    if part_text:
                        return str(part_text)
            except Exception:
                pass
        return str(response)

    
