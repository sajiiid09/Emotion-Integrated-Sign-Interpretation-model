"""Lightweight structured logging helpers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import BrainConfig


def ensure_log_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def append_event(path: str, event: dict, *, max_bytes: int) -> None:
    ensure_log_dir(path)
    target = Path(path)
    if target.exists() and target.stat().st_size > max_bytes:
        rotated = target.with_name(f"{target.stem}-{int(time.time())}{target.suffix}")
        target.rename(rotated)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def build_log_event(
    *,
    cfg: BrainConfig,
    request_id: int | None,
    keywords: list[str],
    detected_emotion: str,
    resolved_emotion: str,
    status: str,
    latency_ms: int | None,
    used_gemini: bool,
    cache_hit: bool,
    error: str | None,
    prompt_hash: str | None,
    response_preview: str,
) -> dict:
    return {
        "ts": time.time(),
        "request_id": request_id,
        "keywords": keywords,
        "detected_emotion": detected_emotion,
        "resolved_emotion": resolved_emotion,
        "status": status,
        "latency_ms": latency_ms,
        "used_gemini": used_gemini,
        "cache_hit": cache_hit,
        "error": error,
        "prompt_hash": prompt_hash,
        "response_preview": response_preview,
        "model": cfg.model_name,
    }


__all__ = ["append_event", "ensure_log_dir", "build_log_event"]
