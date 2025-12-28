import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from brain import (  # noqa: E402
    BrainExecutor,
    BrainInput,
    ResolvedIntent,
    build_prompt,
    load_config,
    postprocess_response_bn,
    respond_from_list,
)
from brain.intent import Intent  # noqa: E402
from brain.types import BrainOutput  # noqa: E402


def test_imports_and_executor_lifecycle():
    executor = BrainExecutor(load_config())
    executor.start()
    executor.stop()


def test_stub_response_non_empty():
    output = respond_from_list(["আমি", "ভাত", "neutral"], cfg=load_config())
    assert isinstance(output, BrainOutput)
    assert output.response_bn


def test_prompt_builder_smoke():
    intent = Intent(keywords=["গণিত"], raw_keywords=["গণিত"], detected_emotion="question", meta=None, flags={}, notes=[])
    resolved = ResolvedIntent(
        keywords=intent.keywords,
        detected_emotion=intent.detected_emotion,
        resolved_emotion="question",
        meta=None,
        flags={},
        notes=[],
        rule_trace=[],
    )
    prompt = build_prompt(resolved, cfg=load_config())
    assert prompt.system and prompt.user and prompt.as_text


def test_postprocess_markdown_strip():
    cfg = load_config()
    cleaned = postprocess_response_bn("```code``` হ্যালো", cfg)
    assert "```" not in cleaned
    assert cleaned
