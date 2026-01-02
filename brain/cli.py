"""Command-line entrypoint for manual Brain validation."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Sequence

from .config import load_config
from .executor import BrainExecutor
from .gemini_client import GeminiClient
from .lang.pipeline import run_language_pipeline
from .prompt_builder import build_prompt
from .rules import resolve_emotion
from .service import (
    parse_intent_from_input,
    parse_intent_from_tokens,
    respond,
    respond_from_list,
    split_keywords_text,
)
from .types import BrainInput


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain module manual runner",
        epilog=(
            "Examples:\n"
            "  python -m brain.cli --tokens \"দুঃখ negation\" --show-debug\n"
            "  python -m brain.cli --tokens \"খুশি question\" --show-debug\n"
            "  python -m brain.cli --tokens \"মহাবিশ্ব question\" --show-prompt\n"
            "  python -m brain.cli --tokens \"খারাপ sad\" --prompt-only\n"
            "  python -m brain.cli --simulate-realtime --sequence \"আমি neutral|আমি question\"\n"
        ),
    )
    parser.add_argument(
        "--keywords",
        type=str,
        help="Space separated keywords",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        help="Emotion tag (neutral/question/negation/happy/sad)",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        help="Space separated tokens; last token may be an emotion tag",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Print debug info even when debug config is off",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the built prompt after the response",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print only the prompt (no stub response)",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Force Gemini usage for this run",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini usage for this run",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Attempt streaming Gemini output (best effort)",
    )
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Simulate realtime submissions via the background executor",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Submission rate for realtime simulation",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="'|' separated token sequences for realtime simulation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config()
    if args.use_gemini:
        cfg = replace(cfg, use_gemini=True)
    if args.no_gemini:
        cfg = replace(cfg, use_gemini=False)
    if args.stream:
        cfg = replace(cfg, streaming=True)

    tokens: list[str] | None = None
    keywords: list[str] | None = None

    if args.simulate_realtime:
        if not args.sequence:
            raise SystemExit("Provide --sequence for realtime simulation")
        _run_realtime_simulation(args, cfg)
        return

    if args.tokens:
        tokens = split_keywords_text(args.tokens)
    elif args.keywords is not None:
        keywords = split_keywords_text(args.keywords)
    else:
        raise SystemExit("Provide either --tokens or --keywords")

    intent = None
    resolved = None
    prompt = None

    if tokens is not None:
        intent, _, _ = parse_intent_from_tokens(tokens)
    elif keywords is not None:
        brain_input = BrainInput(keywords=keywords, emotion=args.emotion)  # type: ignore[arg-type]
        intent, _, _ = parse_intent_from_input(brain_input)

    shaped = None
    if intent is not None:
        resolved = resolve_emotion(intent)
        shaped = run_language_pipeline(resolved.keywords, resolved.resolved_emotion)
        prompt = build_prompt(resolved, shaped=shaped, cfg=cfg)

    if args.prompt_only:
        if prompt is None or resolved is None:
            raise SystemExit("Unable to build prompt")
        print(prompt.system)
        print()
        print(prompt.user)
        print(
            f"resolved_emotion={resolved.resolved_emotion} keywords={' '.join(resolved.keywords)}"
        )
        if args.show_debug or cfg.debug:
            print(f"prompt_debug={prompt.debug}")
            if resolved.rule_trace:
                print(f"rule_trace={resolved.rule_trace}")
        return

    if args.stream:
        if prompt is None or resolved is None:
            raise SystemExit("Unable to stream without a prompt")
        if not cfg.use_gemini:
            raise SystemExit("Streaming requires Gemini; enable with --use-gemini")
        client = GeminiClient(cfg)
        chunks = list(client.stream(prompt))
        meta = {"enabled": cfg.use_gemini}
        response_text = ""
        status = "ready"
        if chunks:
            response_text = "".join(chunks)
            meta["chunk_count"] = len(chunks)
        else:
            response_text, meta = client.generate(prompt)
            status = "error" if meta.get("error") else "ready"
        print(response_text)
        print(
            f"status={status} emotion={resolved.resolved_emotion} keywords={' '.join(resolved.keywords)}"
        )
        if args.show_prompt:
            print("\n--- PROMPT ---")
            print(prompt.system)
            print()
            print(prompt.user)
        if args.show_debug or cfg.debug:
            print(f"gemini={meta}")
            if resolved.rule_trace:
                print(f"rule_trace={resolved.rule_trace}")
        return

    if tokens is not None:
        output = respond_from_list(tokens, cfg=cfg)
    elif keywords is not None:
        brain_input = BrainInput(keywords=keywords, emotion=args.emotion)  # type: ignore[arg-type]
        output = respond(brain_input, cfg=cfg)

    print(output.response_bn)
    print(f"status={output.status} emotion={output.resolved_emotion} latency_ms={output.latency_ms}")
    if cfg.debug or args.show_debug:
        print(f"debug={output.debug}")
    if args.show_prompt and prompt is not None:
        print("\n--- PROMPT ---")
        print(prompt.system)
        print()
        print(prompt.user)


def _run_realtime_simulation(args: argparse.Namespace, cfg) -> None:
    executor = BrainExecutor(cfg)
    executor.start()
    segments = [seg.strip() for seg in (args.sequence or "").split("|") if seg.strip()]
    if not segments:
        raise SystemExit("No segments found in --sequence")

    interval = 1.0 / max(args.fps, 1)
    print(f"Starting realtime simulation with {len(segments)} steps")
    last_status = None
    last_request_id = -1
    expected_requests = len(segments)
    try:
        for raw in segments:
            tokens = split_keywords_text(raw)
            executor.submit_tokens(tokens)
            print(f"submitted: {tokens}")
            time.sleep(interval)
            snapshot = executor.poll_latest()
            if snapshot.status != last_status or snapshot.request_id != last_request_id:
                print(
                    f"status={snapshot.status} request_id={snapshot.request_id} in_flight={snapshot.in_flight}"
                )
                last_status = snapshot.status
                last_request_id = snapshot.request_id

        end_time = time.time() + 3.0
        while time.time() < end_time:
            snapshot = executor.poll_latest()
            if snapshot.status != last_status or snapshot.request_id != last_request_id:
                print(
                    f"status={snapshot.status} request_id={snapshot.request_id} in_flight={snapshot.in_flight}"
                )
                last_status = snapshot.status
                last_request_id = snapshot.request_id
                if snapshot.last_output:
                    print(
                        f"output={snapshot.last_output.response_bn} emotion={snapshot.last_output.resolved_emotion}"
                    )
            if (
                not snapshot.in_flight
                and snapshot.last_output
                and snapshot.request_id >= expected_requests
                and snapshot.status in {"ready", "error"}
            ):
                break
            time.sleep(0.1)
        final_snapshot = executor.poll_latest()
        print(
            f"final status={final_snapshot.status} request_id={final_snapshot.request_id} in_flight={final_snapshot.in_flight}"
        )
        if final_snapshot.last_output:
            print(
                f"final output={final_snapshot.last_output.response_bn} emotion={final_snapshot.last_output.resolved_emotion}"
            )
    finally:
        executor.stop()
        print("executor stopped")


if __name__ == "__main__":
    main()
