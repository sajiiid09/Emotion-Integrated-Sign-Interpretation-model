"""Command-line entrypoint for manual Brain validation."""

from __future__ import annotations

import argparse
from typing import Sequence

from .config import load_config
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config()

    tokens: list[str] | None = None
    keywords: list[str] | None = None

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

    if intent is not None:
        resolved = resolve_emotion(intent)
        prompt = build_prompt(resolved, cfg=cfg)

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


if __name__ == "__main__":
    main()
