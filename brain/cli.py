"""Command-line entrypoint for manual Brain validation."""

from __future__ import annotations

import argparse
from typing import Sequence

from .config import load_config
from .service import respond, respond_from_list
from .types import BrainInput


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain module manual runner")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config()

    if args.tokens:
        tokens = args.tokens.split()
        output = respond_from_list(tokens, cfg=cfg)
    elif args.keywords is not None:
        keywords = args.keywords.split()
        brain_input = BrainInput(keywords=keywords, emotion=args.emotion)  # type: ignore[arg-type]
        output = respond(brain_input, cfg=cfg)
    else:
        raise SystemExit("Provide either --tokens or --keywords")

    print(output.response_bn)
    print(f"status={output.status} emotion={output.resolved_emotion} latency_ms={output.latency_ms}")
    if cfg.debug:
        print(f"debug={output.debug}")


if __name__ == "__main__":
    main()
