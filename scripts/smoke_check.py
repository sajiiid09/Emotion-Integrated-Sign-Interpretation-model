"""Run basic smoke checks for the Brain module."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_smoke.py"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
