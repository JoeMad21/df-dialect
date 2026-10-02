"""Run every check a pull request must pass. Same steps as CI.

Usage (any OS):
    uv run python scripts/check.py          # everything
    uv run python scripts/check.py --fast   # skip pyright

Stops at the first failing step and prints how to fix it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("format", ["ruff", "format", "--check", "."], "run: uv run ruff format ."),
    ("lint", ["ruff", "check", "."], "run: uv run ruff check --fix .  (then fix the rest by hand)"),
    ("types", ["pyright"], "fix the reported type errors in src/ or tests/unit/"),
    ("unit", ["pytest", "-q"], "see docs/TESTING.md; test_registry failures list what is missing"),
    (
        "filecheck",
        ["lit", "-q", "tests/filecheck"],
        "rerun one file with: uv run lit -v tests/filecheck/<path>.mlir",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the pyright step")
    args = parser.parse_args()

    for name, cmd, hint in STEPS:
        if args.fast and name == "types":
            continue
        if shutil.which(cmd[0]) is None:
            print(
                f"[{name}] '{cmd[0]}' not found. Run this through uv: uv run python scripts/check.py"
            )
            return 1
        print(f"[{name}] {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"\nFAILED at step '{name}'. Hint: {hint}")
            return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
