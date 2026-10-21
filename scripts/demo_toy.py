"""Walk through the toy lowerings to Tenstorrent TTIR and Cerebras CSL, one step at a time.

Usage (any OS):
    uv run python scripts/demo_toy.py             # pause after each step
    uv run python scripts/demo_toy.py --no-pause  # run straight through

Generated files go to build/ (ignored by git):
    build/toy_linear.ttir.mlir        TTIR for tt-mlir
    build/toy_linear_pe.cslir.mlir    CSL-IR (xDSL csl dialect)
    build/csl/linear.csl              CSL program for the PE
    build/csl/linear_layout.csl       CSL layout; the file cslc compiles
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
CSL_PIPELINE = "convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper,memref-to-dsd"
PAUSE = "--no-pause" not in sys.argv


def tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        sys.exit(f"'{name}' not found. Run this through uv: uv run python scripts/demo_toy.py")
    return path


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def step(title: str, cmd: list[str] | None = None, show: Path | None = None) -> None:
    print(f"\n== {title} ==")
    if cmd is not None:
        print("> " + " ".join(Path(cmd[0]).stem if i == 0 else c for i, c in enumerate(cmd)))
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print((result.stdout + result.stderr).rstrip())
    if show is not None:
        print(f"--- {show.relative_to(ROOT).as_posix()} ---")
        print(show.read_text(encoding="utf-8").rstrip())
    if PAUSE:
        input("\n[Enter] ")


def main() -> int:
    df_opt = tool("df-opt")
    BUILD.mkdir(exist_ok=True)
    ttir = BUILD / "toy_linear.ttir.mlir"
    cslir = BUILD / "toy_linear_pe.cslir.mlir"
    csl_txt = BUILD / "toy_linear_pe.csl.txt"

    step("1. Toy program: relu(x @ w + b)", show=ROOT / "examples" / "toy_linear.mlir")
    step(
        "2. Lower to Tenstorrent TTIR",
        [df_opt, "examples/toy_linear.mlir", "-p", "convert-toy-to-ttir", "-o", rel(ttir)],
        show=ttir,
    )
    step(
        "3. One Cerebras PE has 48 KB: the 64x128 program does not fit",
        [df_opt, "examples/toy_linear.mlir", "-p", "convert-toy-to-csl", "--verify-diagnostics"],
    )
    step("4. Same layer, sized for one PE", show=ROOT / "examples" / "toy_linear_pe.mlir")
    step(
        "5. Lower to CSL-IR (xDSL csl dialect)",
        [
            df_opt,
            "examples/toy_linear_pe.mlir",
            "-p",
            "convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper",
            "-o",
            rel(cslir),
        ],
        show=cslir,
    )
    step(
        "6. Print CSL source and split it into program and layout files",
        [
            df_opt,
            "examples/toy_linear_pe.mlir",
            "-p",
            CSL_PIPELINE,
            "-t",
            "csl",
            "-o",
            rel(csl_txt),
        ],
    )
    step(
        "7. CSL files for the Cerebras compiler",
        [sys.executable, "scripts/split_csl.py", rel(csl_txt), rel(BUILD / "csl")],
        show=BUILD / "csl" / "linear.csl",
    )
    step("8. Tests for both lowerings", [tool("lit"), "-v", "tests/filecheck/toy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
