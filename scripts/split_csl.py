"""Split `df-opt ... -t csl` output into the files the Cerebras compiler expects.

xDSL prints the program and layout modules as one stream, each starting with a
`// FILE: <name>` line and separated by `// -----`. This writes:

    <program_name>_program  ->  <program_name>.csl          (name used by @set_tile_code)
    <program_name>_layout   ->  <program_name>_layout.csl   (pass this one to cslc)

Usage:
    uv run df-opt IN.mlir -p ... -t csl -o build/linear.csl.txt
    uv run python scripts/split_csl.py build/linear.csl.txt build/csl
"""

from __future__ import annotations

import sys
from pathlib import Path


def split(text: str) -> dict[str, str]:
    files: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("// FILE: "):
            name = line.removeprefix("// FILE: ").strip()
            if name.endswith("_program"):
                name = name.removesuffix("_program") + ".csl"
            elif name.endswith("_layout"):
                name = name + ".csl"
            else:
                name = name + ".csl"
            current = files.setdefault(name, [])
        elif line.strip() == "// -----":
            current = None
        elif current is not None:
            current.append(line)
    if not files:
        raise ValueError("no '// FILE:' markers found; was the input produced with -t csl?")
    return {name: "\n".join(lines).strip("\n") + "\n" for name, lines in files.items()}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in split(src.read_text(encoding="utf-8")).items():
        (out_dir / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
