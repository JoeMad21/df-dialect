"""Create the skeleton of a new op or pass, with every required piece in place.

Usage:
    uv run python scripts/scaffold.py op map --tier a
    uv run python scripts/scaffold.py pass df-place

For an op this:
  1. appends a stub class to src/df_dialect/df/tier_a.py or tier_b.py,
  2. registers it in OPS in src/df_dialect/df/__init__.py,
  3. creates tests/filecheck/dialect/<name>.mlir (generic-form round trip),
  4. adds an "In Progress" section to docs/SPEC.md.

For a pass this:
  1. creates src/df_dialect/passes/<name>.py with a ModulePass stub,
  2. registers it in ALL_PASSES in src/df_dialect/passes/__init__.py,
  3. creates tests/filecheck/passes/<name>.mlir,
  4. adds an "In Progress" section to docs/SPEC.md.

The result passes `make check` immediately. Then replace every TODO.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DF = ROOT / "src" / "df_dialect" / "df"
PASSES = ROOT / "src" / "df_dialect" / "passes"
SPEC = ROOT / "docs" / "SPEC.md"


def camel(snake: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_-]", snake))


def insert_before(path: Path, marker: str, text: str) -> None:
    content = path.read_text(encoding="utf-8")
    if marker not in content:
        sys.exit(f"could not find '{marker}' in {path}")
    path.write_text(content.replace(marker, text + marker, 1), encoding="utf-8")


def scaffold_op(name: str, tier: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        sys.exit("op name must be lowercase letters, digits and underscores, e.g. 'map'")
    cls = camel(name) + "Op"
    op_name = f"df.{name}"
    module = DF / f"tier_{tier}.py"
    if f'name = "{op_name}"' in module.read_text(encoding="utf-8"):
        sys.exit(f"{op_name} already exists")

    with module.open("a", encoding="utf-8") as f:
        f.write(
            f'''

@irdl_op_definition
class {cls}(IRDLOperation):
    """TODO: One-line summary of {op_name}.

    Syntax:
        TODO: show one example line, as in docs/SPEC.md.
    """

    name = "{op_name}"

    # TODO: declare operands, results, properties and regions here.
    # See docs/playbooks/ADD_OP.md for operand_def, result_def, prop_def, region_def.

    # TODO: replace the generic syntax with a readable one, for example:
    # assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def verify_(self) -> None:
        # TODO: add checks. Raise VerifyException("clear message") on failure.
        pass
'''
        )

    init = DF / "__init__.py"
    insert_before(init, "\nOPS = [", f"from df_dialect.df.tier_{tier} import {cls}\n")
    marker = "    # Tier B\n" if tier == "a" else "]\n\nATTRIBUTES"
    insert_before(init, marker, f"    {cls},\n")

    test = ROOT / "tests" / "filecheck" / "dialect" / f"{name}.mlir"
    test.write_text(
        f"""// Round-trip test for {op_name}.
// RUN: df-opt %s | df-opt | filecheck %s
// RUN: df-opt %s --print-op-generic | df-opt | filecheck %s

// TODO: replace with real uses of {op_name} once it has operands and syntax.
"{op_name}"() : () -> ()
// CHECK: "{op_name}"() : () -> ()
""",
        encoding="utf-8",
    )

    heading = "## Tier B Ops" if tier == "a" else "## Passes"
    insert_before(
        SPEC,
        heading,
        f"### `{op_name}` - In Progress\n\nTODO: syntax example, operands, results, "
        f"verifier rules.\n\n",
    )
    print(f"created {cls} ({op_name}) in {module.relative_to(ROOT)}")
    print(f"created {test.relative_to(ROOT)}")


def scaffold_pass(name: str) -> None:
    if not re.fullmatch(r"df-[a-z][a-z0-9-]*", name):
        sys.exit("pass name must start with 'df-' and use lowercase and dashes, e.g. 'df-place'")
    snake = name.replace("-", "_")
    cls = camel(name.removeprefix("df-")) + "Pass"
    module = PASSES / f"{snake}.py"
    if module.exists():
        sys.exit(f"{module.relative_to(ROOT)} already exists")

    module.write_text(
        f'''"""{name}: TODO one-line summary.

Usage:
    df-opt input.mlir -p {name}
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from df_dialect.df import ProgramOp


class TodoPattern(RewritePattern):
    """TODO: rename, and change ProgramOp to the op this pattern rewrites."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ProgramOp, rewriter: PatternRewriter) -> None:
        # TODO: inspect `op` and call rewriter.replace(op, new_op) or rewriter.erase_op(op).
        return


@dataclass(frozen=True)
class {cls}(ModulePass):
    """TODO: describe what {name} does."""

    name = "{name}"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(TodoPattern()).rewrite_module(op)
''',
        encoding="utf-8",
    )

    init = PASSES / "__init__.py"
    insert_before(
        init,
        "\n\nALL_PASSES",
        f"\n\ndef _{snake}() -> type[ModulePass]:\n"
        f"    from df_dialect.passes.{snake} import {cls}\n\n"
        f"    return {cls}\n",
    )
    insert_before(init, "}\n", f'    "{name}": _{snake},\n')

    test = ROOT / "tests" / "filecheck" / "passes" / f"{snake}.mlir"
    test.write_text(
        f"""// Tests for {name}.
// RUN: df-opt %s -p {name} | filecheck %s

// TODO: write input IR, then CHECK lines for what the pass should produce.
"test.op"() : () -> ()
// CHECK: "test.op"() : () -> ()
""",
        encoding="utf-8",
    )
    insert_before(
        SPEC,
        "### Planned Passes",
        f"### `{name}` - In Progress\n\nTODO: what it does, options, an example.\n\n",
    )
    print(f"created {cls} ({name}) in {module.relative_to(ROOT)}")
    print(f"created {test.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="kind", required=True)
    op = sub.add_parser("op", help="new op, e.g. 'map'")
    op.add_argument("name")
    op.add_argument("--tier", choices=["a", "b"], required=True)
    p = sub.add_parser("pass", help="new pass, e.g. 'df-place'")
    p.add_argument("name")
    args = parser.parse_args()

    if args.kind == "op":
        scaffold_op(args.name, args.tier)
    else:
        scaffold_pass(args.name)

    subprocess.run(["ruff", "check", "--fix", "--quiet", "."], cwd=ROOT, check=False)
    subprocess.run(["ruff", "format", "--quiet", "."], cwd=ROOT, check=False)
    print("next: replace every TODO, then run: make check")


if __name__ == "__main__":
    main()
