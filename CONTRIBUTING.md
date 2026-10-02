# Contributing to DF-Dialect

Welcome. You do not need prior MLIR or compiler experience. If you can write
Python and run commands in a terminal, you can contribute. When a word is
unfamiliar, check `docs/GLOSSARY.md`.

## First-Time Setup

You need git and [uv](https://docs.astral.sh/uv/). uv installs Python and every
dependency for you; it does not need sudo.

**Linux, macOS, WSL**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # once
git clone https://github.com/JoeMad21/df-dialect.git
cd df-dialect
uv sync
uv run python scripts/check.py                     # should end with "All checks passed."
```

**Windows**

Use one of these, in order of preference:
1. WSL (Ubuntu), then follow the Linux steps inside it.
2. VS Code with the Dev Containers extension: open the folder and choose
   "Reopen in Container". Setup runs automatically.

If `scripts/check.py` does not pass on a fresh clone, open an issue; that is a
bug in the repo, not in your setup.

## Commands

```sh
uv sync                                              # install (first time and after pulls)
uv run python scripts/check.py                       # the gate: format, lint, types, unit, filecheck
uv run python scripts/check.py --fast                # same without pyright
uv run pytest -q tests/unit/test_einsum.py           # one unit test file
uv run lit -v tests/filecheck/verify/channels.mlir   # one FileCheck test
uv run df-opt examples/matmul_2x2.mlir               # parse, verify, print
uv run python scripts/scaffold.py op NAME --tier a   # start a new op (tier a or b)
uv run python scripts/scaffold.py pass df-NAME       # start a new pass
```

`make check`, `make fmt` and the other Makefile targets wrap the same commands.

## Repository Map

```
src/df_dialect/df/attributes.py   types and attributes (!df.chan, #df.grid, #df.tiles, enums)
src/df_dialect/df/tier_a.py       Tier A ops (df.contract)
src/df_dialect/df/tier_b.py       Tier B ops (df.program, df.channel, df.actor, ...)
src/df_dialect/df/einsum.py       plain-Python helper, unit tested
src/df_dialect/df/__init__.py     OPS and ATTRIBUTES lists: the only registration point
src/df_dialect/passes/            one module per pass; ALL_PASSES in __init__.py registers them
src/df_dialect/universe.py        plugs df into xdsl-opt/df-opt via an entry point
tests/filecheck/dialect/          round-trip tests (parse -> print -> parse)
tests/filecheck/verify/           error tests (one case per verifier rule)
tests/filecheck/passes/           pass tests
tests/filecheck/examples/         keeps examples/ valid
tests/unit/                       pytest; test_registry.py is the coverage gate
docs/                             SPEC, ARCHITECTURE, playbooks, TESTING, ROADMAP, GLOSSARY
```

## Your First Contribution

1. Pick an issue labeled `good first issue` in `docs/ROADMAP.md` or on GitHub
   and comment that you are taking it.
2. Make a branch: `git switch -c yourname/short-topic`.
3. Adding an op or pass? Start with the scaffold, which creates every required
   file and already passes the checks. Then follow
   `docs/playbooks/ADD_OP.md` or `docs/playbooks/ADD_PASS.md`.
4. Write the test first, watch it fail, then make it pass.
5. Run `uv run python scripts/check.py` until it passes.
6. Commit, push your branch, and open a pull request. Fill in the checklist.
7. Address review comments by pushing more commits to the same branch.

## Rules

1. `uv run python scripts/check.py` must pass before a pull request is ready.
2. Every new or changed op, attribute, type or pass ships with, in the same
   pull request: a docstring, a section in `docs/SPEC.md`, a round-trip
   FileCheck test, and one error test per verifier rule.
   `tests/unit/test_registry.py` enforces part of this; do not edit that test
   to make it pass.
3. Start new ops and passes with `scripts/scaffold.py`, then replace every
   `TODO` before asking for review.
4. Never weaken, skip or delete a test to get a green run. If a test is wrong,
   explain why in the pull request.
5. Do not change the syntax or error text of an existing op without updating
   `docs/SPEC.md`, `examples/` and every affected test in the same pull request.
6. Do not add or upgrade dependencies without asking the maintainer. The xDSL
   version is pinned on purpose.
7. One logical change per branch and pull request. Keep diffs small.
8. ASCII only in source, tests and docs. Unix line endings (git handles this
   through `.gitattributes`).
9. If `docs/SPEC.md` does not say what an op should do, ask in the issue or
   propose the behavior in your pull request. Do not invent semantics silently.
10. You must be able to explain every line you submit, however it was written.

## Style

- Follow the patterns in `tier_b.py`: one class per op, a docstring with a
  `Syntax:` example, `__init__` for building from Python, `parse` and `print`
  for readable syntax, `verify_` for checks.
- Error messages are lowercase, name the op or value (`channel %bcol carries
  tensor<128x64xbf16>`), and say what was expected. Tests check the exact text.
- Put complex logic in small plain-Python helpers (like `einsum.py`) that can
  be unit tested without building IR.
- Type hints everywhere; pyright runs in CI. Formatting is automatic:
  `uv run ruff format .`

## xDSL Pitfalls

- Attribute parameters are declared as annotated class fields
  (`rows: IntAttr`). Any other class-level name on a `ParametrizedAttribute`
  is treated as a parameter and fails. Put constants at module level.
- `prop_def` is for inherent data of the op (printed inside the op syntax).
  `attr_def` is for named attributes shown in `attributes {...}`.
- An empty `{ }` parses as a region with zero blocks. Single-block ops call
  `_ensure_block(...)` in `parse`; copy that.
- `parser.parse_operand()` resolves the value immediately, so the operand type
  is known while parsing; custom syntax does not need to repeat it.
- Pass options map dashes to underscores: `-p 'df-check-memory{budget-kb=32}'`
  sets the field `budget_kb`.
- Passes report failures with `PassFailedException`; verifiers with
  `VerifyException`. Both show up under `--verify-diagnostics`.
- Error tests use `--split-input-file --verify-diagnostics` and separate cases
  with `// -----`. Parse errors need `--parsing-diagnostics` instead.

## Definition of Done

- [ ] `scripts/check.py` passes.
- [ ] Docstring, SPEC section, round-trip test and error tests are present.
- [ ] No `TODO` lines left in changed code.
- [ ] `examples/` still valid; new examples have a RUN line in
      `tests/filecheck/examples/examples.mlir`.
- [ ] The pull request description lists what changed, what was tested and
      any open questions.

## Getting Help

Ask in your pull request or issue, even if the code is unfinished. Mark the
pull request as a draft and describe where you are stuck. Questions are part
of the work.

## Maintainer Setup (One Time)

- Make the repository public so branch protection is available on a free plan.
- Branch protection on `main`: require a pull request, one approving review,
  review from code owners, and the status checks `check (3.10)` and
  `check (3.12)` to pass (they appear in the list after CI has run once).
- Create the labels used in `docs/ROADMAP.md`: `good first issue`,
  `intermediate`, `advanced`, `tier-a`, `tier-b`, `pass`, `new-op`, `bug`.
- Open the roadmap items as issues so contributors can claim them.
