# Testing

Two kinds of tests, both run by `uv run python scripts/check.py`.

| Kind | Location | Runner | Use for |
|---|---|---|---|
| FileCheck | `tests/filecheck/**/*.mlir` | `lit` | Syntax, verifier errors, pass output |
| Unit | `tests/unit/*.py` | `pytest` | Plain-Python helpers, Python builders, the coverage gate |

## How a FileCheck Test Works

Each `.mlir` file has `RUN:` lines. lit runs them as shell commands with `%s`
replaced by the file path. The output is piped into `filecheck`, which matches
it against the `CHECK:` lines in the same file.

```mlir
// RUN: df-opt %s | df-opt | filecheck %s
%c = df.channel ...
// CHECK: %c = df.channel ...
```

| Directive | Meaning |
|---|---|
| `CHECK: text` | `text` appears somewhere after the previous match |
| `CHECK-NEXT: text` | `text` is on the very next line |
| `CHECK-NOT: text` | `text` does not appear between the surrounding matches |
| `CHECK-SAME: text` | `text` is on the same line as the previous match |
| `--check-prefix=FOO` | use `FOO:` lines instead of `CHECK:` (one file, several RUN lines) |

## The Three Test Patterns

**Round trip** (`tests/filecheck/dialect/`): parse, print, parse again, and
also go through the generic form. Proves `parse` and `print` agree.

```mlir
// RUN: df-opt %s | df-opt | filecheck %s
// RUN: df-opt %s --print-op-generic | df-opt | filecheck %s
```

**Verifier errors** (`tests/filecheck/verify/`): one case per rule, separated
by `// -----`, each followed by the exact message.

```mlir
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s
...bad IR...
// CHECK: channel %bcol carries tensor<128x64xbf16>
// -----
...next bad IR...
```

**Parse errors**: same shape, with `--parsing-diagnostics` instead of
`--verify-diagnostics`.

## Running Tests

```sh
uv run lit -v tests/filecheck                         # all FileCheck tests
uv run lit -v tests/filecheck/verify/channels.mlir    # one file
uv run pytest -q                                      # all unit tests
uv run pytest -q -k einsum                            # tests matching a name
uv run df-opt tests/filecheck/verify/channels.mlir --split-input-file --verify-diagnostics
                                                      # see the raw output a test checks
```

## When a FileCheck Test Fails

1. Run the RUN command by hand (replace `%s` with the path) and read the output.
2. If the output is right and the CHECK line is stale, update the CHECK line
   and say so in your PR.
3. If the output is wrong, fix the code. Never delete a CHECK line to pass.

## The Coverage Gate

`tests/unit/test_registry.py` fails when an op or attribute lacks a docstring,
a `docs/SPEC.md` section, or any FileCheck test that uses it. The failure
message names what is missing.
