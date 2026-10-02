# Playbook: Add a Pass

## Pass or Verifier?

| Put it in a verifier (`verify_`) when | Put it in a pass when |
|---|---|
| The rule must hold for every valid program | The rule depends on a target or an option (a memory budget) |
| It looks at one op and its neighbors | It needs a whole-program view or an analysis |
| It never changes the IR | It rewrites the IR (lowering, placement) |

`df-check-memory` is a pass because the budget depends on the target.
"Channel element type matches df.recv" is a verifier because it is always true.

## 1. Scaffold

```sh
uv run python scripts/scaffold.py pass df-place
```

This creates `src/df_dialect/passes/df_place.py`, registers it in
`ALL_PASSES`, creates `tests/filecheck/passes/df_place.mlir` and adds a SPEC
section. The checks pass immediately.

## 2. Write the Test First

```mlir
// RUN: df-opt %s -p df-place | filecheck %s

// input IR here
// CHECK: what the output must contain
// CHECK-NEXT: the line right after it
// CHECK-NOT: something that must be gone
```

For failures, add a second RUN line with `--split-input-file --verify-diagnostics`
and check the message.

## 3. Options

Options are dataclass fields. Dashes on the command line map to underscores:

```python
@dataclass(frozen=True)
class CheckMemoryPass(ModulePass):
    name = "df-check-memory"
    budget_kb: int = 48
```

```sh
df-opt in.mlir -p 'df-check-memory{budget-kb=32}'
```

Supported option types: int, float, bool, str, tuples of those, and `X | None`.

## 4. Two Shapes of Pass

**Analysis or check** (reads the IR, may fail): walk and raise.

```python
def apply(self, ctx: Context, op: ModuleOp) -> None:
    for actor in op.walk():
        if isinstance(actor, ActorOp) and too_big(actor):
            raise PassFailedException(f"actor @{actor.sym_name.data} ...")
```

**Rewrite** (changes the IR): one `RewritePattern` per op kind you transform.

```python
class LowerContract(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ContractOp, rewriter: PatternRewriter) -> None:
        new_op = ...  # build the replacement with its __init__
        rewriter.replace_op(op, new_op)  # results of op are rewired to new_op


def apply(self, ctx: Context, op: ModuleOp) -> None:
    PatternRewriteWalker(LowerContract()).rewrite_module(op)
```

The walker keeps applying patterns until nothing changes, so a pattern must
not match its own output.

## 5. Document and Check

Fill in the SPEC section (what it does, options, an example command), mark it
`Implemented`, and run `uv run python scripts/check.py`.

Reference implementation: `src/df_dialect/passes/check_memory.py` and
`tests/filecheck/passes/check_memory.mlir`.
