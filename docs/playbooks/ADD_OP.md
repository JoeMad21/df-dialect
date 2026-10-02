# Playbook: Add an Op

Worked through for a hypothetical Tier A op `df.negate` that negates every
element of a float tensor: `%y = df.negate %x : tensor<4xf32>`.

## 1. Agree on the Op

Before code, write down (in the issue or PR): the syntax, what it means, which
tier, and each verifier rule with its error message. If `docs/SPEC.md` already
has a `Planned` section for the op, use it.

## 2. Scaffold

```sh
uv run python scripts/scaffold.py op negate --tier a
```

This adds a stub `NegateOp` to `tier_a.py`, registers it in `OPS`, creates
`tests/filecheck/dialect/negate.mlir` and adds a SPEC section. The checks
already pass. Your job is to replace every TODO.

## 3. Write the Tests First

Round trip (`tests/filecheck/dialect/negate.mlir`):

```mlir
// RUN: df-opt %s | df-opt | filecheck %s
// RUN: df-opt %s --print-op-generic | df-opt | filecheck %s

%x = "test.op"() : () -> tensor<4xf32>
%y = df.negate %x : tensor<4xf32>
// CHECK: %y = df.negate %x : tensor<4xf32>
```

`"test.op"` is a placeholder op from xDSL's test dialect; use it to create
input values without building a whole program.

Errors (add to `tests/filecheck/verify/`, one case per rule):

```mlir
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s

%x = "test.op"() : () -> tensor<4xi32>
%y = df.negate %x : tensor<4xi32>
// CHECK: df.negate needs a floating-point tensor
```

Run them and confirm they fail: `uv run lit -v tests/filecheck/dialect/negate.mlir`.

## 4. Declare the Fields

| You need | Write | Access as |
|---|---|---|
| One input value | `x = operand_def(TensorType)` | `self.x` |
| Any number of inputs | `xs = var_operand_def(ChannelType)` | `self.xs` |
| One output | `result = result_def(TensorType)` | `self.result` |
| Inherent data (printed in op syntax) | `factor = prop_def(FloatAttr)` | `self.factor` |
| Optional inherent data | `bcast = opt_prop_def(BcastAttr)` | `self.bcast` or None |
| Named attribute (`attributes {...}`) | `grid = attr_def(GridAttr)` | `self.grid` |
| A body | `body = region_def("single_block")` | `self.body` |
| Two variable operand lists | add `irdl_options = (AttrSizedOperandSegments(as_property=True),)` | |
| No terminator in the body | `traits = traits_def(NoTerminator())` | |
| Must sit directly inside X | `traits = traits_def(HasParent(XOp))` | |
| No side effects | `traits = traits_def(Pure())` | |

## 5. Add Readable Syntax

Copy the `parse` and `print` pair from the closest existing op and adapt it:

| Your syntax looks like | Copy from |
|---|---|
| `%r = df.x %a : T` | `RecvOp` |
| `df.x %a, %b : T` | `SendOp` |
| `%r = df.x "str" %a, %b : T` | `ContractOp` |
| `%r = df.x keyword(value) : T` | `ChannelOp` |
| `df.x %a -> %b keyword(value)` | `ScatterOp` |
| `df.x @name ... { body }` | `ActorOp` |

Rules: `print` must produce exactly what `parse` accepts, and both must handle
`attr-dict` so extra attributes survive a round trip. The round-trip test
proves this.

## 6. Write the Verifier

```python
def verify_(self) -> None:
    t = self.x.type
    assert isinstance(t, TensorType)  # guaranteed by operand_def; tells pyright the type
    if not isinstance(t.element_type, AnyFloat):  # AnyFloat from xdsl.dialects.builtin
        raise VerifyException("df.negate needs a floating-point tensor")
```

Complex logic goes in a plain-Python helper with its own unit test (see
`einsum.py` and `tests/unit/test_einsum.py`).

## 7. Document and Check

Fill in the SPEC section (syntax, operands, results, verifier rules), change
its status to `Implemented`, and run:

```sh
uv run python scripts/check.py
```

## 8. Optional: Python Builder Test

If lowering passes will create the op, add a case to
`tests/unit/test_builders.py` that builds it with its `__init__` and verifies.
