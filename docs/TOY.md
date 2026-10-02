# Toy Dialect

A sandbox for demonstrating lowering into Tenstorrent's TTIR while the df
dialect design (`docs/SPEC.md`) is still being settled. Nothing in the df
dialect depends on it, and it can be deleted without affecting anything else.

## Ops

| Op | Syntax | Lowers to |
|---|---|---|
| `toy.matmul` | `%c = toy.matmul %a, %b : tensor<MxNxT>` | `ttir.matmul` |
| `toy.add` | `%c = toy.add %a, %b : tensor<...>` | `ttir.add` |
| `toy.relu` | `%y = toy.relu %x : tensor<...>` | `ttir.relu` |

Verifier rules: `toy.matmul` needs rank-2 operands, matching inner dimensions,
an MxN result and one element type. `toy.add` and `toy.relu` need every
operand and the result to have the same type.

## Files

| File | Contents |
|---|---|
| `src/df_dialect/toy/__init__.py` | The three toy ops |
| `src/df_dialect/ttir/__init__.py` | Minimal output-only mirror of the TTIR ops the lowering emits |
| `src/df_dialect/passes/convert_toy_to_ttir.py` | The lowering pass |
| `examples/toy_linear.mlir` | `relu(x @ w + b)` |
| `tests/filecheck/toy/` | Round-trip, verifier and lowering tests |

## Lowering to TTIR

```sh
uv run df-opt examples/toy_linear.mlir -p convert-toy-to-ttir -o toy_linear.ttir.mlir
```

Output:

```mlir
%0 = "ttir.matmul"(%x, %w) : (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
%1 = "ttir.add"(%0, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
%2 = "ttir.relu"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
```

The TTIR ops are printed in generic form with no output operand, matching
tt-mlir at commit a6bd5d2 (2026-09-16). If a different tt-mlir version rejects
the module wrapper or `func.func` syntax, emit everything in generic form with
`--print-op-generic`.

## Continuing in tt-mlir

On a machine with tt-mlir built and a system descriptor for the device:

```sh
ttmlir-opt --ttir-to-ttnn-runtime-pipeline="system-desc-path=$DESC" toy_linear.ttir.mlir -o toy_linear.ttnn.mlir
ttmlir-translate --ttnn-to-flatbuffer toy_linear.ttnn.mlir -o toy_linear.ttnn
ttrt run toy_linear.ttnn
```

Pipeline names and options follow tt-mlir's documentation for that version.
