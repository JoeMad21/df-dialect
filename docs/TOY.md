# Toy Dialect

A sandbox for demonstrating lowering into real vendor dialects while the df
dialect design (`docs/SPEC.md`) is still being settled. Nothing in the df
dialect depends on it, and it can be deleted without affecting anything else.

## Ops

| Op | Syntax | TTIR | CSL |
|---|---|---|---|
| `toy.matmul` | `%c = toy.matmul %a, %b : tensor<MxNxf32>` | `ttir.matmul` | loop of `@fmacs` |
| `toy.add` | `%c = toy.add %a, %b : tensor<...>` | `ttir.add` | `@fadds` |
| `toy.relu` | `%y = toy.relu %x : tensor<...>` | `ttir.relu` | `@fmaxs` with 0 |

Verifier rules: `toy.matmul` needs rank-2 operands, matching inner dimensions,
an MxN result and one element type. `toy.add` and `toy.relu` need every
operand and the result to have the same type.

## Target Versions

| Target | Source | Version |
|---|---|---|
| Tenstorrent TTIR | github.com/tenstorrent/tt-mlir, `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` | commit `5f396fd` (2026-02-20), the LASSI pin |
| Cerebras CSL-IR | github.com/xdslproject/xdsl: `xdsl/dialects/csl.py`, `csl_wrapper.py`; printer `xdsl/backend/csl/print_csl.py` | `xdsl==0.73.0`, commit `a2544953` |

The CSL dialects, the `linalg-to-csl`, `lower-csl-wrapper` and `memref-to-dsd`
passes, and the CSL printer are xDSL's own, from the pipeline described in
Stawinoga et al., "An MLIR Lowering Pipeline for Stencils at Wafer-Scale",
ASPLOS '26, arXiv:2601.17754.

## Files

| File | Contents |
|---|---|
| `src/df_dialect/toy/__init__.py` | The three toy ops |
| `src/df_dialect/ttir/__init__.py` | Output-only mirror of the three TTIR ops |
| `src/df_dialect/passes/convert_toy_to_ttir.py` | Toy to TTIR |
| `src/df_dialect/passes/convert_toy_to_csl.py` | Toy to a one-PE program in xDSL's CSL dialects |
| `examples/toy_linear.mlir` | `relu(x @ w + b)`, 64x128 inputs |
| `examples/toy_linear_pe.mlir` | The same layer at 16x32, small enough for one PE |
| `scripts/demo_toy.py` | Step-by-step demo of both lowerings |
| `scripts/split_csl.py` | Splits printed CSL into program and layout files |
| `tests/filecheck/toy/` | Round-trip, verifier, TTIR and CSL tests |

## Demo

```sh
uv run python scripts/demo_toy.py
```

## Lowering to TTIR

```sh
uv run df-opt examples/toy_linear.mlir -p convert-toy-to-ttir -o build/toy_linear.ttir.mlir
```

```mlir
%0 = "ttir.matmul"(%x, %w) : (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
%1 = "ttir.add"(%0, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
%2 = "ttir.relu"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
```

At the pinned commit these ops take inputs only. `ttir.matmul` also has
`transpose_a` and `transpose_b`, which default to false.

## Lowering to CSL

The whole program runs on one processing element (PE), so every buffer must
fit in its 48 KB. The pass refuses larger programs:

```
@linear needs 128 KB of buffers; one PE has 48 KB. Use smaller tensors.
```

The pipeline has three stages:

```sh
# Stage 1: toy -> csl_wrapper.module (layout + program) with csl, linalg, memref, scf
uv run df-opt examples/toy_linear_pe.mlir -p convert-toy-to-csl

# Stage 2: xDSL lowers linalg to CSL builtins and splits the wrapper into csl.module ops
uv run df-opt examples/toy_linear_pe.mlir -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper

# Stage 3: xDSL turns buffers into DSDs and prints CSL source
uv run df-opt examples/toy_linear_pe.mlir -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper,memref-to-dsd -t csl -o build/linear.csl.txt
uv run python scripts/split_csl.py build/linear.csl.txt build/csl
```

How the program is built:

| Toy | CSL |
|---|---|
| Function arguments `x`, `w`, `b` | Buffers exported for the host to write with memcpy |
| `toy.matmul` | `C = 0`; for each `i`, `k`: row `i` of C += `x[i,k]` * row `k` of `w`, one `@fmacs` each |
| `toy.add` | `linalg.add`, turned into `@fadds` by xDSL's `linalg-to-csl` |
| `toy.relu` | `linalg.max` with 0, turned into `@fmaxs` |
| Function result | Buffer exported as `out` |
| End of function | `memcpy.unblock_cmd_stream()` returns control to the host |

`pe-memory-kb` and `target` are pass options, for example
`-p 'convert-toy-to-csl{pe-memory-kb=48 target=wse2}'`.

## Next Steps on Vendor Tools

These steps need the vendor toolchains and were not run as part of this repo's
tests.

**tt-mlir** (at commit `5f396fd`, on a machine with a Tenstorrent card):

```sh
ttrt query --save-artifacts                      # writes ttrt-artifacts/system_desc.ttsys
ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=ttrt-artifacts/system_desc.ttsys" \
    build/toy_linear.ttir.mlir -o build/toy_linear.ttnn.mlir
ttmlir-translate --ttnn-to-flatbuffer build/toy_linear.ttnn.mlir -o build/toy_linear.ttnn
ttrt run build/toy_linear.ttnn
```

**Cerebras SDK** (one PE plus the memcpy halo, as in the SDK's single-PE
tutorials):

```sh
cd build/csl
cslc linear_layout.csl --arch=wse3 --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out
```

A host program then writes `x`, `w`, `b` with `memcpy_h2d`, calls `linear`,
and reads `out` with `memcpy_d2h`.
