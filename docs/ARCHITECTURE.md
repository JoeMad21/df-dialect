# Architecture

## Purpose

Every dataflow accelerator ships its own compiler stack and language: CSL for
Cerebras, TT-Metalium for Tenstorrent, IRON for AMD AIE, TCL for Furiosa. With
N source languages and M devices that is N x M ports. DF-Dialect adds one layer
in the middle so each frontend lowers into `df` once and each device is reached
by one lowering out of it: N + M paths.

## Position in the Stack

```
SOURCE        CUDA   C/C++   Fortran   PyTorch/JAX   stencil DSLs
FRONTEND      cir (ClangIR)  fir (Flang)  stablehlo/tosa  torch  stencil
UPSTREAM      linalg  tensor  memref  scf  affine  arith
NEW           df   (Tier A: what to compute | Tier B: where it runs and how data moves)
DEVICE        csl-stencil, csl-wrapper, csl-ir | ttir, ttnn, d2m, ttkernel, ttmetal |
              aie, aiex, aievec | tcl (planned) | gpu, rocdl, llvm (reference)
TOOLCHAIN     Cerebras SDK | TT-Metalium | xclbin | Furiosa EDF | CPU, MI300X
```

Upstream dialects stay the entry point and device dialects stay vendor-owned.
Only the middle layer is new.

## Two Tiers

Devices differ in how much of the mapping they let a program control.

| Entry point | Who places work, moves data, sizes buffers | Enters at |
|---|---|---|
| Cerebras CSL | Program | Tier B |
| Tenstorrent TT-Metalium kernels | Program | Tier B |
| AMD AIE (IRON) | Program | Tier B |
| Tenstorrent TTNN ops | Toolchain | Tier A |
| Furiosa TCL kernels | Toolchain | Tier A |

- **Tier A** (`tier_a.py`): value-semantics ops on tensors. No cores, no
  channels. Example: `df.contract`.
- **Tier B** (`tier_b.py`): an explicit grid, typed channels, actors placed on
  cores, and a host section. Example: `df.program` with `df.actor`.
- Tier A lowers to Tier B through four planned passes: partition, place,
  connect, bufferize.

## Semantics

Tier B actors communicate only through FIFO channels, and `df.recv` blocks
until a value arrives. This is the Kahn process network model: a completed
program produces the same result regardless of timing. Two consequences:
a CPU interpreter can reproduce device output, and verification can reason
about channels without modeling schedules. Bounded channel depth can cause
deadlock but never changes a completed result.

## Package Layout

```
src/df_dialect/
  df/attributes.py   types and attributes
  df/tier_a.py       Tier A ops
  df/tier_b.py       Tier B ops
  df/einsum.py       plain-Python helpers
  df/__init__.py     OPS, ATTRIBUTES, the Df dialect object
  passes/            one module per pass, registered in passes/__init__.py
  universe.py        xDSL plugin entry point (pyproject: xdsl.universe)
```

The `xdsl.universe` entry point means any installed copy of this package adds
`df` and its passes to `xdsl-opt`. `df-opt` is the same tool under a project
name.

## Lowering Plan

| Target | Path | Notes |
|---|---|---|
| CPU reference | Tier A -> linalg -> llvm | Numeric oracle for every other target |
| Cerebras WSE | Tier B -> csl-wrapper, csl-stencil -> csl-ir -> CSL | Reuses the csl dialects that ship in xDSL |
| Tenstorrent (op level) | Tier A -> ttir text -> tt-mlir TTNN pipeline | Stable route |
| Tenstorrent (kernel level) | Tier B -> d2m/ttkernel/ttmetal text -> tt-mlir | For custom data movement |
| AMD AIE | Tier B -> aie, aiex | ObjectFifo maps to df.channel |
| Furiosa RNGD | Tier A -> tcl emission dialect -> TCL source | No public MLIR dialect; print source like csl-ir |

C++ MLIR tools (tt-mlir, mlir-aie, mlir-opt) are reached through MLIR text.
Print generic form with `--print-op-generic` when a C++ tool does not know an
op's custom syntax.

## Relationship to LASSI-DF

DF-Dialect is a separate project. The LASSI-DF MLIR hub depends on it as a
package (`uv add git+https://github.com/JoeMad21/df-dialect`) and uses it in
two ways: `df` is the output language of the transpilation model, and the
verifiers and `df-check-memory` give a fast, graded compile signal before any
vendor compiler runs.

## Future Port to C++

xDSL is the right tool while the dialect is changing weekly and contributors
are learning. If performance, in-process linking with tt-mlir or mlir-aie, or
upstreaming becomes the priority, the proven dialect moves to C++ MLIR with
TableGen. The text format is MLIR-compatible, so tests and examples carry over.
See `docs/decisions/0001-use-xdsl.md`.
