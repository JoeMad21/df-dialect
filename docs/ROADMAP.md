# Roadmap

Each item is written so it can be copied into a GitHub issue. Labels:
`good first issue`, `intermediate`, `advanced`, plus `tier-a`, `tier-b` or `pass`.

## Milestones

| Milestone | Exit criterion |
|---|---|
| M1 Core dialect | Tier A and Tier B ops for matmul, elementwise, reduction and stencil kernels verify and round-trip. **Matmul done.** |
| M2 CPU reference | Every Tier A op lowers to linalg and runs on CPU; outputs match NumPy. |
| M3 Tier A to Tier B | Partition, place, connect and bufferize passes turn the Tier A matmul into the walkthrough program. |
| M4 Cerebras | Tier B lowers to csl-ir; generated CSL passes the Cerebras SDK simulator. |
| M5 Tenstorrent and Furiosa | Same Tier A input reaches TTIR and TCL. |

## Good First Issues

**G1. Host arguments must be memrefs** (`tier-b`)
`df.host` block arguments must all be `memref` types.
Accept: error `df.host argument %X must be a memref, got T`; one verify test.

**G2. Unique actor names** (`tier-b`)
Two `df.actor` ops in one `df.program` must not share a name.
Accept: error `duplicate actor @NAME in @PROGRAM`; one verify test.

**G3. Scatter and gather element types** (`tier-b`)
`df.scatter` and `df.gather`: the memref element type must equal the channel
tensor element type. Accept: error message and one verify test per op.

**G4. `df.map`** (`tier-a`)
Elementwise op with a named function: `%y = df.map "add" %a, %b : tensor<4xf32>`.
Functions: add, sub, mul, max, min. Operands and result share a type.
Accept: scaffold used, round-trip test, error tests for an unknown function
and for a type mismatch, SPEC section.

**G5. A second example** (`tier-b`)
Add `examples/vector_add_1x4.mlir`: a 1x4 grid adding two 256-element
vectors, each core handling 64 elements. Accept: RUN line in
`tests/filecheck/examples/examples.mlir`.

## Intermediate

**I1. Scatter and gather shape rule** (`tier-b`)
`split(rows)` on a 128x128 memref with a 2-row grid must feed a 64x128 channel
element; `split(cols)` and `split(grid)` likewise. Accept: shape computed from
the program grid, one error test per mode.

**I2. `df.reduce`** (`tier-a`)
`%s = df.reduce "add" %x dims [1] : tensor<64x128xf32> -> tensor<64xf32>`.
Accept: verifier checks dims and result shape; tests; SPEC.

**I3. `df.stencil`** (`tier-a`)
Neighborhood computation with explicit offsets. Write the SPEC proposal first
and get it reviewed before coding.

**I4. Memory pass counts channel buffers** (`pass`)
Option `count-channels=true` adds `depth x element size` for every channel an
actor uses. Accept: tests for both settings.

**I5. `convert-df-to-linalg`** (`pass`, M2)
Lower `df.contract` to `linalg.generic` (or `linalg.matmul` for the matmul
spec). Accept: FileCheck test of the output; output passes `mlir-opt` if available.

**I6. Tier B interpreter** (`tier-b`)
Execute Tier B programs on CPU with xDSL's interpreter framework, using
Kahn process network semantics. Accept: the walkthrough matmul matches NumPy.

## Advanced

**A1. Partition, place, connect, bufferize** (`pass`, M3)
Four passes from Tier A to Tier B, driven by `#df.grid` and a per-core memory
budget. Accept: Tier A matmul becomes the walkthrough program.

**A2. `convert-df-to-csl`** (`pass`, M4)
Lower Tier B to xDSL's csl-wrapper and csl-ir; print CSL. Start from the
stencil pipeline in xDSL. Accept: CSL passes the Cerebras SDK simulator.

**A3. Tenstorrent emission** (`pass`, M5)
Tier A to TTIR text accepted by tt-mlir's TTNN pipeline; Tier B to
TTKernel/TTMetal text. Accept: tt-mlir parses the output.

**A4. Furiosa TCL emitter** (`pass`, M5)
Thin `tcl` dialect plus a printer that emits `@tcl.kernel` source from Tier A.
Accept: generated source compiles with Furiosa SDK 2026.3.0.

**A5. LASSI-DF hook**
A small Python API, `df_dialect.verify_text(src) -> list[str]`, returning
verifier and pass errors for a string of IR, for use as the LASSI-DF compile
signal. Accept: unit tests with valid and invalid inputs.
