# Glossary

**MLIR**: A compiler framework from the LLVM project for building compilers out
of reusable layers. Programs are written in a text format called IR.

**IR (intermediate representation)**: The program as the compiler sees it, in
between source code and machine code.

**Dialect**: A named set of ops, types and attributes. `arith`, `linalg` and
`df` are dialects. One program can mix several.

**Op (operation)**: One instruction in the IR, such as `df.recv`. Ops take
operands, produce results, and can hold attributes and regions.

**SSA value**: A named value such as `%a`, defined exactly once. Operands and
results are SSA values.

**Type**: What kind of value something is: `tensor<64x64xbf16>`, `!df.chan<...>`.

**Attribute**: Constant data attached to an op: `#df.grid<2x2>`, a string, a number.

**Property**: An attribute that is part of an op's definition and printed in
its own syntax (for example `bcast(row)` on `df.channel`).

**Region and block**: A region is a body in braces `{ ... }`. It holds blocks,
and a block holds a list of ops. `df.actor` has one region with one block.

**Verifier**: Code that checks an op is well formed (`verify_`) and reports an
error if not.

**Round trip**: Parse text, print it, parse it again, and get the same result.
Proves the parser and printer agree.

**Generic form**: The fallback syntax every op has:
`"df.recv"(%arow) : (...) -> ...`. Printed with `--print-op-generic`.

**Pass**: A program transformation or check that runs over the IR, selected
with `-p name`.

**Lowering**: A pass that rewrites ops into lower-level ops, moving toward
hardware: Tier A to Tier B, Tier B to CSL.

**lit and FileCheck**: Test tools from LLVM. lit runs the `RUN:` lines in a
test file; FileCheck compares output with the `CHECK:` lines.

**xDSL**: A Python framework compatible with MLIR. This project uses it to
define the `df` dialect without building LLVM.

**IRDL**: The way xDSL (and MLIR) describes ops and attributes: the
`operand_def`, `result_def`, `prop_def` declarations.

**Tier A**: `df` ops that say what to compute, on tensors, with no cores.

**Tier B**: `df` ops that say where work runs and how data moves.

**Grid and tile**: The grid is the rectangle of cores (`#df.grid<2x2>`). A
tile range picks some of them (`#df.tiles<0:2, 0:1>`).

**Actor**: The code one core runs, fired by data arriving on its channels.

**Channel**: A typed FIFO queue between cores. `bcast(row)` sends one value to
every core in a grid row.

**Kahn process network**: The model behind Tier B: processes that talk only
through FIFOs with blocking reads always compute the same result, whatever
the timing.

**Host**: The CPU side that loads input arrays into the grid and reads results.
