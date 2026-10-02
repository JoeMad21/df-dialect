# DF Dialect Specification

This is the reference for every type, attribute, op and pass in the `df`
dialect. It is the contract between contributors: if code and this file
disagree, fix one of them in the same pull request.

`tests/unit/test_registry.py` fails if an op or attribute is missing from this
file, so every new construct must get a section here.

## Conventions

- Namespace: every construct is named `df.<name>`.
- Tiers: Tier A ops say what to compute. Tier B ops say where it runs and how
  data moves. See `docs/ARCHITECTURE.md`.
- Semantics: Tier B actors communicate only through FIFO channels with
  blocking receives (the Kahn process network model), so a program's result
  does not depend on timing.
- Status: each section is marked `Implemented` or `Planned`.

## Types and Attributes

### `!df.chan` (ChannelType) - Implemented

A typed FIFO between cores.

```mlir
!df.chan<tensor<64x64xbf16>>              // depth 2 (default, not printed)
!df.chan<tensor<64x64xbf16>, depth = 4>
```

| Parameter | Meaning |
|---|---|
| `element_type` | Type of each value sent on the channel. Must be a type. |
| `depth` | Number of values the FIFO holds. At least 1. Default 2. |

### `#df.grid` (GridAttr) - Implemented

A rectangular grid of cores: `#df.grid<ROWSxCOLS>`, for example
`#df.grid<2x2>`. Both dimensions must be positive.

### `#df.tiles` (TileRangeAttr) - Implemented

The cores an actor is placed on.

```mlir
#df.tiles<all>          // every core of the enclosing grid
#df.tiles<0:2, 1:2>     // rows 0..1, column 1 (half-open ranges)
```

Ranges must be non-empty and start at 0 or above. `df.actor` checks that the
range fits inside the program grid.

### `#df.bcast` (BcastAttr) - Implemented

Broadcast mode of a channel: `row`, `col` or `all`. Written inside
`df.channel` as `bcast(row)`. Generic form: `#df<bcast row>`.

### `#df.split` (SplitAttr) - Implemented

How `df.scatter` cuts a host array and how `df.gather` reassembles it: `rows`,
`cols` or `grid`. Written inside the op as `split(rows)`. Generic form:
`#df<split rows>`.

## Tier A Ops

### `df.contract` - Implemented

Tensor contraction described by an einsum spec.

```mlir
%c = df.contract "mk,kn->mn" %a, %b : tensor<64x64xbf16>
%s = df.contract "bqd,bkd->bqk" %q, %k : tensor<8x16x32xf32>
```

Verifier:
- The spec has exactly one `->`, lowercase letters only, no repeated index
  within one operand, and every output index appears in an input.
- Each operand rank equals its spec length, and a repeated index has the same
  size everywhere.
- The result shape equals the shape the spec produces.
- Both operands and the result share an element type.

### `df.map`, `df.reduce`, `df.stencil` - Planned

Elementwise maps, reductions and neighborhood stencils at Tier A. See
`docs/ROADMAP.md`.

## Tier B Ops

### `df.program` - Implemented

A dataflow program: a grid and everything that runs on it.

```mlir
df.program @matmul attributes {grid = #df.grid<2x2>} {
  ...df.channel, df.actor, df.host...
}
```

- `grid` attribute is required.
- Isolated from above: the body cannot use values defined outside it.
- Verifier: every channel declared in the body is written at least once
  (`df.send` or `df.scatter`) and read at least once (`df.recv` or
  `df.gather`).

### `df.channel` - Implemented

Declares a channel. Must be a direct child of `df.program`.

```mlir
%c = df.channel : !df.chan<tensor<64x64xbf16>>
%a = df.channel bcast(row) : !df.chan<tensor<64x128xbf16>>
```

Without `bcast`, each value goes to one receiving core. `bcast(row)` delivers
each value to every core in a grid row, `bcast(col)` to every core in a grid
column, `bcast(all)` to every core.

### `df.actor` - Implemented

The code each core runs. Must be a direct child of `df.program`.

```mlir
df.actor @tile on #df.tiles<all> ins(%arow, %bcol) outs(%cout) {
  %a = df.recv %arow : tensor<64x128xbf16>
  ...
  df.send %cout, %c : tensor<64x64xbf16>
}
```

- `on` places one copy of the actor on every core in the tile range.
- `ins` and `outs` list the channels the body may read and write.
- Verifier: the tile range fits the program grid; every `df.recv` in the body
  reads a channel in `ins`; every `df.send` writes a channel in `outs`.

### `df.host` - Implemented

The host side of a program. Must be a direct child of `df.program`. Its block
arguments are the host arrays.

```mlir
df.host (%A: memref<128x128xbf16>, %C: memref<128x128xbf16>) {
  df.scatter %A -> %arow split(rows)
  df.gather %cout -> %C split(grid)
}
```

### `df.recv` - Implemented

Waits for the next value on a channel and returns it.

```mlir
%a = df.recv %arow : tensor<64x128xbf16>
```

- Must be nested (at any depth) inside a `df.actor`.
- Verifier: the result type equals the channel element type.

### `df.send` - Implemented

Sends a value on a channel.

```mlir
df.send %cout, %c : tensor<64x64xbf16>
```

- Must be nested (at any depth) inside a `df.actor`.
- Verifier: the value type equals the channel element type.

### `df.scatter` - Implemented

Cuts a host array into pieces and sends them into a channel. Must be a direct
child of `df.host`.

```mlir
df.scatter %A -> %arow split(rows)
```

`split(rows)` sends row block i to grid row i, `split(cols)` sends column
block j to grid column j, `split(grid)` sends one block per core.

### `df.gather` - Implemented

Receives pieces from a channel and assembles them into a host array. Must be
a direct child of `df.host`.

```mlir
df.gather %cout -> %C split(grid)
```

## Passes

### `df-check-memory` - Implemented

Fails if any `df.actor` holds more tensor data than a core's local memory.

```sh
df-opt input.mlir -p df-check-memory               # 48 KB default (one WSE-3 PE)
df-opt input.mlir -p 'df-check-memory{budget-kb=32}'
```

The footprint is the sum of the sizes of every tensor value produced inside
the actor body. For the 2x2 matmul this is A 16 KB + B 16 KB + C 8 KB = 40 KB.

### Planned Passes

`df-partition`, `df-place`, `df-connect`, `df-bufferize` (Tier A to Tier B),
`convert-df-to-csl`, `convert-df-to-ttkernel`, `convert-df-to-linalg` (CPU
reference). See `docs/ROADMAP.md`.
