# 0001: Build the Dialect on xDSL

Status: Accepted

## Context

The dialect changes weekly, contributors include undergraduates, and the
alpha01 server has no sudo. The Cerebras path should reuse existing CSL
dialects, and the output must interoperate with C++ MLIR tools (tt-mlir,
mlir-aie).

## Options

| Option | For | Against |
|---|---|---|
| xDSL (Python) | `uv sync` installs everything; Python skills suffice; ships csl, csl-stencil and csl-wrapper dialects; external dialects plug into `xdsl-opt` through an entry point; MLIR-compatible text | Slower than C++; reaches C++ tools through text, not in-process |
| C++ MLIR with TableGen | Fastest; same language as tt-mlir and mlir-aie; path to upstream | LLVM build takes hours and tens of GB; C++ and TableGen learning curve; hard for undergrads |
| MLIR Python bindings with IRDL dynamic dialects | Real MLIR underneath | Needs an MLIR build with bindings; limited custom syntax and verifiers in Python |

## Decision

Use xDSL, pinned to an exact version (`xdsl==0.71.0`). Upgrade in a dedicated
pull request that runs the full test suite.

## Consequences

- New contributors are productive on day one.
- Interop with C++ tools happens through MLIR text; use `--print-op-generic`
  when a tool does not know `df` syntax.
- Revisit when any of these become true: compile time matters at scale,
  in-process linking with tt-mlir or mlir-aie is required, or the dialect is
  proposed upstream. The port target is C++ MLIR with TableGen; tests and
  examples carry over unchanged.
