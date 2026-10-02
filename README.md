# DF-Dialect

An MLIR dialect, `df`, for dataflow accelerators: Cerebras WSE, Tenstorrent,
AMD AIE and Furiosa RNGD. It sits between the upstream structured dialects
(linalg, tensor, memref) and each vendor's device dialects, so a kernel is
written once and lowered to every target. Built on [xDSL](https://xdsl.dev),
so it is pure Python: no LLVM build, no sudo.

```mlir
df.program @matmul attributes {grid = #df.grid<2x2>} {
  %arow = df.channel bcast(row) : !df.chan<tensor<64x128xbf16>>
  %bcol = df.channel bcast(col) : !df.chan<tensor<128x64xbf16>>
  %cout = df.channel : !df.chan<tensor<64x64xbf16>>
  df.actor @tile on #df.tiles<all> ins(%arow, %bcol) outs(%cout) {
    %a = df.recv %arow : tensor<64x128xbf16>
    %b = df.recv %bcol : tensor<128x64xbf16>
    %c = df.contract "mk,kn->mn" %a, %b : tensor<64x64xbf16>
    df.send %cout, %c : tensor<64x64xbf16>
  }
  df.host (%A: memref<128x128xbf16>, %B: memref<128x128xbf16>, %C: memref<128x128xbf16>) {
    df.scatter %A -> %arow split(rows)
    df.scatter %B -> %bcol split(cols)
    df.gather %cout -> %C split(grid)
  }
}
```

## Quickstart

Linux, macOS or WSL. On native Windows (PowerShell), follow First-Time Setup
in `CONTRIBUTING.md`.

```sh
# 1. Install uv once (no sudo needed): https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code and install everything into .venv
git clone https://github.com/JoeMad21/df-dialect.git
cd df-dialect
uv sync

# 3. Run every check (same as CI)
uv run python scripts/check.py

# 4. Try the dialect
uv run df-opt examples/matmul_2x2.mlir
uv run df-opt examples/matmul_2x2.mlir -p 'df-check-memory{budget-kb=32}'
```

## Status

| Area | State |
|---|---|
| Types and attributes: `!df.chan`, `#df.grid`, `#df.tiles`, `bcast`, `split` | Implemented |
| Tier A: `df.contract` | Implemented |
| Tier B: `df.program`, `df.channel`, `df.actor`, `df.host`, `df.recv`, `df.send`, `df.scatter`, `df.gather` | Implemented |
| Verifiers: type mismatch, unread and unwritten channels, off-grid placement, ins/outs use, contract shapes | Implemented |
| Pass: `df-check-memory` | Implemented |
| Tier A to Tier B passes, CPU reference, Cerebras, Tenstorrent, Furiosa lowerings | Planned (`docs/ROADMAP.md`) |

## Documentation

| File | Read it when |
|---|---|
| `CONTRIBUTING.md` | You are new, or about to change code: setup, rules, style, pitfalls |
| `docs/SPEC.md` | You need the exact syntax and rules of an op |
| `docs/ARCHITECTURE.md` | You want the design: tiers, lowering plan, LASSI-DF link |
| `docs/playbooks/ADD_OP.md` | You are adding an op |
| `docs/playbooks/ADD_PASS.md` | You are adding a pass |
| `docs/TESTING.md` | You are writing or fixing tests |
| `docs/GLOSSARY.md` | A term is unfamiliar |
| `docs/ROADMAP.md` | You are looking for something to work on |
| `docs/decisions/` | You want to know why a choice was made |

## License

No license has been chosen yet. Until one is added, all rights are reserved.
