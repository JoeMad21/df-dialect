// df-check-memory: per-core footprint against a budget (default 48 KB).
// RUN: df-opt %s --split-input-file -p df-check-memory --verify-diagnostics | filecheck %s
// RUN: df-opt %s --split-input-file -p 'df-check-memory{budget-kb=32}' --verify-diagnostics | filecheck %s --check-prefix=SMALL

// 2x2 grid: A 16 KB + B 16 KB + C 8 KB = 40 KB per core. Fits 48 KB, not 32 KB.
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
// CHECK: df.program @matmul
// SMALL: actor @tile holds 40 KB; budget is 32 KB per core

// -----

// 1x1 grid: one core holds all of A, B and C: 3 x 32 KB = 96 KB.
df.program @matmul attributes {grid = #df.grid<1x1>} {
  %ain = df.channel : !df.chan<tensor<128x128xbf16>>
  %bin = df.channel : !df.chan<tensor<128x128xbf16>>
  %cout = df.channel : !df.chan<tensor<128x128xbf16>>
  df.actor @tile on #df.tiles<all> ins(%ain, %bin) outs(%cout) {
    %a = df.recv %ain : tensor<128x128xbf16>
    %b = df.recv %bin : tensor<128x128xbf16>
    %c = df.contract "mk,kn->mn" %a, %b : tensor<128x128xbf16>
    df.send %cout, %c : tensor<128x128xbf16>
  }
  df.host (%A: memref<128x128xbf16>, %B: memref<128x128xbf16>, %C: memref<128x128xbf16>) {
    df.scatter %A -> %ain split(grid)
    df.scatter %B -> %bin split(grid)
    df.gather %cout -> %C split(grid)
  }
}
// CHECK: actor @tile holds 96 KB; budget is 48 KB per core
// SMALL: actor @tile holds 96 KB; budget is 32 KB per core
