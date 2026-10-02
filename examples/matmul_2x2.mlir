// The 128x128 bf16 matmul on a 2x2 grid from the Syntax Walkthrough slide.
// Try it:  uv run df-opt examples/matmul_2x2.mlir

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
