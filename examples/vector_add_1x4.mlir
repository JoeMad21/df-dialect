// Two 256-element vectors on a 1x4 grid; each core adds 64 elements.
// Try it: uv run df-opt examples/vector_add_1x4.mlir

df.program @vector_add attributes {grid = #df.grid<1x4>} {
  %ain = df.channel : !df.chan<tensor<64xf32>>
  %bin = df.channel : !df.chan<tensor<64xf32>>
  %cout = df.channel : !df.chan<tensor<64xf32>>

  df.actor @add on #df.tiles<all> ins(%ain, %bin) outs(%cout) {
    %a = df.recv %ain : tensor<64xf32>
    %b = df.recv %bin : tensor<64xf32>
    %c = arith.addf %a, %b : tensor<64xf32>
    df.send %cout, %c : tensor<64xf32>
  }

  df.host (%A: memref<256xf32>, %B: memref<256xf32>, %C: memref<256xf32>) {
    df.scatter %A -> %ain split(grid)
    df.scatter %B -> %bin split(grid)
    df.gather %cout -> %C split(grid)
  }
}
