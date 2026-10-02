// Ops must appear in the right place.
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s

// df.recv must be inside a df.actor.
df.program @p attributes {grid = #df.grid<1x1>} {
  %in = df.channel : !df.chan<tensor<4xf32>>
  df.host (%A: memref<4xf32>) {
    df.scatter %A -> %in split(grid)
    %x = df.recv %in : tensor<4xf32>
  }
}
// CHECK: df.recv must be nested inside df.actor

// -----

// df.channel must be a direct child of df.program.
%c = df.channel : !df.chan<tensor<4xf32>>
// CHECK: 'df.channel' expects parent op 'df.program'
