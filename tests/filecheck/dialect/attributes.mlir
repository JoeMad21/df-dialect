// Grid, tile range and channel type syntax.
// RUN: df-opt %s | df-opt | filecheck %s

"test.op"() {g = #df.grid<4x8>, all = #df.tiles<all>, part = #df.tiles<0:2, 1:3>} : () -> ()
// CHECK: "test.op"() {g = #df.grid<4x8>, all = #df.tiles<all>, part = #df.tiles<0:2, 1:3>} : () -> ()

df.program @chans attributes {grid = #df.grid<1x1>} {
  // Default depth (2) is not printed.
  %d2 = df.channel : !df.chan<tensor<4xf32>, depth = 2>
  // Other depths are printed.
  %d4 = df.channel bcast(all) : !df.chan<tensor<4xf32>, depth = 4>
  df.actor @a on #df.tiles<all> ins(%d2) outs(%d4) {
    %x = df.recv %d2 : tensor<4xf32>
    df.send %d4, %x : tensor<4xf32>
  }
  df.host (%in: memref<4xf32>, %out: memref<4xf32>) {
    df.scatter %in -> %d2 split(grid)
    df.gather %d4 -> %out split(grid)
  }
}
// CHECK:      %d2 = df.channel : !df.chan<tensor<4xf32>>
// CHECK-NEXT: %d4 = df.channel bcast(all) : !df.chan<tensor<4xf32>, depth = 4>
