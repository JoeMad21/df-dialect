// Placement checks: tile ranges must fit the grid.
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s

// Off-grid placement from the Verifier Checks slide.
df.program @p attributes {grid = #df.grid<2x2>} {
  df.actor @tile on #df.tiles<0:4, 0:4> ins() outs() {
  }
  df.host () {
  }
}
// CHECK: tiles 0:4, 0:4 exceed #df.grid<2x2>

// -----

// Empty or reversed ranges are rejected.
"test.op"() {t = #df.tiles<2:1, 0:1>} : () -> ()
// CHECK: invalid row range in #df.tiles<2:1, 0:1>

// -----

"test.op"() {g = #df.grid<0x4>} : () -> ()
// CHECK: grid dimensions must be positive, got 0x4
