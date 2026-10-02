// Syntax errors are reported with the offending location.
// RUN: df-opt %s --split-input-file --parsing-diagnostics | filecheck %s

"test.op"() {g = #df.grid<2x2x2>} : () -> ()
// CHECK: expected a 2D grid such as <2x2>

// -----

df.program @p attributes {grid = #df.grid<1x1>} {
  %c = df.channel : tensor<4xf32>
}
// CHECK: df.channel must produce a !df.chan type
