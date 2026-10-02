// Every file in examples/ must stay valid. Add a RUN line when you add an example.
// RUN: df-opt %S/../../../examples/matmul_2x2.mlir | filecheck %s --check-prefix=MATMUL

// MATMUL: df.program @matmul attributes {grid = #df.grid<2x2>}
