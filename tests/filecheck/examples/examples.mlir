// Every file in examples/ must stay valid. Add a RUN line when you add an example.
// RUN: df-opt %S/../../../examples/matmul_2x2.mlir | filecheck %s --check-prefix=MATMUL
// RUN: df-opt %S/../../../examples/toy_linear.mlir -p convert-toy-to-ttir | filecheck %s --check-prefix=TOY
// RUN: df-opt %S/../../../examples/toy_linear_pe.mlir -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper,memref-to-dsd -t csl | filecheck %s --check-prefix=PE

// MATMUL: df.program @matmul attributes {grid = #df.grid<2x2>}
// TOY: "ttir.relu"
// PE: @fmaxs(out_buf_dsd, t1_buf_dsd, 0.0);
