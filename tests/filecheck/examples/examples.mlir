// Every file in examples/ must stay valid. Add a RUN line when you add an example.
// RUN: df-opt %S/../../../examples/matmul_2x2.mlir | filecheck %s --check-prefix=MATMUL
// RUN: df-opt %S/../../../examples/toy_linear.mlir -p convert-toy-to-ttir | filecheck %s --check-prefix=TOY
// RUN: df-opt %S/../../../examples/vector_add_1x4.mlir | df-opt | filecheck %s --check-prefix=VECTOR

// MATMUL: df.program @matmul attributes {grid = #df.grid<2x2>}
// TOY: "ttir.relu"

// VECTOR: df.program @vector_add attributes {grid = #df.grid<1x4>}
// VECTOR: %[[A:.*]] = df.channel : !df.chan<tensor<64xf32>>
// VECTOR: %[[B:.*]] = df.channel : !df.chan<tensor<64xf32>>
// VECTOR: %[[C:.*]] = df.channel : !df.chan<tensor<64xf32>>
// VECTOR: df.actor @add on #df.tiles<all> ins(%[[A]], %[[B]]) outs(%[[C]])
// VECTOR: %[[X:.*]] = df.recv %[[A]] : tensor<64xf32>
// VECTOR: %[[Y:.*]] = df.recv %[[B]] : tensor<64xf32>
// VECTOR: %[[SUM:.*]] = arith.addf %[[X]], %[[Y]] : tensor<64xf32>
// VECTOR: df.send %[[C]], %[[SUM]] : tensor<64xf32>
// VECTOR: df.host (%[[HA:.*]]: memref<256xf32>, %[[HB:.*]]: memref<256xf32>, %[[HC:.*]]: memref<256xf32>)
// VECTOR: df.scatter %[[HA]] -> %[[A]] split(grid)
// VECTOR: df.scatter %[[HB]] -> %[[B]] split(grid)
// VECTOR: df.gather %[[C]] -> %[[HC]] split(grid)
