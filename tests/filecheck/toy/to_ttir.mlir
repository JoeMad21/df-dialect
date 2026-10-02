// convert-toy-to-ttir replaces every toy op with the matching TTIR op,
// printed in the generic form tt-mlir expects.
// RUN: df-opt %s -p convert-toy-to-ttir | filecheck %s

func.func @linear(%x: tensor<64x128xf32>, %w: tensor<128x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = toy.matmul %x, %w : tensor<64x64xf32>
  %1 = toy.add %0, %b : tensor<64x64xf32>
  %2 = toy.relu %1 : tensor<64x64xf32>
  func.return %2 : tensor<64x64xf32>
}
// CHECK:      %0 = "ttir.matmul"(%x, %w) : (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: %1 = "ttir.add"(%0, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: %2 = "ttir.relu"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT: func.return %2 : tensor<64x64xf32>
// CHECK-NOT:  toy.
