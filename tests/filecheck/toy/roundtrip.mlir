// Toy dialect syntax round-trips in custom and generic form.
// RUN: df-opt %s | df-opt | filecheck %s
// RUN: df-opt %s --print-op-generic | df-opt | filecheck %s

func.func @linear(%x: tensor<64x128xf32>, %w: tensor<128x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = toy.matmul %x, %w : tensor<64x64xf32>
  %1 = toy.add %0, %b : tensor<64x64xf32>
  %2 = toy.relu %1 : tensor<64x64xf32>
  func.return %2 : tensor<64x64xf32>
}
// CHECK:      %0 = toy.matmul %x, %w : tensor<64x64xf32>
// CHECK-NEXT: %1 = toy.add %0, %b : tensor<64x64xf32>
// CHECK-NEXT: %2 = toy.relu %1 : tensor<64x64xf32>
