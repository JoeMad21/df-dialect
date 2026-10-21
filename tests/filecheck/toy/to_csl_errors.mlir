// convert-toy-to-csl rejects programs one PE cannot hold.
// RUN: df-opt %s --split-input-file -p convert-toy-to-csl --verify-diagnostics | filecheck %s

// examples/toy_linear.mlir: (8 + 8 + 4 x 4) K floats = 128 KB, over a 48 KB PE.
func.func @linear(%x: tensor<64x128xf32>, %w: tensor<128x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = toy.matmul %x, %w : tensor<64x64xf32>
  %1 = toy.add %0, %b : tensor<64x64xf32>
  %2 = toy.relu %1 : tensor<64x64xf32>
  func.return %2 : tensor<64x64xf32>
}
// CHECK: @linear needs 128 KB of buffers; one PE has 48 KB. Use smaller tensors.

// -----

func.func @half(%x: tensor<4x4xf16>, %w: tensor<4x4xf16>) -> tensor<4x4xf16> {
  %0 = toy.matmul %x, %w : tensor<4x4xf16>
  func.return %0 : tensor<4x4xf16>
}
// CHECK: convert-toy-to-csl: only f32 tensors are supported, got tensor<4x4xf16>
