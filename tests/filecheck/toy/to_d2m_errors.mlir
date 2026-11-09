// RUN: df-opt %s --split-input-file -p convert-toy-to-d2m --verify-diagnostics | filecheck %s
func.func @half(%x: tensor<4x4xf16>) -> tensor<4x4xf16> {
  %0 = toy.relu %x : tensor<4x4xf16>
  func.return %0 : tensor<4x4xf16>
}
// CHECK: convert-toy-to-d2m: only f32 tensors are supported

// -----

func.func @dynamic(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = toy.relu %x : tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}
// CHECK: convert-toy-to-d2m: expected unencoded rank-2 tensors with positive static dimensions divisible by 32

// -----

func.func @unaligned(%x: tensor<33x64xf32>) -> tensor<33x64xf32> {
  %0 = toy.relu %x : tensor<33x64xf32>
  func.return %0 : tensor<33x64xf32>
}
// CHECK: convert-toy-to-d2m: expected unencoded rank-2 tensors with positive static dimensions divisible by 32

// -----

func.func @vector(%x: tensor<32xf32>) -> tensor<32xf32> {
  %0 = toy.relu %x : tensor<32xf32>
  func.return %0 : tensor<32xf32>
}
// CHECK: convert-toy-to-d2m: expected unencoded rank-2 tensors with positive static dimensions divisible by 32
