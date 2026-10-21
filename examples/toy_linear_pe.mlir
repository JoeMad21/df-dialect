// The toy linear layer, relu(x @ w + b), sized to fit one Cerebras PE (48 KB).
// Lower to CSL:
//   uv run df-opt examples/toy_linear_pe.mlir -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper,memref-to-dsd -t csl

func.func @linear(%x: tensor<16x32xf32>, %w: tensor<32x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = toy.matmul %x, %w : tensor<16x16xf32>
  %1 = toy.add %0, %b : tensor<16x16xf32>
  %2 = toy.relu %1 : tensor<16x16xf32>
  func.return %2 : tensor<16x16xf32>
}
