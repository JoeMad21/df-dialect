// A toy linear layer, relu(x @ w + b), written in the toy dialect.
// Lower to TTIR:  uv run df-opt examples/toy_linear.mlir -p convert-toy-to-ttir

func.func @linear(%x: tensor<64x128xf32>, %w: tensor<128x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = toy.matmul %x, %w : tensor<64x64xf32>
  %1 = toy.add %0, %b : tensor<64x64xf32>
  %2 = toy.relu %1 : tensor<64x64xf32>
  func.return %2 : tensor<64x64xf32>
}
