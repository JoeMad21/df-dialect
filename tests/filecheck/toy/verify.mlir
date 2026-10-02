// Toy verifier errors.
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s

%a = "test.op"() : () -> tensor<4x8xf32>
%b = "test.op"() : () -> tensor<4x8xf32>
%c = toy.matmul %a, %b : tensor<4x8xf32>
// CHECK: toy.matmul inner dimensions differ: 8 vs 4

// -----

%a = "test.op"() : () -> tensor<4x8xf32>
%b = "test.op"() : () -> tensor<8x2xf32>
%c = toy.matmul %a, %b : tensor<4x8xf32>
// CHECK: toy.matmul result should be tensor<4x2xf32>

// -----

%a = "test.op"() : () -> tensor<4x8x2xf32>
%b = "test.op"() : () -> tensor<8x2xf32>
%c = toy.matmul %a, %b : tensor<4x2xf32>
// CHECK: toy.matmul operands must be rank 2

// -----

%a = "test.op"() : () -> tensor<4xf32>
%b = "test.op"() : () -> tensor<8xf32>
%c = "toy.add"(%a, %b) : (tensor<4xf32>, tensor<8xf32>) -> tensor<4xf32>
// CHECK: attribute tensor<4xf32> expected from variable 'T', but got tensor<8xf32>
