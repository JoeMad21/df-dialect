// df.contract shape and spec checks.
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s

%a = "test.op"() : () -> tensor<4x8xf32>
%b = "test.op"() : () -> tensor<4x8xf32>
%c = df.contract "mk,kn->mn" %a, %b : tensor<4x8xf32>
// CHECK: df.contract: index 'k' is 8 in one operand and 4 in another

// -----

%a = "test.op"() : () -> tensor<4x8xf32>
%b = "test.op"() : () -> tensor<8x2xf32>
%c = df.contract "mk,kn->mn" %a, %b : tensor<4x8xf32>
// CHECK: df.contract result should be tensor<4x2xf32>

// -----

%a = "test.op"() : () -> tensor<4x8xf32>
%b = "test.op"() : () -> tensor<8x2xf32>
%c = df.contract "mk,kn" %a, %b : tensor<4x2xf32>
// CHECK: df.contract: spec 'mk,kn' must contain exactly one '->'

// -----

%a = "test.op"() : () -> tensor<4x8xf32>
%b = "test.op"() : () -> tensor<8x2xbf16>
%c = df.contract "mk,kn->mn" %a, %b : tensor<4x2xf32>
// CHECK: operands and result of df.contract must share an element type
