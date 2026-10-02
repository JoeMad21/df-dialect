// df.contract accepts any well-formed two-input einsum spec.
// RUN: df-opt %s | df-opt | filecheck %s

%a = "test.op"() : () -> tensor<64x128xbf16>
%b = "test.op"() : () -> tensor<128x32xbf16>
%c = df.contract "mk,kn->mn" %a, %b : tensor<64x32xbf16>
// CHECK: %c = df.contract "mk,kn->mn" %a, %b : tensor<64x32xbf16>

%q = "test.op"() : () -> tensor<8x16x64xf32>
%k = "test.op"() : () -> tensor<8x32x64xf32>
%s = df.contract "bqd,bkd->bqk" %q, %k : tensor<8x16x32xf32>
// CHECK: %s = df.contract "bqd,bkd->bqk" %q, %k : tensor<8x16x32xf32>
