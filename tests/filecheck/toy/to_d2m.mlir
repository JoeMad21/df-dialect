// RUN: df-opt %s -p convert-toy-to-d2m | filecheck %s
// RUN: df-opt %s -p convert-toy-to-d2m | df-opt | filecheck %s
// RUN: df-opt %s -p convert-toy-to-d2m | filecheck %s --check-prefix=ABSENT

func.func @linear(%x: tensor<64x128xf32>, %w: tensor<128x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = toy.matmul %x, %w : tensor<64x64xf32>
  %1 = toy.add %0, %b : tensor<64x64xf32>
  %2 = toy.relu %1 : tensor<64x64xf32>
  func.return %2 : tensor<64x64xf32>
}
// CHECK: "d2m.empty"
// CHECK: "d2m.generic"(%x, %w,
// CHECK: !d2m.cb<tensor<64x128xf32>>
// CHECK: "d2m.wait"
// CHECK: tensor<2x4x!ttcore.tile<32x32, f32>>
// CHECK: "d2m.tile_tilize_block"
// CHECK: "d2m.wait"
// CHECK: tensor<4x2x!ttcore.tile<32x32, f32>>
// CHECK: "d2m.tile_tilize_block"
// CHECK: "d2m.reserve"
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK-NEXT: {{.*}}"d2m.tile_fill"
// CHECK-NEXT: linalg.yield {{.*}} : !ttcore.tile<32x32, f32>
// CHECK: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: "d2m.tile_matmul"
// CHECK-NEXT: linalg.yield {{.*}} : !ttcore.tile<32x32, f32>
// CHECK: "d2m.tile_untilize_block"
// CHECK: "d2m.yield"
// CHECK: grid = #ttcore.grid<1x1>
// CHECK-SAME: threads = [#d2m.thread<unified>]
// CHECK-SAME: operandSegmentSizes = array<i32: 2, 1>
// CHECK: "d2m.generic"
// CHECK: "d2m.tile_add"
// CHECK: "d2m.tile_untilize_block"
// CHECK: "d2m.yield"
// CHECK: "d2m.generic"
// CHECK: "d2m.tile_relu"
// CHECK: "d2m.tile_untilize_block"
// CHECK: "d2m.yield"
// CHECK: func.return
// ABSENT-NOT: toy.
// ABSENT-NOT: ttir.
// ABSENT-NOT: arith.mulf
// ABSENT-NOT: arith.addf
// ABSENT-NOT: arith.maximumf
