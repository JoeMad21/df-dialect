// convert-toy-to-csl builds a one-PE program in xDSL's csl_wrapper and csl
// dialects. xDSL's own passes then lower it and print CSL source.
// RUN: df-opt %s -p convert-toy-to-csl | filecheck %s
// RUN: df-opt %s -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper | filecheck %s --check-prefix=CSLIR
// RUN: df-opt %s -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper,memref-to-dsd -t csl | filecheck %s --check-prefix=CSL

func.func @linear(%x: tensor<16x32xf32>, %w: tensor<32x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = toy.matmul %x, %w : tensor<16x16xf32>
  %1 = toy.add %0, %b : tensor<16x16xf32>
  %2 = toy.relu %1 : tensor<16x16xf32>
  func.return %2 : tensor<16x16xf32>
}

// Stage 1: toy -> csl_wrapper + csl + linalg on memrefs.
// CHECK:      "csl_wrapper.module"() <{width = 1 : i16, height = 1 : i16, params = [], target = "wse3", program_name = "linear"}>
// CHECK:      %x_buf = memref.alloc() : memref<512xf32>
// CHECK:      "csl.export"({{.*}}) <{var_name = "x"
// CHECK:      %out_buf = memref.alloc() : memref<256xf32>
// CHECK:      "csl.export"({{.*}}) <{var_name = "out"
// CHECK:      csl.func @linear() {
// CHECK:        "csl.fmovs"(%t0_buf, {{.*}})
// CHECK:        scf.for %i =
// CHECK:          scf.for %k =
// CHECK:            "csl.fmacs"
// CHECK:        linalg.add ins(%t0_buf, %b_buf : memref<256xf32>, memref<256xf32>) outs(%t1_buf : memref<256xf32>)
// CHECK:        linalg.max ins(%t1_buf, {{.*}}) outs(%out_buf : memref<256xf32>)
// CHECK:        "csl.member_call"({{.*}}) <{field = "unblock_cmd_stream"}>
// CHECK-NOT:  toy.

// Stage 2: after linalg-to-csl and lower-csl-wrapper, only CSL-IR modules remain.
// CSLIR:      "csl.module"() <{kind = #csl<module_kind layout>}>
// CSLIR:        "csl.set_tile_code"({{.*}}) <{file = "linear.csl"}>
// CSLIR:      "csl.module"() <{kind = #csl<module_kind program>}>
// CSLIR:        "csl.fadds"(%t1_buf, %t0_buf, %b_buf)
// CSLIR:        "csl.fmaxs"(%out_buf, %t1_buf, {{.*}})
// CSLIR-NOT:  linalg.

// Stage 3: CSL source, program then layout.
// CSL:      // FILE: linear_program
// CSL:      fn linear() void {
// CSL:        @fmovs(t0_buf_dsd, 0.0);
// CSL:        @fmacs({{.*}}, (x_buf[((i * 32) + k)]));
// CSL:        @fadds(t1_buf_dsd, t0_buf_dsd, b_buf_dsd);
// CSL:        @fmaxs(out_buf_dsd, t1_buf_dsd, 0.0);
// CSL:        memcpy.unblock_cmd_stream();
// CSL:      // FILE: linear_layout
// CSL:        @set_tile_code(x, y, "linear.csl", {{.*}});
// CSL:        @export_name("x", [*]f32, true);
// CSL:        @export_name("out", [*]f32, true);
// CSL:        @export_name("linear", fn() void, );
