// The Syntax Walkthrough program must round-trip in both custom and generic form.
// RUN: df-opt %s | df-opt | filecheck %s
// RUN: df-opt %s --print-op-generic | df-opt | filecheck %s

df.program @matmul attributes {grid = #df.grid<2x2>} {
  %arow = df.channel bcast(row) : !df.chan<tensor<64x128xbf16>>
  %bcol = df.channel bcast(col) : !df.chan<tensor<128x64xbf16>>
  %cout = df.channel : !df.chan<tensor<64x64xbf16>>
  df.actor @tile on #df.tiles<all> ins(%arow, %bcol) outs(%cout) {
    %a = df.recv %arow : tensor<64x128xbf16>
    %b = df.recv %bcol : tensor<128x64xbf16>
    %c = df.contract "mk,kn->mn" %a, %b : tensor<64x64xbf16>
    df.send %cout, %c : tensor<64x64xbf16>
  }
  df.host (%A: memref<128x128xbf16>, %B: memref<128x128xbf16>, %C: memref<128x128xbf16>) {
    df.scatter %A -> %arow split(rows)
    df.scatter %B -> %bcol split(cols)
    df.gather %cout -> %C split(grid)
  }
}

// CHECK:      df.program @matmul attributes {grid = #df.grid<2x2>} {
// CHECK-NEXT:   %arow = df.channel bcast(row) : !df.chan<tensor<64x128xbf16>>
// CHECK-NEXT:   %bcol = df.channel bcast(col) : !df.chan<tensor<128x64xbf16>>
// CHECK-NEXT:   %cout = df.channel : !df.chan<tensor<64x64xbf16>>
// CHECK-NEXT:   df.actor @tile on #df.tiles<all> ins(%arow, %bcol) outs(%cout) {
// CHECK-NEXT:     %a = df.recv %arow : tensor<64x128xbf16>
// CHECK-NEXT:     %b = df.recv %bcol : tensor<128x64xbf16>
// CHECK-NEXT:     %c = df.contract "mk,kn->mn" %a, %b : tensor<64x64xbf16>
// CHECK-NEXT:     df.send %cout, %c : tensor<64x64xbf16>
// CHECK-NEXT:   }
// CHECK-NEXT:   df.host (%A: memref<128x128xbf16>, %B: memref<128x128xbf16>, %C: memref<128x128xbf16>) {
// CHECK-NEXT:     df.scatter %A -> %arow split(rows)
// CHECK-NEXT:     df.scatter %B -> %bcol split(cols)
// CHECK-NEXT:     df.gather %cout -> %C split(grid)
// CHECK-NEXT:   }
// CHECK-NEXT: }
