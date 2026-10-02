// Channel checks from the Verifier Checks slide.
// RUN: df-opt %s --split-input-file --verify-diagnostics | filecheck %s

// Type mismatch: df.recv type must equal the channel element type.
df.program @p attributes {grid = #df.grid<2x2>} {
  %arow = df.channel : !df.chan<tensor<64x128xbf16>>
  %bcol = df.channel : !df.chan<tensor<128x64xbf16>>
  df.actor @tile on #df.tiles<all> ins(%arow, %bcol) outs() {
    %a = df.recv %bcol : tensor<64x128xbf16>
  }
  df.host (%A: memref<128x128xbf16>) {
    df.scatter %A -> %arow split(rows)
    df.scatter %A -> %bcol split(cols)
  }
}
// CHECK: channel %bcol carries tensor<128x64xbf16>

// -----

// Type mismatch on the send side.
df.program @p attributes {grid = #df.grid<1x1>} {
  %c = df.channel : !df.chan<tensor<4xf32>>
  df.actor @a on #df.tiles<all> ins() outs(%c) {
    %x = "test.op"() : () -> tensor<8xf32>
    df.send %c, %x : tensor<8xf32>
  }
  df.host (%C: memref<4xf32>) {
    df.gather %c -> %C split(grid)
  }
}
// CHECK: channel %c carries tensor<4xf32>

// -----

// Unread channel: something sends to %cout but nothing reads it.
df.program @p attributes {grid = #df.grid<1x1>} {
  %cout = df.channel : !df.chan<tensor<4xf32>>
  df.actor @a on #df.tiles<all> ins() outs(%cout) {
    %x = "test.op"() : () -> tensor<4xf32>
    df.send %cout, %x : tensor<4xf32>
  }
  df.host () {
  }
}
// CHECK: channel %cout is sent to but never read

// -----

// Unwritten channel: something reads %in but nothing sends to it.
df.program @p attributes {grid = #df.grid<1x1>} {
  %in = df.channel : !df.chan<tensor<4xf32>>
  df.actor @a on #df.tiles<all> ins(%in) outs() {
    %x = df.recv %in : tensor<4xf32>
  }
  df.host () {
  }
}
// CHECK: channel %in is read but never sent to

// -----

// An actor may only receive from channels listed in ins(...).
df.program @p attributes {grid = #df.grid<1x1>} {
  %in = df.channel : !df.chan<tensor<4xf32>>
  df.actor @a on #df.tiles<all> ins() outs() {
    %x = df.recv %in : tensor<4xf32>
  }
  df.host (%A: memref<4xf32>) {
    df.scatter %A -> %in split(grid)
  }
}
// CHECK: df.recv reads %in, which is not in ins(...) of @a
