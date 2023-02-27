module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xi16>, tensor<20x20xi16>)
    %1 = call @expected() : () -> tensor<20x20xi16>
    %2 = stablehlo.and %0#0, %0#1 : tensor<20x20xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xi16>, tensor<20x20xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xi16>, tensor<20x20xi16>) {
    %0 = stablehlo.constant dense<"0x000000000000000001000000020002000000FBFF00000000F9FFFFFFFFFF0400030001000100020000000000FBFFFFFFFEFFFEFF00000200FEFFFFFFFFFF0200040002000000020000000100FDFF0000FEFFFDFFFDFF00000400000000000200FDFFFFFF00000200FEFFFBFF030001000200000003000200030000000200FFFF0000FFFFFFFF0000FCFFFEFFFDFF0300FFFF0000FFFFFFFF0000FFFFFCFFFEFF0200FBFFFDFFFDFF07000400FEFF0200030001000100FEFF0700FFFF020000000300FEFF0600FEFF0200FEFF030003000000010001000300FFFFFEFFFCFF0100FDFFFCFF0000FFFF0000FFFFFDFF0000020001000100FCFF0000FEFF04000400FBFF07000300000002000000040001000200FDFFFFFF0000FEFF010005000300000003000100FAFF00000000FFFF00000100000000000000FEFF0000FAFF00000100FFFF0000FCFF0000FEFF020001000000020003000200FCFF0200FEFFFAFFFEFFFEFFFCFF0000FFFFFBFFFDFF0000000000000000FDFF02000000FEFFFEFF000000000100020004000400010000000200FCFF020002000200030007000300FBFF0200000006000100FFFFFEFFFEFFFDFF010002000000FDFF0000FEFF000000000100FFFF0200FEFF0000010001000400F8FF050002000200050004000000FFFF0000FEFFF9FFFEFF0200FDFFFEFF010001000100FFFF0000FBFF0100FDFF00000300FEFF02000200FEFF020006000000FEFFFEFF0000FEFFFDFFFDFFFEFF000006000400FDFF060004000200FFFF0200FFFF010003000000FDFFFEFFFFFF00000000FFFFFFFF0400000001000000FCFFFFFFFCFFFCFF0000030002000000FDFF04000000FCFFFFFF08000200FEFF04000300010002000000000003000700FEFF0100FDFF0400000001000100040002000200FAFF03000200FFFFFDFF0000FEFF0700000001000300FEFF0500FCFF00000000FEFF00000100FCFF00000400FFFFFEFFFCFFFFFF0000FFFFF7FF0000FBFF0400FFFFFDFF0000FCFF01000300030000000300FBFFFAFF00000100FDFFFEFF00000500FEFF0200020000000100FEFFFEFF00000300FEFF0000FCFF0400FEFF0000000004000000FCFF00000000"> : tensor<20x20xi16>
    %1 = stablehlo.constant dense<"0xFDFFFEFF04000000FFFF00000400FFFF00000500000000000300040000000300000005000000FCFFFFFF000004000300FEFFFFFF010002000100FCFF00000600FDFFFEFFFCFF02000400FEFF0000080000000500FFFFFBFF0000FCFFFEFF0100FDFF0100020006000300FEFF0100FDFF000002000000FDFF05000100FDFFFDFF0000FEFFFFFFFFFF04000300000003000000FEFF0000FFFF01000000FEFF00000500010001000100030001000000FFFF05000000FEFFFFFF03000200FEFFFDFF06000300FFFF03000000000000000000FEFFFAFFFDFF010000000000FFFF01000600F9FF0000FFFF010001000100FBFF050007000000FEFF01000000000001000300FCFF00000000FEFF00000200FDFF0200FFFF000005000100FEFFFFFF000000000000FFFFFEFF0300FFFF01000400060001000100010000000100FCFF0300F9FF04000000FFFF0100FCFF00000000FEFF0200FCFF0200FFFFFDFF05000600FEFFFCFFFCFFFFFFFEFF0100FEFF0100FDFF0000020003000000000000000100FFFF030001000100020006000300FFFF000002000400010001000100FEFF05000000FBFF0100000003000100FFFF0100FFFF0100FAFFFEFF00000100FEFF02000100000002000100010006000100FEFF00000000FEFF00000300010005000000FFFF0000FFFFFCFF00000100040002000200FEFF010000000100FFFFFCFF0100FEFF0000030001000100FDFF01000300FEFFFEFF01000000FBFF00000700F9FFFDFF020005000000FBFFFFFF0100FEFFFFFFFFFF00000000000003000000FEFF0200FEFFFDFFFDFF0100FFFFFDFF06000000040002000500FCFFFEFFFBFF00000300F9FFFAFF04000000000005000000FAFFFFFFFBFF0000FFFFFFFFFFFF000002000000FEFF00000400FFFF0100FBFFFEFFFEFFFEFF0200FEFF000001000000FEFF000002000500FEFFFFFFFCFF000001000000FEFF00000100FEFF010004000000FDFF0200FCFF0000FFFFFBFFFFFF02000200FCFFFFFF0000040002000300FFFF0400FCFFFEFF0200000000000000000002000100FFFFFEFFFFFF05000000010000000400FFFF00000300FFFF00000000FFFF05000200070000000100FBFF"> : tensor<20x20xi16>
    return %0, %1 : tensor<20x20xi16>, tensor<20x20xi16>
  }
  func.func private @expected() -> tensor<20x20xi16> {
    %0 = stablehlo.constant dense<"0x000000000000000001000000000002000000010000000000010004000000000000000100000000000000000000000300FEFFFEFF000002000000FCFF000002000400020000000200000000000000000000000500FDFF00000000000000000000FDFF0100000002000200FAFF010001000000000000000000010000000000FDFF0000FEFFFFFF00000400020000000300000000000000FFFF00000000FCFF000000000100010001000300000000000200010000000000FEFF030002000200000002000200060002000000000000000000000000000100010000000000FCFF01000400F8FF0000FFFF0000010001000000000001000000FCFF0000000000000000030004000000000002000000000001000200FDFF000000000000000005000000000000000100FAFF0000000001000000000000000000000000000000F8FF0000010004000000FCFF0000FCFF000000000000020000000200FCFF000004000200FEFFFCFFFCFF0000FEFF0100FCFF0000000000000000010000000000000000000000000001000000000004000100000000000000000000000000010006000100000002000000000001000100FEFF0000FDFF01000200000000000000FEFF00000000000002000000000000000100000000000000040000000200010004000000FFFF0000FEFFF8FF000000000400020000000000010000000000FBFF0000010000000000020000000000FCFF000002000000FEFF00000000FAFF00000500F8FF0000020004000000020004000000FEFF0200FFFF00000000000001000000FEFF00000000FDFFFDFF000000000100000000000400000004000000020002000000010000000000040000000000000000000000030001000000000000000300000002000000FCFF0000000001000100000002000200FAFF02000200000001000000FEFF0000000001000200FEFF0400000000000000FEFF00000100FCFF000004000000FCFF0000FCFF0000FFFFF3FF000002000000FCFFFDFF0000040000000300030000000000FAFF0200000000000000000000000100FEFF0200020000000000000000000000030000000000FCFF000000000000000000000000000000000000"> : tensor<20x20xi16>
    return %0 : tensor<20x20xi16>
  }
}
