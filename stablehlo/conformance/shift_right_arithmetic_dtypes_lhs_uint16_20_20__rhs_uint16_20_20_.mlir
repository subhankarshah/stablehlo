module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xui16>, tensor<20x20xui16>)
    %1 = call @expected() : () -> tensor<20x20xui16>
    %2 = stablehlo.shift_right_arithmetic %0#0, %0#1 : tensor<20x20xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xui16>, tensor<20x20xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xui16>, tensor<20x20xui16>) {
    %0 = stablehlo.constant dense<"0x0100000001000400010000000100020005000500030000000200020001000200020003000100010003000100000001000500080001000100020000000000000003000000020003000300000000000400050000000600010000000000000001000300040004000100040001000100020001000200010006000300000001000100040004000300000000000000020000000300030000000100000003000100010004000000010000000400040000000400030002000100000001000200030001000400000003000100010002000000010002000300030000000300050004000400020000000500010001000000030002000000010000000100030006000000030004000100000003000000030002000100010000000100000001000000000000000300030002000500060003000200000000000000010001000100020000000300010003000000020001000100000001000500000004000100040004000300010005000500050005000100010004000200020004000500000000000200020000000000000000000200020001000100040005000000000000000000040003000000040000000200050000000400020002000100020000000500030001000400060000000200030003000100020000000300050000000200020000000000000000000200010003000400000001000400020008000100010002000300000002000100010003000000000002000200020000000000010003000100020006000000020002000200000005000100040001000000020000000000020001000200000000000300000001000100000001000000050001000000010000000000020005000000000002000200020000000000000000000500000000000100050001000200010002000200010002000100020002000200020005000000000001000200030001000200040001000000010004000400010004000200010003000000030001000300000001000100030003000200000001000500010001000100020004000200030001000300000004000400040005000200010002000000000000000000020002000200030003000200000001000300030002000200000005000300000001000000"> : tensor<20x20xui16>
    %1 = stablehlo.constant dense<"0x0000010006000300020003000000030003000000010006000500000001000400020004000300010000000100000003000000030001000100010000000300000001000100030000000200010000000200000002000100040000000500050003000600020004000300020000000800000002000100020003000100030005000100030001000000010002000300010001000100000005000000050000000200020001000200020006000100000001000500030002000300020001000000010002000100030002000300030004000000000001000400000001000000020001000000010000000200020003000300040003000200030002000200000002000400040003000000030005000100010001000200010001000000010002000000020004000100000001000100000000000400010003000100030002000000000004000100010003000000000005000200020001000100010005000700030000000000010000000400020003000200000000000300040000000600000002000200020001000500040006000300040002000100020000000000020002000300000000000600010002000000040000000100050004000000030002000200020003000000000003000700050004000200000001000100010006000000010002000100000004000300030001000300030004000500040001000000020002000200000003000800050000000300010003000200030000000000000005000000020000000000020004000000050006000100010000000400030004000000060001000100010000000300010000000400020001000100020000000100040003000300010000000100020001000000040004000500040001000000010003000000010001000000000002000100010002000200020001000500010000000000020003000300020001000000040002000100000002000100010001000100010001000200000003000500030003000200020001000000000001000200030003000300040002000300000004000100010002000300000005000100000000000000030001000200000000000100010004000000000000000100000002000300040002000000000004000000"> : tensor<20x20xui16>
    return %0, %1 : tensor<20x20xui16>, tensor<20x20xui16>
  }
  func.func private @expected() -> tensor<20x20xui16> {
    %0 = stablehlo.constant dense<"0x0100000000000000000000000100000000000500010000000000020000000000000000000000000003000000000000000500010000000000010000000000000001000000000003000000000000000100050000000300000000000000000000000000010000000000010001000000020000000100000000000100000000000000000002000300000000000000010000000100030000000100000003000000000002000000000000000200040000000000000000000000000000000200010000000200000000000000000000000000010001000000030000000300010002000400010000000100000000000000000000000000000000000000030001000000000000000100000000000000010001000000000000000100000000000000000000000100030001000200060003000000000000000000000000000100020000000100000000000000020000000000000000000200000000000000000004000300000005000000010000000000010004000000000004000000000000000000000000000000000000000000000000000000010005000000000000000000040003000000020000000200000000000200000000000100000000000100000000000400060000000000000000000000020000000100020000000200010000000000000000000000000001000000000000000000000004000100000000000000000000000000000003000000000000000000000000000000010000000100000006000000000000000200000000000000020001000000000000000000000000000100000000000000000001000000000000000000010001000000000000000000010005000000000001000200000000000000000000000500000000000100020000000200010000000100000000000000000001000000010005000000000000000000000000000200000000000000010001000200000002000100000001000000030000000000000000000000000001000200000000000100000000000000000001000000030000000100000001000000040000000100010002000000000000000000020002000100010000000200000001000100030000000000000001000300000000000000"> : tensor<20x20xui16>
    return %0 : tensor<20x20xui16>
  }
}
