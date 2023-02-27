module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>)
    %2 = call @expected() : () -> tensor<5x6x7xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi16>, tensor<2x2x1xi32>, tensor<5x2x2x7xi16>) -> tensor<5x6x7xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>) {
    %0 = stablehlo.constant dense<"0x0300FFFF00000000FFFFFCFFFDFFFAFF0100FFFFFEFF0100FDFF00000000FDFFFCFFFEFF02000100010005000400FCFF01000700FDFF01000800FEFFFDFFFFFFFEFFFBFF00000300FFFFFEFF0100FFFF00000000FEFF000001000400FBFFFEFF0100FEFF000004000200020001000400FAFF01000100FAFFFFFFFFFF0200000001000000FFFFFDFF00000200FCFF0300000000000000FFFF0300FDFF000002000300000000000700FEFF03000200FFFF0300030002000300000000000200FDFFFFFF0000020001000100FFFFFFFF01000200FFFFFFFFFAFFFFFF00000100FEFFFEFFFFFFFFFF000000000000F9FF0100FDFF0000FDFF02000000FDFF0300010000000400FBFF0000000000000200FEFFFCFF020001000300FDFF0000FFFFFFFF03000300FEFF0200FDFF0000FEFF0400FAFFFBFF040005000000F9FF020000000000030002000200FFFF01000000FFFFFDFF000000000200FFFF01000000FDFF00000200FDFF0800FCFF0000FCFFFFFF01000000000000000000030004000000FFFF01000100FEFF00000000010000000200FFFF0400FCFFFEFFFFFF0000000000000000"> : tensor<5x6x7xi16>
    %1 = stablehlo.constant dense<"0xFEFF00000100000006000000FDFF00000100FDFF0000FFFF000004000400FEFFFFFF04000400FFFF000003000300FEFF00000000010000000000050005000200000000000000FFFF0600000001000400000005000000FDFFFEFFFEFFFEFF02000100050000000100FAFFFDFF05000200FCFF0100010002000000FAFFFFFFFFFFFDFF0000FCFFFEFFFEFF000001000000FCFF020000000000FEFFFFFFFEFFFEFF0100FEFF030001000300FBFFFCFFFFFFFBFF0000FDFF020001000000000000000200010002000000FDFF02000000FFFF0300FEFFFFFF0300FFFFFEFF0300030000000300FFFF0000FFFF0700FFFF00000100FFFF01000000FFFF02000100010004000300FFFFFEFF03000200FEFF0300FDFF0300FCFFFDFF"> : tensor<5x2x2x7xi16>
    return %0, %1 : tensor<5x6x7xi16>, tensor<5x2x2x7xi16>
  }
  func.func private @expected() -> tensor<5x6x7xi16> {
    %0 = stablehlo.constant dense<"0x030000000100000006000000FDFF00000100FFFF00000100000004000400FEFFFFFF040004000100010005000400FEFF01000700010001000800FEFFFDFFFFFFFEFFFBFF00000300FFFFFEFF0100FFFF000000000000050005000400000000000100FFFF060004000200040001000500000001000100FEFFFFFF02000200050001000100FFFFFDFF05000200FCFF0300000000000000FFFF0300FDFF000002000300000000000700FEFF0300020002000300030002000300000000000200FEFFFFFF00000200010001000200000001000200FFFFFFFFFEFF0100000003000100FEFFFFFFFFFF000000000000F9FF0100FDFF0000FDFF02000000FDFF0300010000000400FBFF00000000020002000000000002000200030002000000FFFF02000300030003000200FFFF0300FFFF040003000300040005000000F9FF020000000000030002000200FFFF01000000FFFF0000030000000200FFFF0700000000000100020001000800FFFF02000100010004000300000000000300030004000300FFFF03000100FEFF00000000010000000200FFFF0400FCFFFEFFFFFF0000000000000000"> : tensor<5x6x7xi16>
    return %0 : tensor<5x6x7xi16>
  }
}
