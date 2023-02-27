module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>)
    %2 = call @expected() : () -> tensor<5x6x7xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      stablehlo.return %arg1 : tensor<i16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi16>, tensor<2x2x1xi32>, tensor<5x2x2x7xi16>) -> tensor<5x6x7xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>) {
    %0 = stablehlo.constant dense<"0xFEFFFEFF0000FFFF00000500000004000000010000000100FCFF030005000500000000000500FFFFFCFFFDFFFFFF02000100FFFF0000FDFFFDFF0000FFFF0200FDFFFCFFFFFF02000000030000000600000002000500FEFF0000FBFFFBFF00000300FEFF0000FCFF02000400FFFFFFFFFDFFFCFF000005000000FFFFFDFF00000000000003000000000003000000FFFFFDFFFDFFFFFF00000100FEFFFEFFFDFFFFFF0000000000000600FDFF0000000000000200FAFF05000200F9FF0200050001000200FEFFFEFF00000000FDFF0300FDFF010001000300FFFF0300FEFF0200FFFF01000000FAFF000001000300FFFF000000000000FFFFFFFFFFFFFEFF0200FDFF0100FCFF0100FEFF00000200000000000200000000000300FDFF00000200FEFF000001000000FBFF0400FEFF010001000500FFFFFFFFFEFF0000FEFF0000000000000500FEFF00000200FEFF00000000000003000500000000000000FCFFFCFF010002000000FEFF040000000000FFFFFFFFFCFFFEFF0100FFFFFFFF03000200FDFF00000200FAFFFFFFFDFF0400FEFF01000A00050001000000000000000100FDFF"> : tensor<5x6x7xi16>
    %1 = stablehlo.constant dense<"0x0600040000000000FAFF000002000200FFFFFDFF01000300000000000000030000000100020000000200FDFF0000020000000000FBFFFDFFFBFF03000000FFFF0000020003000000FDFFFEFF0500FFFF000003000000FBFF00000300030001000000040000000100FFFF000004000200FAFF0300FFFF040000000200FBFF0100FEFF0000000005000200010000000500FBFF0300000000000000FFFF0100FDFF010000000100060000000300FFFF0100020003000000FFFFFBFFFDFF01000200FFFFFAFFFDFFFFFF01000000040003000300FAFF0300FBFFFFFF0000FFFF00000400FDFF02000200FFFF0000000000000200FFFF00000100FFFF01000000000001000400FFFFFBFF01000000FDFF0000FEFF00000300FFFF"> : tensor<5x2x2x7xi16>
    return %0, %1 : tensor<5x6x7xi16>, tensor<5x2x2x7xi16>
  }
  func.func private @expected() -> tensor<5x6x7xi16> {
    %0 = stablehlo.constant dense<"0x0600040000000000FAFF000002000200FFFFFDFF01000300000000000000030000000100020000000200FDFF0000020000000000FBFFFDFFFDFF0000FFFF0200FDFFFCFFFFFF0200000003000000060000000200FBFF03000000FFFF0000020003000000FDFFFEFF0500FFFF000003000000FBFF00000300030001000000040000000100FFFF0000040002000000FFFFFDFFFDFFFFFF00000100FEFFFEFFFDFFFFFF000000000000FAFF0300FFFF040000000200FBFF0100FEFF0000000005000200010000000500FBFF0300000000000000FFFF0100FDFF0100000001000600FFFF01000000FAFF000001000300FFFF000000000000FFFFFFFFFFFF00000300FFFF0100020003000000FFFFFBFFFDFF01000200FFFFFAFFFDFFFFFF01000000040003000300FAFF0300FBFFFFFF0000FFFF0000FFFFFFFFFEFF0000FEFF0000000000000500FEFF00000200FEFF00000400FDFF02000200FFFF0000000000000200FFFF00000100FFFF01000000000001000400FFFFFBFF01000000FDFF0000FEFF00000300FFFFFAFFFFFFFDFF0400FEFF01000A00050001000000000000000100FDFF"> : tensor<5x6x7xi16>
    return %0 : tensor<5x6x7xi16>
  }
}
