module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>)
    %2 = call @expected() : () -> tensor<5x6x7xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi16>, tensor<2x2x2xi32>, tensor<5x2x2xi16>) -> tensor<5x6x7xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>) {
    %0 = stablehlo.constant dense<"0x0200FFFFFBFF00000100FDFF00000200FEFF0300010000000400FAFFFFFFFEFFFDFF00000000FFFF00000700FFFF0200000000000000F9FF06000200F9FF00000000FCFF0000FEFF010000000300FBFF0000000003000000020003000000FEFF0200F8FF0000050000000000FFFF0000040001000200FCFF0100FDFFFFFFFEFF00000100000004000000FDFFFDFFFBFFFDFF0000F9FF0300060002000000000000000200FFFF0300FDFF0000030000000000FDFF020001000600030002000300FBFF0200000000000200FFFFFDFF0000FBFF02000000FFFF03000000000000000400000004000300FDFF0000000005000100FEFFFDFF010003000000FDFFFFFF000003000300000000000700FCFFFEFF01000000FDFF0400000000000200090000000300FEFFFFFF0000040000000500FEFF000003000100FEFF00000100FFFF0800FDFF0000FEFFFDFF0000FFFF0300020001000500FDFFFDFFFFFF00000600FDFF00000000FAFFFDFF00000800FDFF0000FFFF00000000FFFF0000000000000100FEFF000002000300F9FF04000200FDFF00000200030001000400FEFFFFFFFDFFFCFF"> : tensor<5x6x7xi16>
    %1 = stablehlo.constant dense<[[[0, 0], [-3, 0]], [[-3, 1], [-5, -1]], [[-4, 0], [3, -1]], [[-3, -1], [-1, 2]], [[1, 4], [-4, 0]]]> : tensor<5x2x2xi16>
    return %0, %1 : tensor<5x6x7xi16>, tensor<5x2x2xi16>
  }
  func.func private @expected() -> tensor<5x6x7xi16> {
    %0 = stablehlo.constant dense<"0x02000000FBFF00000100FDFF00000200FEFF0300010000000400FAFFFFFFFEFFFDFF00000000FFFF00000700FFFF0200000000000000F9FF06000200F9FF00000000FCFF0000FEFF010000000300FBFF0000000003000000020003000000FEFF0200F8FF0000050000000000FFFF000004000100020001000100FDFFFFFFFEFF00000100000004000000FDFFFDFFFBFFFDFF0000F9FF0300060002000000000000000200FFFF0300FDFF0000030000000000FDFF020001000600030002000300FBFF02000000000002000000FDFF0000FBFF02000000FFFF03000000000000000400000004000300FDFF0000000005000100FEFFFDFF010003000000FDFFFFFF000003000300000000000700FCFF020001000000FDFF0400000000000200090000000300FEFFFFFF0000040000000500FEFF000003000100FEFF00000100FFFF0800FDFF0000FEFFFDFF0000FFFF0300020001000500FDFFFDFFFFFF00000600FDFF00000000FAFFFDFF00000800FDFF0000040000000000FFFF0000000000000100FEFF000002000300F9FF04000200FDFF00000200030001000400FEFFFFFFFDFFFCFF"> : tensor<5x6x7xi16>
    return %0 : tensor<5x6x7xi16>
  }
}
