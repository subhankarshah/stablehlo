module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xui32>, tensor<3x5x2xui32>)
    %2 = call @expected() : () -> tensor<3x5x40xui32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui32>
      stablehlo.return %5 : tensor<ui32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false} : (tensor<3x5x40xui32>, tensor<2x1xi32>, tensor<3x5x2xui32>) -> tensor<3x5x40xui32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xui32>, tensor<3x5x40xui32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xui32>, tensor<3x5x2xui32>) {
    %0 = stablehlo.constant dense<"0x020000000200000003000000000000000400000003000000030000000400000002000000040000000200000002000000010000000000000004000000020000000100000000000000010000000200000000000000000000000000000000000000020000000500000001000000040000000500000004000000000000000400000001000000010000000100000003000000050000000000000005000000010000000200000000000000030000000600000006000000030000000000000000000000020000000000000001000000000000000300000001000000000000000000000000000000010000000800000001000000000000000100000001000000000000000500000003000000030000000300000008000000000000000100000000000000000000000300000005000000010000000000000001000000060000000500000002000000010000000100000000000000030000000000000002000000010000000400000002000000000000000600000002000000030000000300000005000000010000000600000002000000000000000100000001000000020000000100000005000000030000000200000003000000010000000100000001000000010000000000000000000000010000000000000002000000010000000200000005000000020000000100000000000000000000000100000001000000010000000000000000000000020000000600000002000000000000000400000004000000060000000100000002000000070000000100000000000000040000000300000000000000000000000500000001000000030000000300000002000000060000000200000002000000040000000400000000000000000000000200000009000000020000000300000003000000010000000700000002000000080000000300000000000000010000000400000006000000010000000000000004000000010000000300000000000000030000000000000000000000000000000300000003000000020000000400000002000000040000000300000000000000010000000600000002000000030000000500000000000000020000000200000000000000030000000000000002000000030000000000000000000000020000000400000002000000020000000600000001000000050000000400000005000000000000000100000003000000010000000000000001000000010000000300000001000000020000000400000004000000010000000600000000000000000000000000000002000000010000000100000000000000000000000500000001000000020000000100000000000000000000000100000000000000020000000000000000000000030000000400000002000000020000000400000000000000030000000000000005000000030000000100000000000000000000000400000004000000030000000100000002000000020000000100000000000000010000000100000000000000010000000100000000000000050000000200000003000000000000000000000001000000030000000000000005000000000000000000000002000000020000000300000001000000060000000200000003000000020000000200000001000000020000000400000002000000020000000000000004000000030000000000000002000000000000000100000003000000020000000200000001000000020000000000000001000000000000000000000001000000020000000200000002000000030000000400000001000000000000000000000002000000020000000300000000000000000000000100000003000000040000000100000000000000040000000200000002000000000000000000000001000000020000000100000001000000020000000100000002000000040000000000000001000000000000000200000000000000040000000300000001000000040000000500000002000000010000000200000001000000040000000000000001000000020000000000000002000000020000000000000004000000000000000200000001000000010000000000000000000000000000000100000001000000040000000000000001000000000000000000000000000000040000000000000000000000010000000100000002000000040000000700000003000000010000000100000004000000020000000200000002000000030000000100000003000000020000000300000002000000010000000300000003000000030000000600000000000000010000000500000000000000020000000300000000000000040000000200000001000000030000000200000001000000000000000100000006000000000000000100000000000000000000000200000004000000010000000300000000000000020000000200000001000000020000000300000004000000020000000200000001000000030000000200000002000000000000000800000002000000000000000100000003000000010000000300000000000000000000000000000002000000050000000300000001000000020000000200000001000000000000000100000001000000000000000000000004000000020000000200000005000000010000000200000000000000000000000200000004000000010000000200000002000000050000000400000000000000070000000000000002000000000000000200000001000000010000000200000003000000040000000100000001000000030000000000000002000000010000000100000001000000030000000200000003000000020000000300000002000000030000000200000000000000040000000000000004000000020000000500000001000000030000000300000001000000000000000000000005000000040000000200000005000000040000000000000003000000000000000100000001000000040000000700000001000000040000000100000003000000060000000400000004000000050000000000000003000000030000000000000001000000080000000100000003000000020000000400000000000000020000000100000002000000010000000100000000000000000000000100000003000000010000000200000002000000010000000400000003000000020000000500000000000000030000000000000006000000010000000100000002000000050000000100000003000000020000000300000001000000050000000200000000000000000000000600000004000000020000000100000004000000020000000000000001000000010000000100000002000000"> : tensor<3x5x40xui32>
    %1 = stablehlo.constant dense<[[[1, 2], [0, 2], [2, 3], [1, 2], [0, 2]], [[3, 2], [1, 0], [0, 3], [2, 3], [0, 3]], [[0, 1], [4, 2], [0, 3], [3, 2], [2, 4]]]> : tensor<3x5x2xui32>
    return %0, %1 : tensor<3x5x40xui32>, tensor<3x5x2xui32>
  }
  func.func private @expected() -> tensor<3x5x40xui32> {
    %0 = stablehlo.constant dense<"0x020000000200000003000000000000000400000003000000030000000400000002000000040000000200000002000000010000000000000004000000020000000100000000000000010000000200000000000000000000000000000000000000020000000500000001000000040000000500000004000000000000000400000001000000010000000100000003000000050000000000000005000000010000000200000002000000030000000600000006000000030000000000000000000000020000000000000001000000000000000300000001000000000000000000000000000000010000000800000001000000000000000100000001000000000000000500000003000000030000000300000008000000000000000100000000000000000000000300000005000000010000000000000001000000060000000500000002000000030000000100000000000000030000000000000002000000010000000400000002000000000000000600000002000000030000000300000005000000010000000600000002000000000000000100000001000000020000000100000005000000030000000200000003000000010000000100000001000000010000000000000000000000010000000000000002000000010000000200000005000000020000000200000000000000000000000100000001000000010000000000000000000000020000000600000002000000000000000400000004000000060000000100000002000000070000000100000000000000040000000300000000000000000000000500000001000000030000000300000002000000060000000200000002000000040000000400000000000000000000000200000009000000020000000300000003000000010000000700000002000000080000000300000000000000010000000400000006000000010000000000000004000000010000000300000000000000030000000000000000000000000000000300000003000000020000000400000002000000040000000300000000000000010000000600000002000000030000000500000000000000020000000200000000000000030000000000000002000000030000000000000000000000020000000400000002000000020000000600000001000000050000000400000005000000000000000100000003000000010000000000000001000000010000000300000001000000020000000400000004000000010000000600000000000000000000000000000002000000010000000100000000000000000000000500000001000000020000000100000000000000000000000100000000000000020000000000000000000000030000000400000002000000020000000400000000000000030000000000000005000000030000000100000000000000000000000400000004000000030000000100000002000000020000000100000000000000010000000100000000000000010000000100000000000000050000000200000003000000000000000000000001000000030000000000000005000000000000000000000002000000020000000300000001000000060000000200000003000000020000000200000001000000020000000400000002000000020000000000000004000000030000000000000002000000000000000100000003000000020000000200000001000000020000000000000001000000000000000000000001000000020000000200000002000000030000000400000001000000030000000000000002000000020000000300000000000000000000000100000003000000040000000100000000000000040000000200000002000000000000000000000001000000020000000100000001000000020000000100000002000000040000000000000001000000000000000200000000000000040000000300000001000000040000000500000002000000010000000200000001000000040000000300000001000000020000000000000002000000020000000000000004000000000000000200000001000000010000000000000000000000000000000100000001000000040000000000000001000000000000000000000000000000040000000000000000000000010000000100000002000000040000000700000003000000010000000100000004000000020000000200000002000000030000000100000003000000020000000300000002000000010000000300000003000000030000000600000000000000010000000500000000000000020000000300000000000000040000000200000001000000030000000200000001000000000000000100000006000000000000000100000000000000000000000200000004000000010000000300000000000000020000000200000001000000020000000300000004000000040000000200000001000000030000000200000002000000000000000800000002000000000000000100000003000000010000000300000000000000000000000000000002000000050000000300000001000000020000000200000001000000000000000100000001000000000000000000000004000000020000000200000005000000010000000200000000000000000000000200000004000000010000000300000002000000050000000400000000000000070000000000000002000000000000000200000001000000010000000200000003000000040000000100000001000000030000000000000002000000010000000100000001000000030000000200000003000000020000000300000002000000030000000200000000000000040000000000000004000000020000000500000001000000030000000300000003000000000000000000000005000000040000000200000005000000040000000000000003000000000000000100000001000000040000000700000001000000040000000100000003000000060000000400000004000000050000000000000003000000030000000000000001000000080000000100000003000000020000000400000000000000020000000100000002000000010000000100000000000000040000000100000003000000010000000200000002000000010000000400000003000000020000000500000000000000030000000000000006000000010000000100000002000000050000000100000003000000020000000300000001000000050000000200000000000000000000000600000004000000020000000100000004000000020000000000000001000000010000000100000002000000"> : tensor<3x5x40xui32>
    return %0 : tensor<3x5x40xui32>
  }
}
