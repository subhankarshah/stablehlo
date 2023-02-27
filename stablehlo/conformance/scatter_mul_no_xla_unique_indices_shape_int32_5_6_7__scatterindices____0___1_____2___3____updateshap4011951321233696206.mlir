module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi32>, tensor<5x2x2x7xi32>)
    %2 = call @expected() : () -> tensor<5x6x7xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi32>, tensor<2x2x1xi32>, tensor<5x2x2x7xi32>) -> tensor<5x6x7xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi32>, tensor<5x6x7xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi32>, tensor<5x2x2x7xi32>) {
    %0 = stablehlo.constant dense<"0x0000000003000000010000000000000001000000FEFFFFFF06000000FDFFFFFFFCFFFFFF02000000FFFFFFFF01000000FFFFFFFFFEFFFFFF010000000000000000000000FEFFFFFF01000000FEFFFFFFFDFFFFFF05000000FEFFFFFF00000000FEFFFFFFFFFFFFFF0100000000000000FFFFFFFF0000000000000000000000000400000000000000FEFFFFFFFEFFFFFF00000000FFFFFFFFFEFFFFFF00000000FDFFFFFF050000000300000003000000040000000100000004000000010000000200000004000000000000000000000003000000000000000100000005000000FDFFFFFF000000000000000004000000FDFFFFFFFCFFFFFF01000000FFFFFFFF01000000FEFFFFFF0000000004000000FFFFFFFF00000000FEFFFFFF00000000000000000400000001000000FEFFFFFF0500000001000000FDFFFFFF02000000FFFFFFFF020000000000000006000000FDFFFFFFFBFFFFFF00000000FFFFFFFFFBFFFFFF00000000FFFFFFFFFBFFFFFFFFFFFFFF030000000000000001000000FFFFFFFFFDFFFFFF0500000002000000000000000000000000000000FCFFFFFFFEFFFFFFFDFFFFFF0000000001000000FEFFFFFF000000000100000004000000020000000000000002000000010000000200000001000000050000000500000000000000010000000000000000000000FBFFFFFF0000000002000000030000000400000001000000FDFFFFFFFEFFFFFF010000000200000001000000FCFFFFFF000000000200000002000000070000000000000003000000FDFFFFFF010000000800000000000000FEFFFFFFFFFFFFFF0100000003000000FEFFFFFF05000000FFFFFFFFFEFFFFFFFBFFFFFF0100000004000000FEFFFFFF0400000000000000FFFFFFFF0200000000000000FFFFFFFF020000000000000008000000FEFFFFFFFEFFFFFF00000000FFFFFFFFFEFFFFFFFEFFFFFF0000000002000000FFFFFFFF01000000010000000100000002000000000000000000000000000000FEFFFFFF0200000000000000FFFFFFFFFDFFFFFF01000000FEFFFFFF0100000001000000FDFFFFFFFEFFFFFFFFFFFFFF0100000000000000FFFFFFFF0200000003000000FFFFFFFF01000000FDFFFFFF01000000FEFFFFFF02000000FFFFFFFF04000000FDFFFFFF00000000"> : tensor<5x6x7xi32>
    %1 = stablehlo.constant dense<"0x00000000FEFFFFFF00000000FEFFFFFF02000000FEFFFFFF04000000F7FFFFFF0000000005000000010000000400000000000000FFFFFFFF00000000FFFFFFFF0200000005000000FCFFFFFFFDFFFFFF030000000000000000000000040000000100000000000000FAFFFFFFFBFFFFFF01000000FFFFFFFFFFFFFFFFFEFFFFFFFAFFFFFFFFFFFFFF03000000FFFFFFFFFFFFFFFF0000000001000000FDFFFFFF01000000FDFFFFFF000000000000000002000000FDFFFFFFFFFFFFFF000000000500000000000000040000000100000003000000FFFFFFFF0400000000000000FEFFFFFFFFFFFFFF00000000040000000100000004000000020000000000000005000000FFFFFFFFFBFFFFFF000000000500000001000000FAFFFFFFFDFFFFFF020000000000000000000000000000000000000003000000FEFFFFFF00000000FFFFFFFF00000000FDFFFFFFF9FFFFFFFDFFFFFF030000000300000001000000FFFFFFFF0000000001000000010000000000000000000000FDFFFFFF07000000010000000100000003000000FDFFFFFFFEFFFFFF000000000200000005000000FFFFFFFFFDFFFFFFFEFFFFFFFEFFFFFF0400000000000000FFFFFFFFFDFFFFFFFFFFFFFF00000000FEFFFFFFFEFFFFFF0000000005000000FFFFFFFFFCFFFFFFFFFFFFFF00000000F9FFFFFF05000000FEFFFFFF0600000000000000000000000400000003000000000000000300000002000000FFFFFFFF0300000006000000FCFFFFFF0200000000000000F7FFFFFF"> : tensor<5x2x2x7xi32>
    return %0, %1 : tensor<5x6x7xi32>, tensor<5x2x2x7xi32>
  }
  func.func private @expected() -> tensor<5x6x7xi32> {
    %0 = stablehlo.constant dense<"0x00000000FAFFFFFF00000000000000000200000004000000180000001B000000000000000A000000FFFFFFFF040000000000000002000000000000000000000000000000F6FFFFFFFCFFFFFF06000000F7FFFFFF000000000000000000000000FEFFFFFF00000000FAFFFFFF00000000FFFFFFFF0000000000000000000000000400000000000000FEFFFFFFFEFFFFFF00000000FFFFFFFFFEFFFFFF00000000FDFFFFFF0500000003000000FDFFFFFFFCFFFFFFFEFFFFFFE8FFFFFFFFFFFFFF06000000FCFFFFFF0000000000000000030000000000000001000000F1FFFFFF000000000000000000000000F4FFFFFF0300000000000000050000000000000004000000FEFFFFFF00000000FCFFFFFFFCFFFFFF00000000FEFFFFFF00000000000000000400000001000000FEFFFFFF0500000001000000FDFFFFFF02000000FFFFFFFF020000000000000006000000060000000500000000000000FCFFFFFFFBFFFFFF00000000FEFFFFFF00000000FBFFFFFFFDFFFFFF0000000000000000FBFFFFFFFDFFFFFFE2FFFFFFFAFFFFFF0000000000000000000000000000000000000000F7FFFFFF00000000000000000200000000000000FDFFFFFFE4FFFFFF020000000000000002000000010000000200000001000000050000000500000000000000010000000000000000000000FBFFFFFF00000000FAFFFFFF090000000C00000001000000030000000000000001000000020000000000000000000000000000000E000000020000000700000000000000F7FFFFFF060000000000000010000000000000000200000003000000FEFFFFFFFAFFFFFFF8FFFFFF000000000100000006000000FBFFFFFF0100000004000000FEFFFFFF0400000000000000FFFFFFFF0200000000000000FFFFFFFF020000000000000008000000FEFFFFFF020000000000000002000000040000000000000000000000FEFFFFFF04000000FFFFFFFF00000000F9FFFFFF0A00000000000000000000000000000000000000080000000000000000000000F7FFFFFF020000000200000003000000060000000C000000FCFFFFFF00000000F7FFFFFF00000000FFFFFFFF0200000003000000FFFFFFFF01000000FDFFFFFF01000000FEFFFFFF02000000FFFFFFFF04000000FDFFFFFF00000000"> : tensor<5x6x7xi32>
    return %0 : tensor<5x6x7xi32>
  }
}
