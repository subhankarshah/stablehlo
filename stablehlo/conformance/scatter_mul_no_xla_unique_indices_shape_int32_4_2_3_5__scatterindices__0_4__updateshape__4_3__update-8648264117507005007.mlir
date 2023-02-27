module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi32>, tensor<2xi32>, tensor<4x3xi32>) -> tensor<4x2x3x5xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi32>, tensor<4x2x3x5xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>) {
    %0 = stablehlo.constant dense<"0x01000000FFFFFFFFFEFFFFFF0100000000000000FFFFFFFF0000000000000000FEFFFFFFFCFFFFFF03000000010000000300000002000000FAFFFFFF00000000FFFFFFFFFFFFFFFF03000000FEFFFFFFFCFFFFFFFDFFFFFF020000000300000000000000FFFFFFFF0000000002000000FCFFFFFF0100000000000000FDFFFFFF0600000006000000F9FFFFFF0100000001000000FBFFFFFF0000000000000000FDFFFFFF010000000300000003000000FEFFFFFFFFFFFFFFFBFFFFFF0100000000000000FBFFFFFF00000000050000000000000000000000FDFFFFFFFFFFFFFFFAFFFFFFFEFFFFFF0300000002000000FCFFFFFFFEFFFFFF01000000FEFFFFFFFCFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFCFFFFFFFDFFFFFF0100000000000000FDFFFFFF07000000FBFFFFFF000000000400000000000000FCFFFFFFFFFFFFFFFCFFFFFF00000000020000000000000002000000FBFFFFFF0400000000000000000000000200000000000000FFFFFFFF0000000000000000FDFFFFFF02000000FFFFFFFF000000000000000000000000FCFFFFFF00000000000000000300000001000000FFFFFFFFFEFFFFFF0300000002000000FFFFFFFF0000000000000000FDFFFFFFFFFFFFFF0500000002000000F9FFFFFF000000000000000000000000"> : tensor<4x2x3x5xi32>
    %1 = stablehlo.constant dense<[[-3, -2, -5], [3, 0, -2], [5, 4, 0], [-2, 1, 0]]> : tensor<4x3xi32>
    return %0, %1 : tensor<4x2x3x5xi32>, tensor<4x3xi32>
  }
  func.func private @expected() -> tensor<4x2x3x5xi32> {
    %0 = stablehlo.constant dense<"0x01000000FFFFFFFFFEFFFFFF0100000000000000FFFFFFFF0000000000000000FEFFFFFF08000000030000000100000003000000020000001E00000000000000FFFFFFFFFFFFFFFF03000000FEFFFFFFFCFFFFFFFDFFFFFF020000000300000000000000FFFFFFFF0000000002000000FCFFFFFF0100000000000000FDFFFFFF0600000006000000EBFFFFFF0100000001000000FBFFFFFF0000000000000000FDFFFFFF01000000030000000300000004000000FFFFFFFFFBFFFFFF0100000000000000FBFFFFFF00000000050000000000000000000000FDFFFFFFFFFFFFFFFAFFFFFFFEFFFFFF0300000002000000FCFFFFFFFEFFFFFF01000000FEFFFFFFECFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFCFFFFFFF4FFFFFF0100000000000000FDFFFFFF0700000000000000000000000400000000000000FCFFFFFFFFFFFFFFFCFFFFFF00000000020000000000000002000000FBFFFFFF0400000000000000000000000200000000000000FFFFFFFF00000000000000000600000002000000FFFFFFFF000000000000000000000000FCFFFFFF00000000000000000300000000000000FFFFFFFFFEFFFFFF0300000002000000FFFFFFFF0000000000000000FDFFFFFFFFFFFFFF0500000002000000F9FFFFFF000000000000000000000000"> : tensor<4x2x3x5xi32>
    return %0 : tensor<4x2x3x5xi32>
  }
}
