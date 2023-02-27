module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi32>, tensor<2xi32>, tensor<4x3xi32>) -> tensor<4x2x3x5xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi32>, tensor<4x2x3x5xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>) {
    %0 = stablehlo.constant dense<"0x0000000003000000FEFFFFFFFCFFFFFFFFFFFFFFFBFFFFFF03000000FDFFFFFFFFFFFFFF03000000FCFFFFFFFEFFFFFF010000000000000000000000FEFFFFFF0000000003000000FBFFFFFF00000000FFFFFFFFFBFFFFFF0500000000000000000000000000000000000000010000000200000000000000FDFFFFFFFAFFFFFF03000000050000000000000000000000FEFFFFFF030000000000000002000000000000000000000002000000FFFFFFFFFCFFFFFF01000000FDFFFFFFFDFFFFFF0300000000000000010000000100000000000000FDFFFFFFFFFFFFFF040000000100000001000000FFFFFFFF00000000FAFFFFFFFEFFFFFF050000000300000000000000FFFFFFFF020000000000000007000000FEFFFFFF0300000000000000FDFFFFFFFAFFFFFF010000000100000000000000000000000400000003000000FFFFFFFF02000000010000000000000000000000FCFFFFFF01000000020000000100000000000000FFFFFFFF000000000200000001000000FFFFFFFF02000000FBFFFFFFF8FFFFFF020000000700000000000000030000000000000001000000FFFFFFFFFAFFFFFF0300000001000000FFFFFFFF0000000000000000FAFFFFFF02000000010000000300000002000000020000000000000003000000FEFFFFFF"> : tensor<4x2x3x5xi32>
    %1 = stablehlo.constant dense<[[-3, 1, 0], [0, -1, 2], [-1, 0, -1], [4, 0, -3]]> : tensor<4x3xi32>
    return %0, %1 : tensor<4x2x3x5xi32>, tensor<4x3xi32>
  }
  func.func private @expected() -> tensor<4x2x3x5xi32> {
    %0 = stablehlo.constant dense<"0x0000000003000000FEFFFFFFFCFFFFFFFDFFFFFFFBFFFFFF03000000FDFFFFFFFFFFFFFF01000000FCFFFFFFFEFFFFFF010000000000000000000000FEFFFFFF0000000003000000FBFFFFFF00000000FFFFFFFFFBFFFFFF0500000000000000000000000000000000000000010000000200000000000000FDFFFFFFFAFFFFFF03000000050000000000000000000000FEFFFFFF0300000000000000FFFFFFFF000000000000000002000000FFFFFFFFFCFFFFFF01000000FDFFFFFFFDFFFFFF0300000000000000010000000100000000000000FDFFFFFFFFFFFFFF040000000100000001000000FFFFFFFF00000000FAFFFFFFFEFFFFFF0500000003000000FFFFFFFFFFFFFFFF020000000000000007000000FEFFFFFF0300000000000000FDFFFFFFFAFFFFFFFFFFFFFF0100000000000000000000000400000003000000FFFFFFFF02000000010000000000000000000000FCFFFFFF01000000020000000100000000000000FFFFFFFF000000000200000001000000FFFFFFFF02000000FBFFFFFFF8FFFFFF020000000000000000000000030000000000000001000000FDFFFFFFFAFFFFFF0300000001000000FFFFFFFF0000000000000000FAFFFFFF02000000010000000300000002000000020000000000000003000000FEFFFFFF"> : tensor<4x2x3x5xi32>
    return %0 : tensor<4x2x3x5xi32>
  }
}
