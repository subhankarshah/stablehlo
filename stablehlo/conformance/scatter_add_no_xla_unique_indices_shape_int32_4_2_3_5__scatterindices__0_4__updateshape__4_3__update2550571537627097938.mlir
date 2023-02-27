module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi32>, tensor<2xi32>, tensor<4x3xi32>) -> tensor<4x2x3x5xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi32>, tensor<4x2x3x5xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>) {
    %0 = stablehlo.constant dense<"0x0000000000000000000000000000000000000000000000000000000000000000FEFFFFFF000000000000000000000000FFFFFFFF01000000FFFFFFFF05000000FDFFFFFF0000000002000000040000000000000002000000FFFFFFFF0100000004000000010000000100000001000000000000000100000000000000020000000200000000000000010000000000000002000000FFFFFFFF0000000000000000000000000000000002000000000000000200000000000000FDFFFFFF03000000FCFFFFFFFFFFFFFFFDFFFFFFFEFFFFFFFBFFFFFF000000000400000000000000FFFFFFFFFCFFFFFFFEFFFFFFFCFFFFFF00000000060000000000000000000000FEFFFFFFFEFFFFFFFEFFFFFF00000000000000000100000000000000010000000000000001000000FBFFFFFF04000000FFFFFFFFFEFFFFFF03000000FFFFFFFF00000000FFFFFFFF01000000FCFFFFFFFEFFFFFF01000000000000000000000003000000FFFFFFFFFDFFFFFFFFFFFFFF00000000FBFFFFFFFFFFFFFF00000000010000000300000000000000FAFFFFFFFEFFFFFF000000000200000000000000FFFFFFFF01000000000000000000000004000000FBFFFFFF03000000FEFFFFFF0000000000000000FBFFFFFF000000000300000000000000FFFFFFFF00000000"> : tensor<4x2x3x5xi32>
    %1 = stablehlo.constant dense<[[0, 1, 1], [-4, 1, -3], [2, -4, 4], [0, 0, 0]]> : tensor<4x3xi32>
    return %0, %1 : tensor<4x2x3x5xi32>, tensor<4x3xi32>
  }
  func.func private @expected() -> tensor<4x2x3x5xi32> {
    %0 = stablehlo.constant dense<"0x0000000000000000000000000000000000000000000000000000000000000000FEFFFFFF010000000000000000000000FFFFFFFF010000000000000005000000FDFFFFFF0000000002000000040000000000000002000000FFFFFFFF0100000004000000010000000100000001000000000000000100000000000000020000000200000000000000FDFFFFFF0000000002000000FFFFFFFF000000000100000000000000000000000200000000000000FFFFFFFF00000000FDFFFFFF03000000FCFFFFFFFFFFFFFFFDFFFFFFFEFFFFFFFBFFFFFF000000000400000000000000FFFFFFFFFCFFFFFFFEFFFFFFFCFFFFFF0000000006000000000000000000000000000000FEFFFFFFFEFFFFFF0000000000000000FDFFFFFF00000000010000000000000001000000FFFFFFFF04000000FFFFFFFFFEFFFFFF03000000FFFFFFFF00000000FFFFFFFF01000000FCFFFFFFFEFFFFFF01000000000000000000000003000000FFFFFFFFFDFFFFFFFFFFFFFF00000000FBFFFFFFFFFFFFFF00000000010000000300000000000000FAFFFFFFFEFFFFFF000000000200000000000000FFFFFFFF01000000000000000000000004000000FBFFFFFF03000000FEFFFFFF0000000000000000FBFFFFFF000000000300000000000000FFFFFFFF00000000"> : tensor<4x2x3x5xi32>
    return %0 : tensor<4x2x3x5xi32>
  }
}
