module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x2x3xi16>, tensor<1x3xi16>)
    %2 = call @expected() : () -> tensor<1x2x3xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x2x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x2x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x2x3xi16>, tensor<1x2x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<[[[-6, -2, 1], [2, 1, 1]]]> : tensor<1x2x3xi16>
    %1 = stablehlo.constant dense<[[-2, 0, -2]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x2x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x2x3xi16> {
    %0 = stablehlo.constant dense<[[[-6, -2, 1], [-2, 0, -2]]]> : tensor<1x2x3xi16>
    return %0 : tensor<1x2x3xi16>
  }
}
