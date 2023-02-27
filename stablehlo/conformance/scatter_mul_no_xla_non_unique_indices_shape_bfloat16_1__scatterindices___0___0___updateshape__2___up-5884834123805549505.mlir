module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<1xbf16>, tensor<2xbf16>)
    %2 = call @expected() : () -> tensor<1xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<1xbf16>, tensor<2x1xi32>, tensor<2xbf16>) -> tensor<1xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<-3.984380e+00> : tensor<1xbf16>
    %1 = stablehlo.constant dense<[2.421880e+00, 3.328130e+00]> : tensor<2xbf16>
    return %0, %1 : tensor<1xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<1xbf16> {
    %0 = stablehlo.constant dense<-3.200000e+01> : tensor<1xbf16>
    return %0 : tensor<1xbf16>
  }
}
