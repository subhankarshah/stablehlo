module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @expected() : () -> tensor<2x3xui32>
    %1 = stablehlo.iota dim = 0 : tensor<2x3xui32>
    %2 = stablehlo.custom_call @check.eq(%1, %0) : (tensor<2x3xui32>, tensor<2x3xui32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @expected() -> tensor<2x3xui32> {
    %0 = stablehlo.constant dense<[[0, 0, 0], [1, 1, 1]]> : tensor<2x3xui32>
    return %0 : tensor<2x3xui32>
  }
}
