module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<2xui32>)
    %1 = call @expected() : () -> tensor<2xui32>
    %2 = stablehlo.remainder %0#0, %0#1 : tensor<2xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<2xui32>) {
    %0 = stablehlo.constant dense<[2, 0]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[2, 1]> : tensor<2xui32>
    return %0, %1 : tensor<2xui32>, tensor<2xui32>
  }
  func.func private @expected() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<0> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
}
