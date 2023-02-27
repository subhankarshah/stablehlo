module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<100x100xi1>
    %1 = call @expected() : () -> tensor<100x100xcomplex<f32>>
    %2 = stablehlo.convert %0 : (tensor<100x100xi1>) -> tensor<100x100xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<100x100xcomplex<f32>>, tensor<100x100xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<100x100xi1> {
    %0 = stablehlo.constant dense<true> : tensor<100x100xi1>
    return %0 : tensor<100x100xi1>
  }
  func.func private @expected() -> tensor<100x100xcomplex<f32>> {
    %0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<100x100xcomplex<f32>>
    return %0 : tensor<100x100xcomplex<f32>>
  }
}
