module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<complex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.real %0#0 : (tensor<complex<f32>>) -> tensor<f32>
    %3 = stablehlo.real %0#1 : (tensor<complex<f32>>) -> tensor<f32>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.imag %0#0 : (tensor<complex<f32>>) -> tensor<f32>
    %6 = stablehlo.imag %0#1 : (tensor<complex<f32>>) -> tensor<f32>
    %7 = stablehlo.compare  LE, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = stablehlo.compare  LE, %2, %3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.select %4, %7, %8 : tensor<i1>, tensor<i1>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<complex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<(1.99115109,-2.57805085)> : tensor<complex<f32>>
    %1 = stablehlo.constant dense<(-3.47288561,-1.91607642)> : tensor<complex<f32>>
    return %0, %1 : tensor<complex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<i1> {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    return %0 : tensor<i1>
  }
}
