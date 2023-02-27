module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x1xf32>, tensor<1xf32>)
    %2 = call @expected() : () -> tensor<1x1xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<1x1xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x1xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x1xf32>, tensor<1xf32>) {
    %0 = stablehlo.constant dense<-1.34357524> : tensor<1x1xf32>
    %1 = stablehlo.constant dense<0.287947088> : tensor<1xf32>
    return %0, %1 : tensor<1x1xf32>, tensor<1xf32>
  }
  func.func private @expected() -> tensor<1x1xf32> {
    %0 = stablehlo.constant dense<-1.05562818> : tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
}
