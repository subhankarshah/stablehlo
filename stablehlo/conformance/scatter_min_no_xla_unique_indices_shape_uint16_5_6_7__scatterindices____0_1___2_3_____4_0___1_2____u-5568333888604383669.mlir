module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui16>, tensor<5x2x2xui16>)
    %2 = call @expected() : () -> tensor<5x6x7xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui16>, tensor<2x2x2xi32>, tensor<5x2x2xui16>) -> tensor<5x6x7xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui16>, tensor<5x2x2xui16>) {
    %0 = stablehlo.constant dense<"0x060001000500000000000200060003000300000002000100000001000300020003000500010004000100030003000300010002000200030001000300010004000100030002000500040008000000030003000200020001000200020005000300060004000000010002000000030001000200040002000200030004000000000001000400000004000300040004000300010000000300010000000400030001000200050000000000050006000300010006000700000003000100050001000100020000000100000000000200030003000200000000000100040002000000020001000300000002000000040002000600010000000000000007000200020000000400030000000400030000000000020000000300020001000300060002000200000001000200030001000000010002000100040005000200020000000100000004000600040000000000030001000000010003000400000001000500010003000000010002000000020001000000000002000000020003000000040004000000000001000000050000000200000002000200000000000700040000000200020003000500"> : tensor<5x6x7xui16>
    %1 = stablehlo.constant dense<[[[2, 2], [1, 1]], [[0, 3], [0, 1]], [[1, 0], [1, 0]], [[4, 4], [7, 0]], [[0, 0], [1, 1]]]> : tensor<5x2x2xui16>
    return %0, %1 : tensor<5x6x7xui16>, tensor<5x2x2xui16>
  }
  func.func private @expected() -> tensor<5x6x7xui16> {
    %0 = stablehlo.constant dense<"0x060001000500000000000200060003000300000002000100000001000300020003000200010004000100030003000300010002000200030001000300010004000100030002000500040008000000030003000200020000000200020005000300060004000000010002000000030001000200040002000200030004000000000001000400000004000300040000000300010000000300010000000400030001000200050000000000050001000300010006000700000003000100000001000100020000000100000000000000030003000200000000000100040002000000020001000300000002000000040002000600010000000000000007000200020000000400030000000400030000000000000000000300020001000300060002000200000001000200030001000000010002000100040005000200020000000100000004000600040000000000030001000000010000000400000001000500010003000000010002000000020001000000000002000000020003000000040004000000000001000000050000000200000002000200000000000700040000000200020003000500"> : tensor<5x6x7xui16>
    return %0 : tensor<5x6x7xui16>
  }
}
