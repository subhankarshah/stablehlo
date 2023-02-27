module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui16>, tensor<2x7xui16>)
    %2 = call @expected() : () -> tensor<5x6x7xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui16>, tensor<2x2xi32>, tensor<2x7xui16>) -> tensor<5x6x7xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui16>, tensor<2x7xui16>) {
    %0 = stablehlo.constant dense<"0x000002000300050004000800000000000200020004000300020000000400030005000000020001000500010001000000020004000100000000000300010000000200020000000100030003000000000002000500010001000100020007000000040002000000030001000500000000000300050000000100050000000100010003000300030002000200020000000100000000000000050000000300000002000400050000000300030004000700020001000000040000000000020005000100010002000400040002000100030003000100020001000100050002000300000001000000000000000100000002000000000000000200000003000000020003000100040003000200010001000200020002000400000001000200030001000300040004000100020001000000000000000300010000000100010002000300020002000000040000000200000000000000070001000800020001000100030000000000010002000000000000000300030002000100010004000200010000000100020005000400040001000300010003000000070002000600090001000000050000000100"> : tensor<5x6x7xui16>
    %1 = stablehlo.constant dense<[[2, 1, 2, 1, 0, 2, 3], [0, 0, 0, 8, 4, 3, 4]]> : tensor<2x7xui16>
    return %0, %1 : tensor<5x6x7xui16>, tensor<2x7xui16>
  }
  func.func private @expected() -> tensor<5x6x7xui16> {
    %0 = stablehlo.constant dense<"0x000002000300050004000800000000000200040004000000040000000400030005000000020001000500010001000000020004000100000000000300010000000200020000000100030003000000000002000500010001000100020007000000040002000000030001000500000000000300050000000100050000000100010003000300030002000200020000000100000000000000050000000300000002000400050000000300030004000700020001000000040000000000020005000100010002000400040002000100030003000100000000000000280008000900000001000000000000000100000002000000000000000200000003000000020003000100040003000200010001000200020002000400000001000200030001000300040004000100020001000000000000000300010000000100010002000300020002000000040000000200000000000000070001000800020001000100030000000000010002000000000000000300030002000100010004000200010000000100020005000400040001000300010003000000070002000600090001000000050000000100"> : tensor<5x6x7xui16>
    return %0 : tensor<5x6x7xui16>
  }
}
