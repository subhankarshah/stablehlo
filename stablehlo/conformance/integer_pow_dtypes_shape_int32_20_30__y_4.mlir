module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x30xi32>
    %1 = call @expected() : () -> tensor<20x30xi32>
    %2 = call @integer_pow(%0) : (tensor<20x30xi32>) -> tensor<20x30xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x30xi32>, tensor<20x30xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x30xi32> {
    %0 = stablehlo.constant dense<"0x010000000300000001000000010000000000000000000000FCFFFFFF000000000000000003000000010000000000000003000000FCFFFFFF0000000001000000FFFFFFFF000000000000000001000000000000000000000000000000000000000000000005000000FBFFFFFFFFFFFFFFFFFFFFFF03000000010000000000000000000000FFFFFFFF04000000FCFFFFFF030000000400000000000000FEFFFFFFFCFFFFFF0200000006000000FEFFFFFF000000000500000000000000000000000200000002000000FCFFFFFF00000000FDFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF01000000000000000000000003000000F7FFFFFF0500000001000000000000000100000000000000FDFFFFFF0000000002000000FDFFFFFF00000000FDFFFFFFFFFFFFFF01000000FEFFFFFF0200000000000000FDFFFFFF030000000000000000000000FFFFFFFF000000000100000003000000FAFFFFFFFEFFFFFF0300000002000000FEFFFFFFFBFFFFFFFFFFFFFF07000000FFFFFFFF0600000000000000FBFFFFFFFFFFFFFF010000000300000003000000000000000400000001000000FDFFFFFF020000000100000001000000FEFFFFFFFCFFFFFF02000000FBFFFFFF07000000FFFFFFFF030000000000000002000000FFFFFFFF03000000FAFFFFFF0000000004000000000000000000000000000000FFFFFFFF01000000FEFFFFFFFDFFFFFFFAFFFFFF00000000FCFFFFFF00000000FDFFFFFFFDFFFFFFFFFFFFFF0000000000000000FEFFFFFFFCFFFFFFFEFFFFFF0000000000000000FDFFFFFF01000000FDFFFFFFFFFFFFFF0300000000000000FFFFFFFF0600000002000000030000000100000000000000FDFFFFFFFFFFFFFFFDFFFFFF02000000FEFFFFFF00000000FEFFFFFF0200000000000000FDFFFFFF020000000000000005000000FFFFFFFF00000000FFFFFFFF050000000500000000000000030000000000000000000000000000000300000005000000FCFFFFFFFFFFFFFFFDFFFFFF0300000001000000FDFFFFFFFDFFFFFF00000000FFFFFFFFFCFFFFFF000000000A0000000200000001000000FCFFFFFF0100000002000000FEFFFFFF000000000100000000000000FFFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFBFFFFFF00000000FBFFFFFF00000000000000000100000000000000000000000500000004000000FEFFFFFF000000000000000000000000FCFFFFFF0000000000000000000000000500000001000000FFFFFFFF0300000002000000030000000000000006000000FDFFFFFFFEFFFFFF03000000FEFFFFFF0000000002000000FEFFFFFFFDFFFFFFFCFFFFFF020000000200000004000000FCFFFFFF00000000FFFFFFFFFEFFFFFF0400000000000000FEFFFFFF01000000FFFFFFFF0200000003000000FFFFFFFF02000000FFFFFFFF0200000002000000000000000300000001000000FDFFFFFF02000000FDFFFFFF0200000002000000FBFFFFFFFEFFFFFFFEFFFFFF0300000004000000FFFFFFFF0800000000000000FEFFFFFF04000000050000000000000000000000000000000100000002000000FEFFFFFFFEFFFFFF0000000001000000FDFFFFFF0300000000000000050000000200000004000000000000000000000002000000000000000000000004000000FBFFFFFFFEFFFFFFFEFFFFFFFDFFFFFF0000000001000000FCFFFFFF00000000020000000100000001000000000000000000000003000000FFFFFFFFFFFFFFFF0000000004000000FFFFFFFF000000000000000000000000FAFFFFFF00000000FFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFF01000000FFFFFFFF0300000000000000FEFFFFFF06000000FAFFFFFF02000000030000000100000000000000FDFFFFFF00000000FBFFFFFF01000000FEFFFFFF02000000040000000000000000000000FFFFFFFF0300000000000000020000000400000003000000FFFFFFFFFFFFFFFF02000000FEFFFFFFF9FFFFFF02000000030000000300000000000000000000000100000000000000FFFFFFFFFFFFFFFF00000000010000000100000001000000FCFFFFFFFDFFFFFF0200000002000000FEFFFFFF0000000001000000FEFFFFFF05000000FCFFFFFFFBFFFFFFFEFFFFFF010000000000000000000000F9FFFFFF0100000002000000FDFFFFFF000000000200000004000000000000000400000001000000FFFFFFFFFFFFFFFFFEFFFFFF01000000FEFFFFFFFEFFFFFF0000000000000000FDFFFFFF0000000009000000000000000400000000000000030000000300000000000000FEFFFFFF0200000001000000FFFFFFFF030000000100000001000000FFFFFFFF0200000000000000FDFFFFFFFFFFFFFF0000000005000000FFFFFFFFFDFFFFFFFDFFFFFF00000000FCFFFFFF00000000FFFFFFFFFBFFFFFF04000000FDFFFFFFFFFFFFFF00000000FFFFFFFF05000000010000000100000002000000000000000000000002000000FCFFFFFF0000000000000000FFFFFFFF00000000FEFFFFFFFCFFFFFFFFFFFFFFFDFFFFFF02000000FFFFFFFF03000000FFFFFFFFFAFFFFFF0200000002000000FEFFFFFF020000000100000000000000FCFFFFFF00000000000000000100000000000000000000000000000001000000FFFFFFFF0000000000000000FFFFFFFFFCFFFFFFFCFFFFFF0000000000000000FFFFFFFF0000000000000000FDFFFFFF03000000FFFFFFFFFFFFFFFF03000000FCFFFFFF0000000000000000FEFFFFFFFDFFFFFF000000000500000000000000020000000200000005000000FEFFFFFF03000000010000000100000006000000000000000300000004000000FDFFFFFFFFFFFFFF0300000002000000FBFFFFFF00000000010000000200000003000000030000000500000000000000FFFFFFFF00000000FEFFFFFF04000000050000000000000003000000FFFFFFFFFDFFFFFFFCFFFFFF00000000070000000000000000000000F9FFFFFF03000000000000000400000000000000FEFFFFFF02000000FFFFFFFFFBFFFFFF0400000001000000FDFFFFFF0000000004000000FDFFFFFF03000000FFFFFFFF03000000FFFFFFFFFDFFFFFF000000000000000000000000FEFFFFFF0000000002000000FEFFFFFF06000000000000000100000002000000FFFFFFFFFEFFFFFF03000000FFFFFFFFFDFFFFFF0000000003000000FBFFFFFF0000000000000000010000000000000001000000FCFFFFFFFDFFFFFFFFFFFFFF0200000001000000FEFFFFFF000000000400000004000000060000000100000000000000020000000200000003000000030000000200000001000000"> : tensor<20x30xi32>
    return %0 : tensor<20x30xi32>
  }
  func.func private @expected() -> tensor<20x30xi32> {
    %0 = stablehlo.constant dense<"0x01000000510000000100000001000000000000000000000000010000000000000000000051000000010000000000000051000000000100000000000001000000010000000000000000000000010000000000000000000000000000000000000000000000710200007102000001000000010000005100000001000000000000000000000001000000000100000001000051000000000100000000000010000000000100001000000010050000100000000000000071020000000000000000000010000000100000000001000000000000510000001000000001000000100000000100000001000000000000000000000051000000A119000071020000010000000000000001000000000000005100000000000000100000005100000000000000510000000100000001000000100000001000000000000000510000005100000000000000000000000100000000000000010000005100000010050000100000005100000010000000100000007102000001000000610900000100000010050000000000007102000001000000010000005100000051000000000000000001000001000000510000001000000001000000010000001000000000010000100000007102000061090000010000005100000000000000100000000100000051000000100500000000000000010000000000000000000000000000010000000100000010000000510000001005000000000000000100000000000051000000510000000100000000000000000000001000000000010000100000000000000000000000510000000100000051000000010000005100000000000000010000001005000010000000510000000100000000000000510000000100000051000000100000001000000000000000100000001000000000000000510000001000000000000000710200000100000000000000010000007102000071020000000000005100000000000000000000000000000051000000710200000001000001000000510000005100000001000000510000005100000000000000010000000001000000000000102700001000000001000000000100000100000010000000100000000000000001000000000000000100000010000000010000007102000071020000000000007102000000000000000000000100000000000000000000007102000000010000100000000000000000000000000000000001000000000000000000000000000071020000010000000100000051000000100000005100000000000000100500005100000010000000510000001000000000000000100000001000000051000000000100001000000010000000000100000001000000000000010000001000000000010000000000001000000001000000010000001000000051000000010000001000000001000000100000001000000000000000510000000100000051000000100000005100000010000000100000007102000010000000100000005100000000010000010000000010000000000000100000000001000071020000000000000000000000000000010000001000000010000000100000000000000001000000510000005100000000000000710200001000000000010000000000000000000010000000000000000000000000010000710200001000000010000000510000000000000001000000000100000000000010000000010000000100000000000000000000005100000001000000010000000000000000010000010000000000000000000000000000001005000000000000010000000100000001000000000000000100000001000000010000005100000000000000100000001005000010050000100000005100000001000000000000005100000000000000710200000100000010000000100000000001000000000000000000000100000051000000000000001000000000010000510000000100000001000000100000001000000061090000100000005100000051000000000000000000000001000000000000000100000001000000000000000100000001000000010000000001000051000000100000001000000010000000000000000100000010000000710200000001000071020000100000000100000000000000000000006109000001000000100000005100000000000000100000000001000000000000000100000100000001000000010000001000000001000000100000001000000000000000000000005100000000000000A11900000000000000010000000000005100000051000000000000001000000010000000010000000100000051000000010000000100000001000000100000000000000051000000010000000000000071020000010000005100000051000000000000000001000000000000010000007102000000010000510000000100000000000000010000007102000001000000010000001000000000000000000000001000000000010000000000000000000001000000000000001000000000010000010000005100000010000000010000005100000001000000100500001000000010000000100000001000000001000000000000000001000000000000000000000100000000000000000000000000000001000000010000000000000000000000010000000001000000010000000000000000000001000000000000000000000051000000510000000100000001000000510000000001000000000000000000001000000051000000000000007102000000000000100000001000000071020000100000005100000001000000010000001005000000000000510000000001000051000000010000005100000010000000710200000000000001000000100000005100000051000000710200000000000001000000000000001000000000010000710200000000000051000000010000005100000000010000000000006109000000000000000000006109000051000000000000000001000000000000100000001000000001000000710200000001000001000000510000000000000000010000510000005100000001000000510000000100000051000000000000000000000000000000100000000000000010000000100000001005000000000000010000001000000001000000100000005100000001000000510000000000000051000000710200000000000000000000010000000000000001000000000100005100000001000000100000000100000010000000000000000001000000010000100500000100000000000000100000001000000051000000510000001000000001000000"> : tensor<20x30xi32>
    return %0 : tensor<20x30xi32>
  }
  func.func private @integer_pow(%arg0: tensor<20x30xi32>) -> tensor<20x30xi32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<20x30xi32>
    %1 = stablehlo.multiply %0, %0 : tensor<20x30xi32>
    return %1 : tensor<20x30xi32>
  }
}
