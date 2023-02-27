module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xi32>, tensor<20x20xi32>)
    %1 = call @expected() : () -> tensor<20x20xi32>
    %2 = stablehlo.subtract %0#0, %0#1 : tensor<20x20xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xi32>, tensor<20x20xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xi32>, tensor<20x20xi32>) {
    %0 = stablehlo.constant dense<"0x0200000003000000FCFFFFFF03000000FEFFFFFFFAFFFFFF04000000F9FFFFFF0000000006000000FCFFFFFF01000000FEFFFFFFFFFFFFFF010000000000000000000000FBFFFFFFFAFFFFFF000000000200000007000000FAFFFFFFF9FFFFFF00000000010000000500000005000000000000000000000000000000FCFFFFFF00000000FFFFFFFF06000000FCFFFFFF0000000003000000020000000600000001000000F9FFFFFFFFFFFFFF020000000000000003000000FFFFFFFF02000000010000000400000001000000FDFFFFFF03000000FDFFFFFF0000000002000000FDFFFFFF00000000FFFFFFFFFCFFFFFF020000000300000001000000FEFFFFFFFEFFFFFFFDFFFFFF0000000000000000FFFFFFFFFEFFFFFFFFFFFFFF0300000000000000020000000400000000000000FDFFFFFFFBFFFFFFFDFFFFFF00000000FFFFFFFF03000000FEFFFFFFFEFFFFFF02000000000000000100000001000000FDFFFFFF0500000003000000020000000500000003000000FEFFFFFFFFFFFFFF01000000FEFFFFFF01000000FCFFFFFFFDFFFFFFF8FFFFFF0200000001000000000000000000000002000000FEFFFFFF05000000FCFFFFFFFEFFFFFFFEFFFFFF04000000FFFFFFFFFDFFFFFF0100000001000000FDFFFFFF010000000100000000000000FFFFFFFFFCFFFFFF0000000002000000FDFFFFFF02000000FAFFFFFF00000000020000000000000003000000FFFFFFFF000000000000000002000000030000000500000001000000FEFFFFFF00000000FFFFFFFFFFFFFFFF00000000FEFFFFFF030000000200000000000000FEFFFFFF01000000FEFFFFFFF6FFFFFF04000000FFFFFFFF0300000000000000010000000300000000000000FFFFFFFFFFFFFFFFFEFFFFFF00000000FBFFFFFFFCFFFFFF00000000020000000400000002000000FBFFFFFFFFFFFFFFFCFFFFFF0000000000000000FEFFFFFF0300000001000000000000000200000000000000070000000400000000000000FCFFFFFFFDFFFFFF010000000000000001000000FFFFFFFF01000000FFFFFFFF03000000FCFFFFFFFDFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFBFFFFFF020000000200000000000000FCFFFFFFFDFFFFFF03000000FFFFFFFF0000000000000000FEFFFFFF01000000FCFFFFFF0000000000000000FBFFFFFF030000000400000002000000030000000100000000000000FDFFFFFFFCFFFFFF0200000002000000FEFFFFFFFFFFFFFF0200000002000000FFFFFFFF01000000FCFFFFFF0000000001000000020000000100000000000000FFFFFFFFFBFFFFFFFFFFFFFF0000000000000000FEFFFFFF020000000100000003000000FDFFFFFFFDFFFFFF00000000FEFFFFFFFCFFFFFFFEFFFFFF0400000003000000FFFFFFFF0000000004000000FFFFFFFFFCFFFFFFFEFFFFFF04000000050000000000000000000000FEFFFFFFF7FFFFFF010000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFF0000000000000000FEFFFFFF0000000004000000FFFFFFFF03000000020000000000000002000000040000000200000000000000020000000100000000000000FFFFFFFF01000000FFFFFFFFFDFFFFFFFBFFFFFFFCFFFFFFFEFFFFFF0100000003000000FEFFFFFF0800000000000000FEFFFFFFFCFFFFFF020000000000000004000000000000000000000000000000FFFFFFFF00000000FFFFFFFF02000000FCFFFFFFFDFFFFFF03000000FDFFFFFF00000000FCFFFFFFFFFFFFFFFCFFFFFFFCFFFFFFFFFFFFFF01000000FFFFFFFF00000000FEFFFFFFFFFFFFFFFCFFFFFF020000000000000000000000FFFFFFFFFBFFFFFFFDFFFFFF00000000020000000400000002000000000000000000000001000000FFFFFFFF00000000FDFFFFFFFCFFFFFFFEFFFFFFFEFFFFFFFDFFFFFF0100000001000000010000000000000003000000FEFFFFFF020000000000000002000000FDFFFFFF04000000030000000000000005000000FEFFFFFF0000000000000000000000000200000003000000FDFFFFFFFDFFFFFF0200000004000000050000000300000001000000FDFFFFFF02000000FDFFFFFFFCFFFFFF03000000FEFFFFFF0200000000000000020000000400000000000000FFFFFFFFFFFFFFFF0300000000000000FEFFFFFF0000000006000000FDFFFFFF000000000000000001000000FEFFFFFF00000000FCFFFFFF00000000"> : tensor<20x20xi32>
    %1 = stablehlo.constant dense<"0x02000000FAFFFFFFFBFFFFFF0200000003000000FFFFFFFF0200000003000000FEFFFFFF000000000100000000000000FEFFFFFFFBFFFFFFFFFFFFFF01000000FFFFFFFF050000000000000000000000FEFFFFFF020000000100000003000000000000000000000002000000FFFFFFFFFCFFFFFFFEFFFFFF0000000001000000FCFFFFFFFDFFFFFF000000000000000002000000FDFFFFFF080000000000000002000000FBFFFFFFFDFFFFFF000000000000000002000000030000000000000000000000FEFFFFFFFFFFFFFF00000000FFFFFFFF03000000FEFFFFFFFEFFFFFFFFFFFFFFFEFFFFFF0000000001000000FEFFFFFF0100000000000000FEFFFFFF0000000001000000000000000100000001000000FDFFFFFFFBFFFFFFFBFFFFFF01000000FFFFFFFF0400000003000000FEFFFFFF020000000000000000000000FEFFFFFF0200000002000000020000000000000003000000FFFFFFFFFDFFFFFF0000000001000000FFFFFFFF00000000FEFFFFFFFFFFFFFFFEFFFFFF0100000002000000FFFFFFFFFEFFFFFF01000000FFFFFFFFFEFFFFFFFEFFFFFFFCFFFFFF00000000FFFFFFFF0100000003000000FDFFFFFF0100000001000000FDFFFFFFFEFFFFFFFEFFFFFFFBFFFFFF0200000005000000FFFFFFFF0100000005000000FFFFFFFF00000000FFFFFFFFFFFFFFFF000000000300000000000000FEFFFFFFFFFFFFFFFDFFFFFF0200000003000000FBFFFFFF01000000FDFFFFFFFFFFFFFFFDFFFFFFFDFFFFFF020000000200000003000000FEFFFFFF00000000FFFFFFFFFEFFFFFFF9FFFFFFFDFFFFFF0000000002000000050000000300000004000000020000000200000001000000040000000200000000000000FEFFFFFF00000000FAFFFFFF01000000FBFFFFFF00000000000000000000000000000000FEFFFFFF0200000002000000000000000000000004000000FEFFFFFF02000000FDFFFFFF01000000FFFFFFFF03000000FDFFFFFF000000000000000001000000FFFFFFFFFBFFFFFF000000000200000004000000FEFFFFFF000000000400000003000000FEFFFFFF0000000001000000FFFFFFFFF9FFFFFF0200000003000000010000000300000000000000FBFFFFFF0100000000000000FDFFFFFF0100000000000000FDFFFFFFFDFFFFFF0200000004000000FDFFFFFF0300000000000000FAFFFFFFFDFFFFFF03000000FFFFFFFF0000000001000000010000000200000000000000FEFFFFFFFEFFFFFF010000000200000001000000FDFFFFFFFEFFFFFF0200000002000000FEFFFFFFFEFFFFFF04000000FCFFFFFFFBFFFFFFFFFFFFFFFDFFFFFF0000000004000000010000000000000005000000FEFFFFFFFCFFFFFF0000000005000000FFFFFFFF00000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF05000000FFFFFFFFFFFFFFFFFFFFFFFF030000000000000000000000FBFFFFFFFBFFFFFF000000000000000000000000FCFFFFFF00000000FEFFFFFFFEFFFFFF000000000100000003000000FCFFFFFF00000000FFFFFFFF0200000005000000FEFFFFFF010000000200000001000000FDFFFFFF0100000000000000020000000200000000000000FFFFFFFFFEFFFFFF0100000000000000020000000000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFDFFFFFF00000000040000000300000000000000040000000000000005000000FDFFFFFF0000000000000000FAFFFFFFFEFFFFFFFEFFFFFF00000000FEFFFFFF0100000001000000FDFFFFFF0000000000000000FEFFFFFF01000000FEFFFFFFFDFFFFFF02000000FBFFFFFF080000000100000004000000FDFFFFFF0000000004000000000000000400000000000000030000000000000001000000FCFFFFFFFBFFFFFFFEFFFFFF07000000FDFFFFFF00000000FFFFFFFFFFFFFFFFFEFFFFFF00000000010000000000000001000000FEFFFFFFF8FFFFFF01000000FEFFFFFF0200000004000000FEFFFFFF02000000FDFFFFFFFEFFFFFFFDFFFFFFFEFFFFFF0200000000000000FEFFFFFF02000000010000000000000000000000010000000100000000000000FFFFFFFF00000000000000000100000004000000000000000100000002000000020000000100000000000000FFFFFFFFFCFFFFFF0100000000000000FDFFFFFFFDFFFFFF0000000003000000FEFFFFFF00000000000000000100000000000000FFFFFFFF"> : tensor<20x20xi32>
    return %0, %1 : tensor<20x20xi32>, tensor<20x20xi32>
  }
  func.func private @expected() -> tensor<20x20xi32> {
    %0 = stablehlo.constant dense<"0x00000000090000000100000001000000FBFFFFFFFBFFFFFF02000000F6FFFFFF0200000006000000FBFFFFFF01000000000000000400000002000000FFFFFFFF01000000F6FFFFFFFAFFFFFF000000000400000005000000F9FFFFFFF6FFFFFF00000000010000000300000006000000040000000200000000000000FBFFFFFF040000000200000006000000FCFFFFFFFEFFFFFF06000000FAFFFFFF06000000FFFFFFFFFEFFFFFF02000000020000000000000001000000FCFFFFFF02000000010000000600000002000000FDFFFFFF04000000FAFFFFFF0200000004000000FEFFFFFF02000000FFFFFFFFFBFFFFFF04000000020000000100000000000000FEFFFFFFFCFFFFFF00000000FFFFFFFFFEFFFFFF010000000400000008000000FFFFFFFF0300000000000000FDFFFFFFFFFFFFFFF9FFFFFFFDFFFFFF000000000100000001000000FCFFFFFFFCFFFFFF02000000FDFFFFFF0200000004000000FDFFFFFF040000000400000002000000070000000400000000000000FEFFFFFFFFFFFFFFFFFFFFFF03000000FBFFFFFFFEFFFFFFFAFFFFFF0400000005000000000000000100000001000000FBFFFFFF08000000FBFFFFFFFDFFFFFF01000000060000000100000002000000FFFFFFFFFCFFFFFFFEFFFFFF00000000FCFFFFFF01000000FFFFFFFFFDFFFFFF0100000002000000FAFFFFFF02000000FCFFFFFF0100000005000000FEFFFFFF0000000004000000FFFFFFFF03000000030000000600000008000000FFFFFFFFFCFFFFFFFDFFFFFF01000000FFFFFFFF01000000000000000A0000000500000000000000FCFFFFFFFCFFFFFFFBFFFFFFF2FFFFFF02000000FDFFFFFF02000000FCFFFFFFFFFFFFFF0300000002000000FFFFFFFF05000000FDFFFFFF05000000FBFFFFFFFCFFFFFF00000000020000000600000000000000F9FFFFFFFFFFFFFFFCFFFFFFFCFFFFFF02000000FCFFFFFF060000000000000001000000FFFFFFFF030000000700000004000000FFFFFFFFFDFFFFFF0200000001000000FEFFFFFFFDFFFFFF0100000001000000FBFFFFFF00000000FEFFFFFFFDFFFFFFFBFFFFFF0000000005000000F9FFFFFFFFFFFFFF01000000FDFFFFFFFCFFFFFF0200000002000000FFFFFFFF03000000FFFFFFFFFEFFFFFF04000000FFFFFFFFFEFFFFFFFCFFFFFFFEFFFFFF00000000040000000800000006000000FEFFFFFF01000000FDFFFFFFFBFFFFFF0100000000000000FEFFFFFF010000000400000001000000FDFFFFFF00000000FFFFFFFF02000000FFFFFFFF000000000300000002000000FBFFFFFFFFFFFFFF040000000100000003000000FEFFFFFFFEFFFFFF0000000003000000F8FFFFFFFFFFFFFF04000000FEFFFFFFF7FFFFFFFFFFFFFF040000000400000000000000020000000500000000000000FDFFFFFFF9FFFFFF050000000600000001000000FDFFFFFFFEFFFFFFF7FFFFFF060000000600000000000000FFFFFFFFFFFFFFFF03000000FCFFFFFF0200000002000000FEFFFFFFFFFFFFFF01000000030000000300000003000000FEFFFFFFFDFFFFFF0600000001000000FEFFFFFF0100000004000000FFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFDFFFFFFFCFFFFFFFEFFFFFFFDFFFFFF0100000001000000FEFFFFFF0800000002000000FFFFFFFFFEFFFFFF050000000000000000000000FDFFFFFF00000000FCFFFFFFFFFFFFFFFBFFFFFF0200000002000000FCFFFFFF0300000005000000FFFFFFFF00000000FEFFFFFFFEFFFFFFFBFFFFFFFFFFFFFFFFFFFFFF0100000001000000FFFFFFFF0000000002000000FAFFFFFF07000000F8FFFFFFFFFFFFFFFBFFFFFFFEFFFFFFFDFFFFFFFCFFFFFF020000000000000002000000FDFFFFFF00000000000000000300000005000000FFFFFFFFF5FFFFFF01000000FEFFFFFFFEFFFFFF020000000300000001000000FFFFFFFF03000000FDFFFFFF040000000800000001000000FFFFFFFF02000000FFFFFFFF0200000003000000010000000200000003000000020000000000000003000000FFFFFFFFFBFFFFFF0100000004000000050000000200000000000000FDFFFFFF03000000FDFFFFFFFCFFFFFF02000000FAFFFFFF02000000FFFFFFFF0000000002000000FFFFFFFFFFFFFFFF0000000007000000FFFFFFFFFEFFFFFF0300000009000000FDFFFFFFFDFFFFFF0200000001000000FEFFFFFFFFFFFFFFFCFFFFFF01000000"> : tensor<20x20xi32>
    return %0 : tensor<20x20xi32>
  }
}
