module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x20xi32>, tensor<20x20xi32>)
    %1 = call @expected() : () -> tensor<20x20xi32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1] : (tensor<1x20xi32>) -> tensor<20x20xi32>
    %3 = stablehlo.shift_right_logical %2, %0#1 : tensor<20x20xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<20x20xi32>, tensor<20x20xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x20xi32>, tensor<20x20xi32>) {
    %0 = stablehlo.constant dense<[[0, 0, 0, 5, 1, 2, 0, -3, 1, 0, 0, 0, 1, -3, 3, 0, -2, 0, 4, -2]]> : tensor<1x20xi32>
    %1 = stablehlo.constant dense<"0xFFFFFFFF000000000000000000000000FFFFFFFF00000000FDFFFFFF0100000000000000FAFFFFFFFCFFFFFFFFFFFFFF040000000200000004000000000000000000000000000000010000000200000002000000FFFFFFFF02000000FDFFFFFF05000000FFFFFFFF0500000000000000FFFFFFFF01000000F8FFFFFFFFFFFFFF00000000FFFFFFFF0000000000000000FFFFFFFFFCFFFFFF000000000000000000000000FBFFFFFF00000000FCFFFFFFFFFFFFFF01000000FFFFFFFF00000000000000000800000004000000FEFFFFFF0100000000000000FDFFFFFFFFFFFFFF01000000FCFFFFFFFAFFFFFF00000000FFFFFFFF00000000FFFFFFFF03000000FFFFFFFF00000000FFFFFFFF03000000FBFFFFFF00000000FFFFFFFFFFFFFFFF000000000100000001000000FDFFFFFF0200000000000000FFFFFFFFFDFFFFFFFEFFFFFF06000000FFFFFFFFFEFFFFFF04000000FEFFFFFF00000000FFFFFFFF02000000FBFFFFFF02000000010000000100000000000000FFFFFFFF0000000003000000F8FFFFFFFCFFFFFFFBFFFFFFFFFFFFFF00000000000000000000000003000000000000000200000003000000FFFFFFFFFFFFFFFFFEFFFFFF030000000100000003000000FAFFFFFF0200000000000000FFFFFFFFFEFFFFFFFFFFFFFF02000000FAFFFFFF02000000FFFFFFFFFFFFFFFFFCFFFFFF00000000000000000000000002000000FFFFFFFF0100000002000000020000000000000003000000000000000100000000000000FFFFFFFF01000000FEFFFFFF00000000F8FFFFFFFDFFFFFF0100000000000000FFFFFFFF02000000FBFFFFFFFDFFFFFF05000000FBFFFFFFFFFFFFFF02000000FAFFFFFFFEFFFFFFFDFFFFFF050000000000000000000000010000000000000001000000FDFFFFFF0000000000000000FEFFFFFFFEFFFFFF00000000000000000100000005000000030000000100000001000000FAFFFFFF030000000200000002000000010000000000000002000000FDFFFFFF00000000FAFFFFFFF8FFFFFF00000000FFFFFFFFFEFFFFFF04000000FDFFFFFF02000000FAFFFFFFFFFFFFFF0000000000000000FEFFFFFF050000000000000000000000FEFFFFFF0100000002000000FEFFFFFF060000000000000000000000FCFFFFFFFBFFFFFFFFFFFFFFFFFFFFFF020000000300000002000000FCFFFFFF0100000000000000FEFFFFFF00000000FDFFFFFF0700000000000000FEFFFFFF0000000001000000FFFFFFFFFFFFFFFF0000000000000000000000000200000001000000FCFFFFFFFCFFFFFFFFFFFFFF000000000300000000000000FFFFFFFF00000000000000000300000001000000FBFFFFFFFFFFFFFF0000000000000000FCFFFFFF02000000010000000300000000000000000000000500000000000000FBFFFFFFFBFFFFFFFCFFFFFF000000000000000000000000FEFFFFFFFEFFFFFFFFFFFFFFFCFFFFFF020000000500000007000000FDFFFFFF0300000000000000FFFFFFFF04000000FFFFFFFFFDFFFFFFFFFFFFFF020000000000000000000000020000000100000001000000000000000000000003000000FBFFFFFFFFFFFFFFFEFFFFFF00000000FFFFFFFFFDFFFFFFFEFFFFFFFFFFFFFFFDFFFFFF010000000400000001000000FDFFFFFFF8FFFFFF000000000000000000000000FEFFFFFF0300000000000000FDFFFFFFFFFFFFFF0000000001000000FBFFFFFF0000000001000000FFFFFFFFFEFFFFFF0000000002000000FAFFFFFF000000000000000000000000FDFFFFFF020000000400000000000000FFFFFFFF000000000000000000000000FEFFFFFF02000000FEFFFFFF01000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFEFFFFFF03000000FCFFFFFFFFFFFFFF010000000300000000000000030000000200000001000000FFFFFFFFFFFFFFFF01000000000000000100000001000000000000000200000003000000030000000400000000000000FEFFFFFF00000000000000000200000005000000000000000500000004000000FEFFFFFF0100000000000000FCFFFFFF00000000000000000200000003000000000000000000000003000000FFFFFFFF00000000FFFFFFFF0000000000000000040000000000000002000000010000000200000002000000FEFFFFFFFEFFFFFFFFFFFFFFFFFFFFFF020000000500000000000000010000000300000002000000"> : tensor<20x20xi32>
    return %0, %1 : tensor<1x20xi32>, tensor<20x20xi32>
  }
  func.func private @expected() -> tensor<20x20xi32> {
    %0 = stablehlo.constant dense<"0x00000000000000000000000005000000000000000200000000000000FEFFFF7F0100000000000000000000000000000000000000FFFFFF3F0000000000000000FEFFFFFF0000000002000000FFFFFF3F00000000000000000000000000000000000000000000000000000000FDFFFFFF0000000000000000000000000000000001000000000000000300000000000000000000000000000004000000FEFFFFFF00000000000000000000000000000000000000000100000000000000FDFFFFFF0100000000000000000000000000000000000000FDFFFFFF0000000000000000FFFFFF7F0000000000000000FEFFFFFF00000000000000000000000000000000000000000200000000000000FFFFFF1F0000000000000000000000000000000001000000FEFFFF7F0100000000000000FFFFFF3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FDFFFFFF0000000000000000FFFFFF1F00000000000000000000000000000000000000000000000005000000000000000200000000000000FFFFFF1F0000000000000000000000000000000000000000FFFFFF1F0000000000000000FEFFFFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000FDFFFFFF0100000000000000000000000000000000000000FFFFFF3F0300000000000000FEFFFFFF00000000040000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FEFFFFFF00000000000000000000000002000000000000000200000000000000000000000000000000000000000000000000000000000000FFFFFF1F0100000000000000000000000000000001000000FFFFFF3F00000000000000000000000000000000010000000000000000000000FDFFFFFF0000000000000000000000000000000000000000000000000000000000000000FEFFFFFF0000000000000000FEFFFFFF00000000000000000000000001000000000000000000000000000000FDFFFFFF0000000000000000000000000000000000000000FFFFFF1F0000000000000000FFFFFF7F0000000000000000FEFFFFFF00000000000000000000000000000000010000000100000000000000000000000100000000000000000000000000000000000000000000000000000000000000FEFFFFFF00000000040000000000000000000000000000000000000002000000000000000000000000000000FDFFFFFF0000000000000000000000000000000001000000FDFFFFFF0000000000000000000000000000000000000000FEFFFFFF00000000000000000000000000000000000000000000000000000000FFFFFF070000000000000000000000000000000000000000FFFFFF0F0000000000000000000000000000000004000000FEFFFFFF00000000000000000000000005000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FFFFFF0F00000000000000000000000000000000000000000000000000000000000000000200000000000000000000000100000000000000000000000000000000000000000000000000000000000000FFFFFF3F0000000004000000FEFFFFFF00000000000000000000000000000000010000000000000000000000FDFFFFFF0100000000000000000000000000000000000000000000000000000000000000FEFFFFFF0000000000000000FFFFFF1F00000000000000000000000000000000010000000000000000000000FEFFFF7F0000000000000000000000000000000000000000FEFFFF7F0300000000000000FFFFFF1F0000000000000000FEFFFFFF00000000000000000000000001000000000000000200000000000000FFFFFF0F0000000000000000000000000000000001000000FDFFFFFF0000000000000000FEFFFFFF00000000000000000000000000000000000000000000000005000000000000000200000000000000FEFFFF7F0000000000000000000000000000000000000000000000000000000000000000FEFFFFFF0000000000000000FFFFFF3F"> : tensor<20x20xi32>
    return %0 : tensor<20x20xi32>
  }
}
