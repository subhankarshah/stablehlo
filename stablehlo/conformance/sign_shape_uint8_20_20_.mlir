module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xui8>
    %1 = call @expected() : () -> tensor<20x20xui8>
    %2 = stablehlo.constant dense<0> : tensor<ui8>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui8>) -> tensor<20x20xui8>
    %4 = stablehlo.compare  EQ, %0, %3,  UNSIGNED : (tensor<20x20xui8>, tensor<20x20xui8>) -> tensor<20x20xi1>
    %5 = stablehlo.constant dense<0> : tensor<ui8>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui8>) -> tensor<20x20xui8>
    %7 = stablehlo.constant dense<1> : tensor<ui8>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<ui8>) -> tensor<20x20xui8>
    %9 = stablehlo.select %4, %6, %8 : tensor<20x20xi1>, tensor<20x20xui8>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<20x20xui8>, tensor<20x20xui8>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xui8> {
    %0 = stablehlo.constant dense<"0x02030001040203020002000203000100050501000000000402020101000004020000040304030204010200000201050503000101030203000104000002020301000101040100010101050101050301020300000100000400000500020103000102000202000103040200000402000100050102060400000103000500040200010202020101050100030000010203000200020300030304000301010100000203000201000202070400040001030300040005040301030101060300030304000000000300000002020101030100000204020101030202020402000205000303020201000401070100000002010101030600000102000502020003030101030002040204010302000001010002020406010201000103030102020103030302020002020200010200020300030603000101000101020005070100010101020403030406030700010802030101030100050102060304000104020301010201020305020000040301000303030302030004030101000501030102040301020501070303010000040001020100000303000402"> : tensor<20x20xui8>
    return %0 : tensor<20x20xui8>
  }
  func.func private @expected() -> tensor<20x20xui8> {
    %0 = stablehlo.constant dense<"0x01010001010101010001000101000100010101000000000101010101000001010000010101010101010100000101010101000101010101000101000001010101000101010100010101010101010101010100000100000100000100010101000101000101000101010100000101000100010101010100000101000100010100010101010101010100010000010101000100010100010101000101010100000101000101000101010100010001010100010001010101010101010100010101000000000100000001010101010100000101010101010101010101000101000101010101000101010100000001010101010100000101000101010001010101010001010101010101000001010001010101010101000101010101010101010101010001010100010100010100010101000101000101010001010100010101010101010101010100010101010101010100010101010101000101010101010101010101010000010101000101010101010001010101000101010101010101010101010101010000010001010100000101000101"> : tensor<20x20xui8>
    return %0 : tensor<20x20xui8>
  }
}
