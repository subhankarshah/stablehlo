module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x30xui8>
    %1 = call @expected() : () -> tensor<20x30xui8>
    %2 = call @integer_pow(%0) : (tensor<20x30xui8>) -> tensor<20x30xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x30xui8>, tensor<20x30xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x30xui8> {
    %0 = stablehlo.constant dense<"0x000302000003010004010104020102010000020000000100000201000102010402000003040003020001030000010504010303000106010002040103000001040401010103000202020301020301040003010100010103000003010301000002010104080201010300040504050300010004030002010401020303010203030301030004020001020304000401020000040002000000020001000000040200030200030301020402000101030201030401010300020100020005000302050000010002020401020703010202030403040004000002050401000400040100010401000301020103000301020302050000010501000806000405000102000202080504020001030001000201000203000202040100000101020003000204040207020401000402000305010205000102000704020106000001020303040003020400040700000201000402050004020002030304030002020200010204000003010200010202000201070101020501010301010403000004010100000504020202020303030002010300020102050102000302010001000000010003020000030202020102010002030000020301000302020103010303000305000102010003000001010202050302010502010202020300040105020001010104010401050101000100030405030201010102020501000001010302000201020102050303060200040001020001020000000308010300070106000001010802000000020000000001030100010004010001030003020002020101030201020403000102030202000405010004010102040403020102000500000300020002"> : tensor<20x30xui8>
    return %0 : tensor<20x30xui8>
  }
  func.func private @expected() -> tensor<20x30xui8> {
    %0 = stablehlo.constant dense<"0x00AB000000AB0100000101000001000100000000000001000000010001000100000000AB0000AB000001AB000001CD0001ABAB0001000100000001AB0000010000010101AB00000000AB0100AB010000AB0101000101AB0000AB01AB0100000001010000000101AB0000CD00CDAB00010000AB000001000100ABAB0100ABABAB01AB000000000100AB00000001000000000000000000000001000000000000AB0000ABAB01000000000101AB0001AB000101AB000001000000CD00AB00CD000001000000000100B7AB010000AB00AB000000000000CD000100000000010001000100AB010001AB00AB0100AB00CD000001CD010000000000CD00010000000000CD00000001AB00010000010000AB0000000001000001010000AB0000000000B700000100000000ABCD0100CD00010000B70000010000000100ABAB0000AB00000000B700000001000000CD0000000000ABAB00AB00000000000100000000AB010000010000000001B7010100CD0101AB010100AB00000001010000CD0000000000ABABAB000001AB00000100CD010000AB000100010000000100AB000000AB0000000100010000AB000000AB0100AB000001AB01ABAB00ABCD0001000100AB000001010000CDAB0001CD0001000000AB000001CD000001010100010001CD0101000100AB00CDAB000101010000CD0100000101AB00000001000100CDABAB00000000000100000100000000AB0001AB00B70100000001010000000000000000000001AB0100010000010001AB00AB000000000101AB00010000AB000100AB00000000CD0100000101000000AB00010000CD0000AB00000000"> : tensor<20x30xui8>
    return %0 : tensor<20x30xui8>
  }
  func.func private @integer_pow(%arg0: tensor<20x30xui8>) -> tensor<20x30xui8> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<20x30xui8>
    %1 = stablehlo.multiply %arg0, %0 : tensor<20x30xui8>
    %2 = stablehlo.multiply %0, %0 : tensor<20x30xui8>
    %3 = stablehlo.multiply %1, %2 : tensor<20x30xui8>
    %4 = stablehlo.multiply %2, %2 : tensor<20x30xui8>
    %5 = stablehlo.multiply %3, %4 : tensor<20x30xui8>
    %6 = stablehlo.multiply %4, %4 : tensor<20x30xui8>
    %7 = stablehlo.multiply %5, %6 : tensor<20x30xui8>
    %8 = stablehlo.multiply %6, %6 : tensor<20x30xui8>
    %9 = stablehlo.multiply %7, %8 : tensor<20x30xui8>
    %10 = stablehlo.multiply %8, %8 : tensor<20x30xui8>
    %11 = stablehlo.multiply %9, %10 : tensor<20x30xui8>
    return %11 : tensor<20x30xui8>
  }
}
