module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xf32>, tensor<5x2x2xf32>)
    %2 = call @expected() : () -> tensor<5x6x7xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xf32>, tensor<2x2x2xi32>, tensor<5x2x2xf32>) -> tensor<5x6x7xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf32>, tensor<5x6x7xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf32>, tensor<5x2x2xf32>) {
    %0 = stablehlo.constant dense<"0xAC0A1840E185F53FB4ACEA3EC9B352BFE41A7BC0C763433F39F003409CCDFD4094A40FC099D10641EC34FE405A863DBF1A082F40008B6F4007F295C0B504B0BFDEEB9F3FBA9C8340FA16E6BF0F7BB03F5E13DE3F7E396AC08E4586BFB2B82BC02EE4603FA0CA373F53DE17C032D389C0703085BF0ABF53C035C23A40FC639F3FC1698E3E5D1A294029825FBFFEB34C3F17A29A40A488B73F6789C1C06CE43A40F2D480C004E19F3FC7EB193F5C901FBF9D1707C18CE99BBE7AAF29C049E5F7BE88022CC01871BABF29DC963FBE2B9F40CD7EB7BFA7E285C0C14C21C0ABEA033EFBA0D93F002422401AF52140433693C08AB293BF1EED2B40CD78BF3D35E5AB40FABFFDBF33776ABE63A7144012F9DEBFA40EA140F150683E770821C03F103C400A0F1F3F09C84C4074C420C091A5C0BFF01381C0E9EFF43FDE059EBE714599C05CBBD3BEBFFB9C403F5B96404C5FD63F00D6193FE4F9DDC0FB27DABEEA66C6C0CFA0E5C01763C2C088DC98C0B6AA01BF0372AAC04B8C8BC056056A408DE56AC0629B703F83858AC003752440F6E61D404B3D2C3FC9B8A2406225144080AC2EC091FAD33F1054D4409AACDEBFF074BAC06A9DEF3F2B561FC056481AC016612AC01DF07BBF2AC654C049E783BF1BDFD13E749BA9C0F6A21CBFB9AF38C08405C8BFC1B65EC00325033F810C41C04B6014C00F5F10BE2EB717BEE7ED6AC0AD688E40278D11C033E087BF54711F3D2EB1FD3EEE3C5C40DB74A23EC846183F793E8A3F8907043E31377040DDAA2040AEC829BF7790363F76668B3F555B0B405094A840A4D92940F19666402971C63FBBFB92C0ED1ADFBFE69FB240DAE63540962F483F4921973F22066C3E38CA16405FC1FF40B658C13D04922840597B723F57413840193E12C04910463FC19E00C0C8E6D4C0C90B53C0D7B246BF37F935C013233C40033F1440B0E288C0047A84BFD234F9BFBFE1A9BF93044BBF7F7C09C05799C3C09CD80A4037EC85BFA636673F5E1F823FB9038640688E1240EB1CCB3FBF4710C0814A3ABFD158E84066DE2440790F6740DAAAA1402548493F83C19DC0EFF461C031D9AE405B79BCBCB6EE5E3E817932BF26EF9FC0B25FF7BF18AA29C0445EB93F98990CC0A8DDD2BF8B801BC0D20AAABFFFB026406EAA28BF2E34ACBF04DA123FD5B3B33FC934E1BE"> : tensor<5x6x7xf32>
    %1 = stablehlo.constant dense<[[[-1.38839912, 2.88754797], [-1.14961457, 4.35638428]], [[1.57343924, 3.23197055], [1.54987252, 0.0545112751]], [[-2.30178356, -0.239619657], [3.00983477, 0.0196213443]], [[-2.4032948, 0.810928285], [-0.771382927, 0.678326785]], [[4.099800e+00, 1.30362475], [-1.09447253, -3.37127972]]]> : tensor<5x2x2xf32>
    return %0, %1 : tensor<5x6x7xf32>, tensor<5x2x2xf32>
  }
  func.func private @expected() -> tensor<5x6x7xf32> {
    %0 = stablehlo.constant dense<"0xAC0A1840E185F53FB4ACEA3EC9B352BFE41A7BC0C763433F39F003409CCDFD4094A40FC099D10641EC34FE405A863DBF1A082F40008B6F4007F295C0B504B0BFDEEB9F3FBA9C8340FA16E6BF0F7BB03F5E13DE3F7E396AC08E4586BFB2B82BC02EE4603FA0CA373F53DE17C032D389C0703085BF0ABF53C035C23A40FC639F3FC1698E3E5D1A294029825FBFFEB34C3F17A29A40A488B73F6789C1C06CE43A40F2D480C004E19F3FC7EB193F7566C93F9D1707C18CE99BBE7AAF29C049E5F7BE88022CC01871BABF29DC963FBE2B9F40CD7EB7BFA7E285C0C14C21C0ABEA033EFBA0D93F002422401AF521409BD84E408AB293BF1EED2B40CD78BF3D35E5AB40FABFFDBF33776ABE63A7144012F9DEBFA40EA140F150683E3962C63F3F103C400A0F1F3F09C84C4074C420C091A5C0BFF01381C0E9EFF43FDE059EBE714599C05CBBD3BEBFFB9C403F5B96404C5FD63F00D6193F6C5013C0FB27DABEEA66C6C0CFA0E5C01763C2C088DC98C0B6AA01BF0372AAC0F1BCA03C56056A408DE56AC0629B703F83858AC003752440F6E61D404B3D2C3FC9B8A2406225144080AC2EC091FAD33F1054D4409AACDEBFF074BAC06A9DEF3F2B561FC056481AC016612AC022A140402AC654C049E783BF1BDFD13E749BA9C0F6A21CBFB9AF38C08405C8BFC1B65EC00325033F810C41C04B6014C00F5F10BE2EB717BEE7ED6AC0AD688E40278D11C033E087BF54711F3D2EB1FD3EEE3C5C40DB74A23EC846183F793E8A3F8907043E31377040DDAA2040AEC829BF7790363F76668B3F555B0B405094A840A4D92940F19666402971C63FBBFB92C0ED1ADFBFE69FB240DAE63540962F483F4921973F22066C3E38CA16405FC1FF40B658C13D04922840597B723F57413840193E12C04910463FC19E00C0C8E6D4C0C90B53C0D7B246BF37F935C013233C40033F144090318340047A84BFD234F9BFBFE1A9BF93044BBF7F7C09C05799C3C09CD80A4037EC85BFA636673F5E1F823FB9038640688E1240EB1CCB3FBF4710C0814A3ABFD158E84066DE2440790F6740DAAAA1402548493F83C19DC0EFF461C031D9AE405B79BCBCB6EE5E3E817932BFAD178CBFB25FF7BF18AA29C0445EB93F98990CC0A8DDD2BF8B801BC0D20AAABFFFB026406EAA28BF2E34ACBF04DA123FD5B3B33FC934E1BE"> : tensor<5x6x7xf32>
    return %0 : tensor<5x6x7xf32>
  }
}
