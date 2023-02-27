module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>)
    %2 = call @expected() : () -> tensor<5x6x7xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      stablehlo.return %arg1 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xf32>, tensor<2x2x1xi32>, tensor<5x2x2x7xf32>) -> tensor<5x6x7xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf32>, tensor<5x6x7xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>) {
    %0 = stablehlo.constant dense<"0xFCE1993FD4C18940627D85BF8FCF943FF94E334065AE93BFC253C0BFFC7556409473E9BF2385D1BDC0E3AE3FE21866C048D31E40E0753AC079E32C3FAACB753FC8C6B0BD5C49D63FC88157BF6D9CC13F6441ED3F650F674008F120400D46A1BF7D0889BE389E0E3F6C6033C0539EA9C05EB80D3C78D3624058C6DB406215A03FBE190DBF65AF743F64A301C0738E21BE6915E23F7A48444039750AC07A6CA240D96F08406E7127BF46DC763FED4EA0C0706AEABF18CEDB3F0E12053F8D7015403F9EB24074F60441CF1C46C086B21840AC666CC040DF0A40086A11C161B523C0071BC63FA160A23EDF4BB5BFF51708C0C70CAEBFF9408940CA50094029F3183FE009843E4BD5F9BFFDA82BBF700832BF6996A840A2F192BF89CA7140311A093F9C850D40BFE82E3FD1E02340C158C43FCADBA1400BE5CEBFEFF6A740BA3AE9C09CBEE9BF9520133F21353F40D20E2FC0A49B19400D0BBEBE98FB933ED9A69AC01DB0AF3F3DC933407E0BD83F8930813C374DC24068F28C40579A223F406997BDFEB3B63FB74DF1BFC7454D40F41FBA408D92B6BF897EB1BEEE3BA6C04DFF43405DC4ABBEB698154015DA803EE7EC81C0B5272240233637403A1B8B40F16898C09427A8BF848B643E8904C2BD4BF9F3BF21821740BBAC30C0769A193EE64A3B3E1E068440FD3809C0DF672A3FF5BF3B401570DC40CB440E409A81BD3FE489AEC0F88CAB3DAD91A43E431F97BFAF0B5040F824223FFADF6B4083A235403A5693C0D5037F409B2C433E869CEDBF8086A7BF3DB370C0BD3A8BC03B0991C066C9BF3F9480F9BE4C288AC0A49A70C049A1CD3FDD933440D80F82BFD8B568C065637C40715AF1BE2AB9FE3F33A9163E0E1F02C0943F24BE25EAC2C006AF98BF82E7DD3E41E7723FA4A643BF2FA7A340DD4C5EBFB975B4BF518D2940B869F83E55356BBFFB7D583E43000B3D0798F43F5D348F3F17D61B405F197840607120BF1979E0BF3E7786400F2A383D022553C011848BC0CEDFC13F3C42D9BF8EDCAEBF904AAA3FCD8EF4C0057774C0FBE6C0BE1E06F3BF8EC82C3E5DFB6F40A40F9B3E76EADFC0FFD978407402213FA8C651C0BC23CB3D6F70CB3FD6D611C0B54BB0402EB0AABF294533C0B6EDD7BF5B20B93D4731DE3F5C0B433F4D390FC0A6353E3F7BA6094091B787C0C1276FC0"> : tensor<5x6x7xf32>
    %1 = stablehlo.constant dense<"0xE037484013F1D1BF9E72F1BFAE079EBF43F1D7BEC775DDBE9BAE18C03E2E3AC0606DF5BF26AC67BE69449CBD123D01C012521FC03352BC400FB76DBFAE8F04405E912E4098CD87C03D2C804061ACE53FD56A72C0DFDFCBC07EEDC4406C7ED340974B9DBF27AF693F17E2D83EBC9739C0A502CBBFE50941C0A9904DC01D245DC0D3BF17C0CA2035C05662983ECF4654C0CF494F40C75D9EBF6D1005BD5BCAA2C00B26C1BF40CA0C415F160DC089552D40C2C6D3BFD29B4A40FF84834025B87EC05D9FA74093C9EFBFF43249401447DBC06481093FD1BCCF3FF49F7C3F47C171C0AA9641C0BAFD074029F57140DAF837C08B527D40441D9BC04F0BD8BF236FEFBFA99466C0BB7638C03578A23EF06F613F6AA905BF1B57EFBF1DFBE2BFE40C09C06F59EC3FA76CCA40F526903EEEEB8B40BFBCF33F88230B40DC0D31C02A035CC0D1183AC004427EBFAF8F5D3E1DD3334080CDF4BE126437C0E07A28C0D21BE7BE16A3863F899DCA3FAB31B940E15F0C3FC5903EC0F29652C0A3A2DFBF07CD76C01476B13F3AE02D4094D7E43F393132C0877BAFBFFE85D03FC9513A403B9E0F40177F613F04D7513DB86AB7C084340240A09627C0AC1DD03FC0860B3FCF0ACA40304DC13D23D9EBBFACA0304007AF85C02EECF33FBCD6CDBF523249401ED0CB3D0B650340C2062E3FAE292CC0F2A73AC083FFAB3DE4D43B3D1F1A0D40263B5240C26A79C04EECBF3F57449B404F8462C0604B913F06AD2840F29AC83D9CFE42C069DDF2BF08F920C0344E43404DE091C0"> : tensor<5x2x2x7xf32>
    return %0, %1 : tensor<5x6x7xf32>, tensor<5x2x2x7xf32>
  }
  func.func private @expected() -> tensor<5x6x7xf32> {
    %0 = stablehlo.constant dense<"0xE037484013F1D1BF9E72F1BFAE079EBF43F1D7BEC775DDBE9BAE18C03E2E3AC0606DF5BF26AC67BE69449CBD123D01C012521FC03352BC400FB76DBFAE8F04405E912E4098CD87C03D2C804061ACE53FD56A72C0DFDFCBC07EEDC4406C7ED340974B9DBF27AF693F17E2D83EBC9739C05EB80D3C78D3624058C6DB406215A03FBE190DBF65AF743F64A301C0738E21BE6915E23F7A48444039750AC07A6CA240D96F08406E7127BFA502CBBFE50941C0A9904DC01D245DC0D3BF17C0CA2035C05662983ECF4654C0CF494F40C75D9EBF6D1005BD5BCAA2C00B26C1BF40CA0C415F160DC089552D40C2C6D3BFD29B4A40FF84834025B87EC05D9FA74093C9EFBFF43249401447DBC06481093FD1BCCF3FF49F7C3F47C171C089CA7140311A093F9C850D40BFE82E3FD1E02340C158C43FCADBA1400BE5CEBFEFF6A740BA3AE9C09CBEE9BF9520133F21353F40D20E2FC0AA9641C0BAFD074029F57140DAF837C08B527D40441D9BC04F0BD8BF236FEFBFA99466C0BB7638C03578A23EF06F613F6AA905BF1B57EFBF1DFBE2BFE40C09C06F59EC3FA76CCA40F526903EEEEB8B40BFBCF33F88230B40DC0D31C02A035CC0D1183AC004427EBFAF8F5D3E1DD333409427A8BF848B643E8904C2BD4BF9F3BF21821740BBAC30C0769A193EE64A3B3E1E068440FD3809C0DF672A3FF5BF3B401570DC40CB440E4080CDF4BE126437C0E07A28C0D21BE7BE16A3863F899DCA3FAB31B940E15F0C3FC5903EC0F29652C0A3A2DFBF07CD76C01476B13F3AE02D4094D7E43F393132C0877BAFBFFE85D03FC9513A403B9E0F40177F613F04D7513DB86AB7C084340240A09627C0AC1DD03FC0860B3FCF0ACA4033A9163E0E1F02C0943F24BE25EAC2C006AF98BF82E7DD3E41E7723FA4A643BF2FA7A340DD4C5EBFB975B4BF518D2940B869F83E55356BBF304DC13D23D9EBBFACA0304007AF85C02EECF33FBCD6CDBF523249401ED0CB3D0B650340C2062E3FAE292CC0F2A73AC083FFAB3DE4D43B3D1F1A0D40263B5240C26A79C04EECBF3F57449B404F8462C0604B913F06AD2840F29AC83D9CFE42C069DDF2BF08F920C0344E43404DE091C06F70CB3FD6D611C0B54BB0402EB0AABF294533C0B6EDD7BF5B20B93D4731DE3F5C0B433F4D390FC0A6353E3F7BA6094091B787C0C1276FC0"> : tensor<5x6x7xf32>
    return %0 : tensor<5x6x7xf32>
  }
}
