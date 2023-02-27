module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = chlo.lgamma %0 : tensor<20x20xbf16> -> tensor<20x20xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xFC40974010BFDF3E16C066BFE93FC5BF1EC090BF2CC005405540ABC042C061BF2240E9BF44C0BBC0C63FB63F884064BF17C023BF6BC03C4033BF15C0D0BF6040BDC07BC025C0E33FB04035C00E3F7ABD27C0B13F243D59C0D7BFFD3FF13F3FBEB4BFDCBE47BF094011C0ED3F91C047400E40BFBE873F4CC0883F4BBE203F99C0F1BF113E8C3F05C024C04940953F34C04EBEC33F59BFC2405A4093BF54C029C095C00B3F44C0723F26BF5440AA3F9040F0BFD7BF88BF9640B7BE92BF703EA7C076C025BD8B3FA43E00405FC04AC021C04D40A2BFCC3F17407FBF14C0EEBF803FCABF504009C080BFA74013401D406BC02D40643FA0402F3F9DBF4B40C33F22C086BF97407D40AB3F853FD93F2BC09B4026409A3F6CC0A43E38BF02C0333E6B409B3F17BFBBBE0B40F03F9140033FD83FAA40114028BF0FC0C8BEF4C0BF3F33408E3E494086BF4FBF09404CBEA83FE4BDA2BF923EC23F08C0053FF93FF6BF9D40B2BFAC3E223EE53E89BF934000C0DC4015C0BFBFA13F923F34BF173ECF3F0141F23EC4C0A14067C0E3BF7D40D5405ABF00C02FBF2B40FCBF3DBE33C063BF71BFF43F90BFB5BFDDC03CC0D040A240694047C079BFD73F934071C011C031C099BF2CBFE0C09CBFE73F27BD4BBFDABEDD3EA83F773F9F3F3340C7404CC0B840C63E1B40A53F9540EABF88BF60C0C1BF9F40B34062400DC0EC3F683F04C057403F3FF23CD540043F03401CBFA340FDBF5640A23F033F7B40163F1CC018BFB9C0A83E8EBFFD4052C01F3E9BBFCFBF2740D5BF13BFE1C0BBBE2840A4BFB0BF314043C05040D63FA5C01B40953E03BFE24052C005C0B6BFFA3E3FC0A7BFADC029BF8DC083BFF5BF1940DB3F8C3F16C08BBE47C0E33FDBBEE33F8D403CBFCD3EAE40C4C0414021C0C03FDF3F0FC0134029409EBFA7409A40B9C045C00DBF30BFA2BD70C0EBBF19C0C0C01941CABFA13FA84070C00F40774088C0FABF41C0983F583F61C0FF3F3D40DEBEBBC0AEBEC44037BFB0BFE13D34BF77BD2B40E73FFA3F03C00FC00640013FA4C02940D03E18C072BFF1BFA7407F402AC089BC4240BC3F4C40BBC0004012BF45BF80C0B0BE5A40C13F3D40094001C1F2C0E7BE90BF34C0DF3F243F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x04413140A43F363F733E164085BD573F8CBC0340ACBD0F3D823F84C0D13F0C409D3E9E3F693F8DC0F2BDF7BD07401240493EAC3FB3BF233FBA3F903E593F9A3F81C000BFE7BDA1BD7D406B3EF23E3540F3BDF2BD4D408DBF653F1FBC38BDE93F783FA53FD53F863DFC3E63BD37C04C3FDC3DAC3FEFBC99BE07BDE23FB93E34C0CA3FF23F41BDE63FDABD543F98BD2F3EE13FF5BDFD3F9D408D3FF03F5EBFE9BD3CC0FC3E693F0C3DAE3F803FE6BD1D40C33F653F30402E40AF3FF73FAE3F64C086BF4F4033BD833F0000A5BFC2BD94BD633FA83FE8BD433EB140A93EB73F0035563F703F953F807F61401C3E813EB3BFE03E973D4B40903EBB3F5C3FF5BDB0BD43403140DE3FE9BDAFBCC6BDC7BD3D40B53EB1BDB2BF833FBF3F3040D53FB33FB5BDA63FAD3FA83D43BD20400D3FC9BD6A40093EB03F1F3FA93F09C1F8BD043F973F543F4340E43F863DE23FE2BD1140A83F933FF6BDA63F093FB5BCF83F42407F3F7A3FE33F2F3F28402540807FCB40903E5E3FCDBD87BDBB3FEC3FE2BD0A41213F97C04E40B3BF8A3FDE3FBE400040807FB63FD33E3340EA3FF63D1040374017BD0340753FBFC0883FB5405140AE3F9B3E6740CDBD2540A5BFFC3E1A3DCD3FB33F807FBF3F8FBD4E40DC3FA63F383FE2BDAF3CC6BD043FA54099BE8C40553F6D3EDABD2B40A33F3040A8BF5B3F484084409E3F463F6DBD7D3D0240863F553E6040BE400B3FA83CA83F54404540843FD0BD0D3FD93FD73E013DA73F92C0803F0C40054141BFE53FC43F583FBB3E613FA53FA4C0AD3FC13EA13F833FFA3E9B3F703FD0BD4AC06D3E903FA23FD64041BFE63F723F193F1940983F8AC0B13F28C07040ED3F583EBFBD41BD733EC43F9B3EA1BDA53FA1BD1540C43F4C3F774097C0353F94BDF7BDB1BD1F3F1C3EC73EB73F61403A4092C02B3FA33FB73F2640A9BFA73FFB3D807F3D41563FCDBD6440A9BFEE3DCF3F04C01A401640A7BDE53DAABF57BB263FA53F8DC0B23FA040BE3F833F0A40BB3F3640D33E8FBD9CBC15401F3F2E3D113F39C0C73E483F223E3C40CA3F6140E33FDBBD8340393FF9BD5F3F8DC00000A53FD13F807FB13F8D3FF7BD263F863DFFC008C1A43F03402F3EB1BDAD3E"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
}
