module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.sine %0 : tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0xAB2E8FBFAC6856C063EA0240F1ED4A407DA0434012EF0B409E039FBF23BBB8BFB4F942C02738034097C74340ADB6A4BF2866F7BF4B664D40E35C70C0B7EE683FA20857C04DEBC6BFB08B8A3F671448BFAD1722C0D903A9BF7C6710411CDC0C406AFEBB3E8744B53FC7CC80400D5E85BCC814B6BF0EA235BFF94083C05B54C3BF7CF861C05FAB063F83C701BF617B36404EC8A33EFBBA593E2FADBABF688DDD4072E04140AE5E83BF4F6EC23FA698EE3FD2D213C04F0580400388DC4006BF6ABFBCED8DC0E90EE43EFCB4943FC3D88940E35045C0F9C3C13F78ABF6BE8737A7BFABAD1EC02A14ABBE1647EE403D085540229BA3C0930B3EC0D6260CC08FE2C2BE6885DFBEE397CEBFC4384340C9CBDA40BB3B4840F99FAB3F4067D93DB9EFC3BE301F3BC02260D9BFF1F290BF25ECE0BFB126ACBFCA808540D7020540251415C0D8A2063F79571CC0CEF63B40E883E93F69A983404E6F0B4073309E3FF284993FF55190409C9A5140DD6131C03E148BBF49160240F2DD4BC0634102C08C11853F70F0D43FC1AA7EC0C9BE9B3F2E964C3F9772023EB19116407456C13E710FD7C070EF45BF0281A9BF7E8C2540DA3E5740CB31BFC0ACEC87BF893B863EB597E6BF0CF482C0EB609D3F7C1CE7BF1DB387BF38C794406020D63F35510EC05DC8E1C0C81ADE40CE3F204040438F3FACE515BF9BDDCEBECA30A240F9482FC0C54A3B3F08A89F40925259407230EBBFCE418B3F51E636C04DCE47C0F4AF333F527AB23E0F975540C525893F6E895740416697402E6DCC3F36BDE13F0E32763FD2BD33C0016D14BF62D3653F2E283F4000A10740F719024165C1F13F3A8E5840CCD4F0BE2657ACC0C58917C0D2F7AD3F44BCCF3FF4C63940E8261640553C2640C7514F40D609E73FE73CEBBF5349ECBE107FE53F9199A13CBF6C0A409069AF3F1AD8B93F2C963DC059DC02C0BD1E1EC0FAECF83FEE4FC24034E72540E78065C04080AE3F89C26240CBDF043F15BC9B3F8E3B0E3F035D6F40A51242BE2D675DBEA73D2CBFFDC3BDBF276E14C0F783DDBF23AB9040AA4713C0DF219D3FCC64F13F4C795D40D08A0CBE673739408F644640D5AD2AC048C999C0A6F70BBF16E432C02036E63F232AEBBD0A2D76BFC536CDBF464381BE57C34CBF527E823E22592040D54599C017AD80C031AED44021ADCFBF809900C014DA3DBEA714CA409F05AA4028BC8D3FFEABEE3FDF359EBF6732B4BF5E62943DCB408ABF01478A40EF74BBBFBA122AC060DFC53E44E0143FD1BC0BBFDDB7B4BFFDF8DF3FE2748CC07C630DC09B35BDC088EE62BFE8BB7A3E9ACA56BFACCF5FC04E3EF0BF186A7D3F46D2123FBF545E3F08DA8B4011856E40E49AE9407DB741C040E405C086B834BFCE03A6C04CBD8EBE9EC9B7C098694CC0004C3FBFE4A3AB40614B1DC0A5A043402ABF103F1E5D593F9AC556BFED12DC3E233F23C04B1431C02C18D83F50402A401B2238405AD2D43E92164340F739653D7B78EB405D4AB84005491FC0ED1085C01789643F96FB67BFA55C4740C20C253E29AA444004FBE9BFC3CA943F13B3B73F530D42BFE0017E40DD0F78C058945CBD77DC413E946BE4BED12C4140B38EDE4074FC153F8455424035DB4ABF212CC93F565E01C0FDC33A40C365203F026691BFD442DDBFAAE19FBFF0B4ABBF20763FC0AF8D2040DC6AD0BF94D256C0E04038C020FDC3C0730D09BFF59F34BF9BD585BFE4AC2C407EDFAAC001CE843FB286653F028F0F40D79EA83E8F67FABD0E848B3FDD04AA3FEACA3F40011F0B404F6AECC0B8DC8140F251B73FB54606BF7712F03F9EF793BF430549BF678E59C0585482BFB78F3240B33B55C00FEE4C4051EEE03E31891840CA794B40BBD1D3BEE2912EC1B8B92BBF34FDA83EAA072AC0924C3A408DE568BF8BCDBFBF57BFBABFCE569B4022952BC0B56D1BBF40194D3FC08110BD4A9B5940749ACE3E871392408020FC3F3452343F1458D3BFCD0D8140901CFFBFB72F17414D2089C01B3C4E40D3A6C8BEE8809640B05784BF0BBFA2C07E096A40EE50C2BF345B843EA268E53FD53D1840D2C48640F3DBF2BD946239C0B8842E4022A722BF0F8DB03E5CBDD3C06CB2434065BE573FB27666BE39C9F03FD5CFD63EADDAB4BF409087C0AC1505C06EFFD440078D3A3EA54DA43F0ABB51C0867288BECA43A13E21A0DCBF7775153FE8FBAC40A1339E3FCA1FD33DA0E3D4BF636709C02F0BAB409C122A4092324540D77B8F40A26BCD40A6C9AB3F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x494566BFAC01543EE5AF633F8002EFBC35B6AD3DB4FE503FA64F72BF53EB7DBFB479C2BD0B21633F57D7A83D4EBF75BF70686FBFD9B28ABD1E82133F98184A3FF8C65D3ED1F67FBF9511623FCE5234BF346C12BFB30478BF1127C73E8FD54E3F4CCCB73EF0F17C3F03DD45BF8B5C85BCC8307DBFECC526BFDEB9513F4ABE7FBFE446C23E298B003F3D95F8BE8A8F923EE100A13EEF17583E8E627EBFF2F2183F8271E53DA4FB5ABF14A87F3F7A12753FD23A3DBF8FD941BF6652123F4D344BBFA73B763F6697DC3EECE16A3F96526BBFDF9A6FBD8E957F3FD03CEDBE581A77BF3E691DBFBDE9A7BEA6016B3FC9683EBE0BC36B3FF6652FBE937D50BF8B36BEBE067DD8BED1C27FBF75A0BA3DACB8063F1E1E543C8C42793FC8FED83D5730BFBE3F485DBE74EC7DBFE2CA67BFA5907BBF6D7F79BFB8845BBFEDBB5F3FE6CF39BFC983003FF3AC24BF131B503ED3D0773F9B9453BF3824523F1BC6713F228B6E3F6AC67ABF4C4408BEFE24B9BE3A9162BFD42E653F2177333DCEE164BF2FB95C3F10E67E3FE7363E3F851F703F947E373F5718023E45A9353F90C6BC3E1DE6D8BE87CB32BF414278BFC7DF063F911561BE83639B3EC98F5FBF3AB3843E9A3779BF9A56503F6E3C713F5BFA78BF9C575FBFB27D7FBFEFAA7E3F82584BBF7CAC32BF05781C3FB568183F4357663F357A0DBF7D48C9BE44F16FBF4DAFC8BE85062B3F104076BF99B080BE6BF476BF98BB623F0C5B8FBE2DC4A0BCCA4A253F96E2AE3E2D2C47BE40BE603F08A165BE5EF47FBF5CE97F3FD7417B3F68F6513F1E6DA7BEB03F0CBF992C483F06D91D3E9A755A3F5F37763FE02B733FE47B75BE5B0CE8BE0824483FCBE832BF51497A3FF3A67F3FF63C723E3DD5363FFA87043FE9EBC7BD0503793FDAED76BFE9FCE3BE9CB5793FE296A13C2B6C543F98E07A3F62317E3F789F36BE8BC963BF912A1FBF434F6E3F7B6456BEDAAA053FF91DDC3E16827A3F971AC8BED3F9FD3EA51D703FF606073F083910BFC0E940BE99AE5BBE6B891FBFF3007FBF2E963BBF15BD7CBF03527BBFF6AF3EBF1D12713F8565733F6E8AA0BEF6190CBE2CF27A3E3CC62A3D1E09EABE29E27E3FEF1805BF66D7ADBEF263793FEEA5EABD89F351BF86DD7FBFB7C97FBE0F9E37BFD215813E4217183FDE3B7F3FC83B453FAFD6B53E85A87FBF97C567BF1DC43CBE8B60023DAE2B53BFBFFD643F4907753FACC971BF3A9B7CBF2441943D26CB61BF5CA86CBF268E7EBF0A56EEBEC9FBC03E0FA00C3FAAE604BF02C67CBF00E97B3F73C0723FAC944DBF8455B93E835B46BF6D3C783E6F763EBF9A2FB23E871974BF3502563F61E70A3FCC6A433F032C71BF2F6C0DBFA5BB593FB287EABD53005EBF7A1426BF6E97633FC9E58CBE8394033F4556563D01FC2DBFE0954BBF53BD21BF3AB1AD3D1628093FF92B403F17733EBF105CD53EB89C0EBF0767BBBE913C7E3F0A13ED3E1DDB853EF2BECE3E0DE2BE3D571B653D4B35613FE21C00BF637D1BBF12B3593F0D5E473FFF8249BF4494D93C0B56243EDF998C3DA79477BF3BF36A3F04A67D3FD0FE2FBF3A703CBF3C8D2B3F0E795CBD8AB4403E03EBDCBECDBDFB3DF3511F3F318D0D3F60E5D63D9F4836BFFAFF7F3FF97266BF12D8623E6A1B163F342C68BFAFD17CBF80DD72BF174C79BF5A0819BEF96D173FD3937FBF507A5A3E75ED84BE75A9213E529902BFC70126BFCE7E5DBF37B8DB3E603F4F3FA2745C3FC2FC473F814A483FBA96A53EF6C7F9BDFFF8623F1A82783F71CA133E04DB523FF1B364BF47254BBF768B7D3F113400BFE533743F4A4A6ABF76FD34BFAB7F823EE3E559BFAD51B03E8F91413E856677BDDFC3D93E7D08303F77721ABD13D4CDBEA0137F3F27221FBF02F0A53E5AA4EEBEBA1D6A3EF8124ABF9E547FBF9E667EBFC66C7DBF6C95E3BECF0D12BFE1D9373F147A10BD59E382BEBC0AC93E55387DBF88E56B3F8FC6253FAB2C7FBF7C2547BF938369BFFEA0C7BC5AFE683FEC59A5BD088EC3BE41FD7FBFF9FB5BBF515B6E3F8748FCBE02A57FBF14E3823E7ABF793FE0E2303F1D8E60BF574AF2BD345478BEFB4FCE3E9EED17BFAD12AD3EFBB0A7BE607AAB3DFE183F3FE18564BE3EC5733FA490D03EFED07CBFC789633F47975FBF7393BA3E4C85393E2184753F7D460A3E85D686BE919C9E3E06047DBFA31C0D3FE0E344BF32C8713F1CC0D23D6FE87EBFAEAC56BF6F714EBFDF56EE3EB12B773D375879BFA10A0B3E7F55793F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}
