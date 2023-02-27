module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = chlo.cosh %0 : tensor<20x20xf32> -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0xB82BDABE64E93B40322BFEBFAD63923EAE3CC9C04273FB3F13158040E1333EC0EC3D4D40C6182A3FDBE860BF7C9588409EBFB3BE976364C04D813640EC4A8CBF1899AD40CDC4F33F534F8DC0484DDA3FD49953C041FB50408FB2EDBE8C31083F9999FBBF9067B03F1AFF4DBE250F50BF912D11BEA3565DC05DBF5F400895903F1B8BD13ECEB90F4080513A3F5C6DD6BE29526AC0FC6CBBBF81F5C33FAF55343FF834383F7CC0F8BF3C1435BED8CD80BE348107BF8E2E9A3F204D95C088E355BF31C783C0D66494408B95A4BF6A1EA4C07A552CC0BEE181C0CFC112C0B67D15BF28BDF0BF2E0545C0799912409BC7AEC03513A73ECEC3F8BC347C1ABEA854A2BF1A92EBBF73ECFA3F2DD269C08542293F58BEBE40AF41A7C04AC5BEC0F8ED093F522389BF112260C09644A1C09033A0BB1C76F33F5987A1BCE1DFF33EDF3857C03A111E40477BCFBF3774103FE98BEC3F4F4F48C041941E3F3CE888BF25CB02BF19240BC0C2D671BF2D39AAC0891A7CC032C9DEBFA8E6B3BFF2E949C09F696AC0CA5E3AC081E5AA3F378F283F4965543E1B61963F202CB43FCEF18A3F786CA0BF28C9834077FD833E3245C83F23A9994079A32A40B40BC8BF6FBE4940FF9585409ED3D8BF3E732E3F6C08ADC0D992D1BF4B1C36C01457F63F5810AAC0ABA0F9BF233ADF40682CAAC0FADE0740380B3AC00683F23F2563003FFF457CBF0A80A8C08B2EA1405AA9923FF5FFBF40A7044EC00CF9AD40BB65FBBE86D8F53F59824F3F453A09BFCA710640925B5FC07316D0C06988993F8FECA9409CEA9340ECAC7ABF6558FEBF474296C0F682DBBEEC4B9940361C00BEB0B0B03E24E706BEDFA296C02385A83DD887C63F275C413FB2C4A2BF36CE11C09C5CB1C0719769402E62BEBF72E5FABEE35E8FBF64D428C015F2674056852E3E79BEA4BD9984353FAE33123FF1AD37BF51EFCFBF0D4F4E400F3E41C012ED24BE74BD58BF57285E4006353CBF4EF0663E2304C5BF6A9036400990B3BF7ABB323E210732C057AD28BC3FB463C06F2AAA3F1931EF3E769E483D5C710341FD45BB3F5420CEBF7AB775C07A09BF3F20727040DD6B543FAAC68BC001B2D2BF346022C062B8ECBFE4F12DC0117F56C0687E65BE4E5365C05A879E40C443013F38D334BFF53BE33E5E0B5040076C77C0A4B50241C11B454055DC2F40AFA2A2BF270B914058BD0D40B943423E1EF8FE3E41D4DFBFF051023FFAAFA3BE5FF82AC02EC76A406F979EBFF9231CC0B298C0BF34D205C047A946C0F7D2873F72238040D8DD1CC0A8ED4C4007A0D140D14140BFC44AE1C0A9220FBFC9484E4047310A3C651B4640424721C075CCB13DF6962BC04E0648C02E9C51C0819AE63E7B2286C00FB9413FE1C8AEC0130553BF462EA9402254273F32FDD83FF1894DC043F6E1BFA97BB5BE354E0C3F140138408BF2513E0E87BB3F54DDDCBF77691C4035089640E02CC44023C34EBF322866BE881E0A40649AA84039099C4040C9B23EB68B4AC08E8C87BFD5715840B94922C0D3371540F5DFA6BDF186FC40829C1A408CACAA408D07BD3F8916A2BF76B8203FA5C68FBEB00A2CC0B98663BF0C833EBF9C80FD3F7EA18DC03C973BC01E47B0BE194E88400A5093BD0509A3BDBA8060407E02923E5CFE4DC0B9B358C01DE604BF683261BED2D8E93F4AE6E23EB404A6C0299376C09A046CC03BC2D73F6142E840E4B4583F9235333F58808540BAD440401E5E70C0C901743F781E7A407F071ABFFBA5A33EE70442C01C39C13F12E966C0B3CD9B4074771B3C3D52ADBE5EF7BB3F8C0CBC3F2AB98CC0F4208F3F499545403B6A413F5FF3653FB8FFE33E7EAE0BC09221A740BC8B2E401F8BDBC0E55386BFF1A6014168161CC0E5780640806995C00ACF854044BA2140BBFB5BC08325A440CC9B274077C132C098A244407CCCEC3F8DE9E5BF91A4D2BFA5B55F3FC0754B3FB78B63C00EEBF4BF2C724BC04A929DC0D5646B4074FC25C076FD39400E6C80C04388743FDD2449C035C86640599FCA3E5BB02FC041B87E3F39380C405A97FB3F0D0562C03881D9BE17CEC93F92425B3F74FD2FC03181A3BE1BC1BFBEB3530BC07131EFC01C337A40CA85B240334F993F34E345C0394A4E3FB65866409BFC8C40224F7840ACADA5BFB46316C020D764C0A76D5BC0D8C534C0B568793FCDAC3140DB6AA33FC0D4C93EC7D7B8BEC9E1394039C48940C50D87C035C44940CE9AC83C4C5385C0B7B766BFC080A73FDD4285C0A2409F3F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x31CC8B3F542A1741F37B6D408444853FA09B864322AE68403907DB41E5A41C41A0F14541494F9D3FB6A8B43F22D20E422CF8873F33FA8D41D4FD0A41F9E4D43F7100E34298AA5B40D08A254281F13540A98E5A410DD15141240B8E3F948B923F2FF16840470707404899823F1FA7AC3FE049813F4D627E41C00F84414AB7DA3FCCDE8A3F32D998408D6CA33FF0638B3FB1BE9B4185C7114081D71A40F617A13F7197A23F04046440AC01823F6312843F905A923F04A5E83FBE7D54421A56AF3F7FD3F541328D4E42E337F93F3BCEA842676EED4090B0E741FE19A0407873963FDCC05640112A2E4140B79F406E8AEB422AE0863F1C0F803F9C75813FFB7DF53F32A64E4009C36740DD889A41F8029D3FECF241430832BA42081D42435108933F15C4D03FAADB84410C6B9A426400803F90295B405F06803FFCCB8E3F403F6741A779BE40E42D2840A2EC943F36275040FE4F3741745A993F0A78D03F9412913F4D868E40C47DBD3F4D48CC423F92CD41CD053C40B8530A4089F23B41BFF79B41AA961341180902406EC39C3F67C3823FACF9E23F5B960A40501DD33FF965F23F96E2F541E146843FFAAD1F40A57F73422448E740546C1F4063733B41A00D024212FE33403AE39E3F3807DF426ABE2A4099240A41B3EE5F404B44CB42B0846540DBCE0544D9F6CB42849C874093D71241099D5940EC6F903F6E58C33FC792C142D5009A4242A0DD3FEEB64943015A484117ADE5420ABE8F3FF51A5F408067AC3FACD5923F20B58440D6428341FBBDA6439FA9E73F7E61CA42427E4B429A6DC23FC4CC6D40F3F15A42FFF18B3F75BE7042C700813FA8B2873FC41C813FA78A5D42FF6E803FBDB41D40EE47A63FB635F63F6BC99D405B55FF427CFB99415FD81440ACAD8F3F130CD93F30E6E040001496410EDD813F146A803F248CA13F9472953FA662A23F28BB2840B142494188392441EEA9813F8DAEB03FC3D380419A2DA43FDC44833FF9021C40721E0B41DB000A406AF4813F45AA0141BD01803FE9778C41346401404E398E3F5027803F030AE744A49F1140BD8A2640E40FBA4183881540DB5DAB41FEA6AE3F03CC1D4238232C40A28ECB40146C5040DD6DF3404BA56441613A833F0C11904143C08D42DBAA903F0048A13F88D08C3FA1C64E41ED14BF418AB0DC4448672E414EC3FA40DBFDF53F11063A426C489440704F823FBF33903F45783D4051F2903FE998863FCA78E8401ADC9C41027AEF3F65E8B840DD301740967583403DAB32418215CF3F9D69DB415CFCBA40CAFA4441D8F2AE4343D3A53F31BA0E44C488943F0E2F49412A01803F71223141CA23C840907B803F28B6EA4027803641B1E1534131348D3F3C4D04427D6EA63FCE93EB422101AE3FFABBC54280549C3FC034344085DC4641C17740402620883FD7B5933F97430E4122B3823F35E21140D463394090AEB9401B66594238D465436C11AC3F2F3F833F4E538C406D32C2421B2183422AE2873F37CE3D4116BCCE3F35B26B411248CB40EB40A640D76C803F1126A744D99BB440FB2DCF42946E13406E18F53FB1109A3F7C14853F1C5CEC40BBF8B53F441CA53FFC4B6C40E5352742DA69164160A9873F09950D42CE54803FE367803F61A085417C3D853F5F464841CBA46C4197A3913F9A1B833F93054C40B8C68C3FFB20B3424592BC41A8EC9F415C983240F97131447BAAB03F47AAA03FC9B50142672D23416028AB4171AEBE3F964CC7415CE0973F1998863F72382641FEDC174070AC93411A2E82427A01803F8467873F745512403D6B12404E8922425CB7D83F4CB22F41C34DA63F6834B73F04E78C3FF0B38F408D77B942B3B4F5404684EE43F031CD3F7793CE44CDC1B8406DC38440753A554228F602421488C940AA107941B0F3A8424FB1DC4010230341B91F2D413D8B50408B29464072122C403A10B43F5699AA3F351F8C41F08F5D40187C4041E3928942DC5F9E41363AD74035B81241595DDD41CAF8BE3F56B53941CC609341A8278A3FA318FA40B6C3C43FBAE2904041ED684066D3884181B98B3FF8712140A8E3B13F2744FB401C95863F1915893FC1ED8E40C3605C44DB8CC741086204436953E73FD38730412FDBAB3F4C6192412BE123421CBFC141340EFB3F8544A94000FB8E4147EB7641134E0741DCB4C13F2BF500418C47F73F21138A3FBA6E883F317912410C321442AE2708423E843B41D309803FC8FF00422599B73FDC28FE3FA2BD00425F86F03F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}
