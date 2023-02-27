module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xcomplex<f32>>
    %1 = call @expected() : () -> tensor<20x20xcomplex<f32>>
    %2 = stablehlo.exponential_minus_one %0 : tensor<20x20xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0xEBCC5540FDCED63F226857C090E28340D2091F401BFA3240EB6F6AC04842B1BEC8BF2BBFB4340B408E3DBF3F86AD7C3F8BD72E4076CA4FBF39A2FBBEA79FB34036CF40BFCDDED13DDFAC58C01D5109C0C0775E3F3F1E03C00DD1923F9A3E9AC03F0216C08BFE8F3EE9B047BFCAF813BF97CF15C0245A29405246614054F44EC007BCB43C914841C087304FBE8E810140B4B9C1C0E862723E6CAA3FBF8E7F46BF84A143BD73406EC015FB763DBDE768BF741885C0C1BBBA3F3EBB8B3F95CFD53E5CC26DBFF0E21A40E687423F5475AE3F0C9118BF00F71140C52072C08FBCA2BF7C5EB73E7A31A4BF7FAD8640574B3CC0D61D0F406BA8B5BFC8268E3FAA026BC01373E2BD640216407FCA80BF674297C047D6D3C02CAB9ABE3FE070C0F13C25C04210D4BE561C54C045F817BF5533363F75D8A2BFE9616D40005D51401F83863EB21F16402A3452C033F07B409ADF46BF06FBA5BF7B49B9408ACDF33FB4D79F3F14624BBF48B66F4073284FC0469E1DC06549843F450193400D470DBF745972BF61FF6DC0298808400ED4A4C00942F83FC2BDE6BF4966FE3F1F1904409BA622C0DE2E4D40D1D549C0B55AF1BFE3BAA2409000E4BDAC7C23C084354FC04A9D574011DBB540E5DBDA4064D982C09D0702C08E48ABBF9A48CA3FCA2C4DC0A6818240C08999BF5532FDBF091489BE6037B0BF133B5940166390C045DA49401AF657BF1CFF3C3EA31086C018F4DC3F238E20BF145F4C3FA13880BF6073C240CAAF3E3F73BEE8BEB972ADBFC9CA00C0EFEF01C06DFAE5406B1BB140A5A0AE3FED554D40C15583BFA7342C40ADEBA0C05230733E12628E3F88AE7A40A5BDF0BFAF55A8BFE9D50840762F1540B4AD7440541A1DC0F45D09C0889DDFBF20A393C0036E113F0F23CA3E203F8BC0DFAE4BC0BDA90A4062BCB2BFBD51E23FD5AF4DBF935EAC3F5E16204017AAD73EF15704405FFF9C3E2109203F73D72D3FBFBEFEBFBA37A8402CEED13F33C3BFBEDD93C93D256BC13FEC28933F88445DC088B741C0E12543BFAB794FC03D3323C01551E13F33B64F40E40DAABD87F412C0BA6289C04281CABF6A1700C0DC4CDABEA67A64C01B1845C06A7B894065E12D40E720E13F9C0320C085E2E13E5DDAAA3FC44C5C404E9546C0C9DABBBF328632C001D187BE9724A8BFB98526C0E0B10F407C4980BFF8BB4240F3FD5C40B360ACBF5E0D1ABE0E84CFBFC88C5E40494575C0726F294043789ABFC5FCB2401B18B93F8D82F93E3298E0BF019017C04488DE3E046BBCC089B42B4073090C3F90FB1A3FB01B2FBB790502C01B2435C03A58543F05B668407EFFD63F90927A3FC5E7B2BF5EFC1EC02F0E05BF552E8DBE052980BF24CCA9BF3DDF0840C016A73E34EF353FD9C3BB3E930791C06CBA18400E74D1BFF96872C09A1718BF308A2F408C9A68BFAFE5F0BE72941A40C87B00C1157A58C0D6DCD1BF4288C03DFDE3A140005437C08C2B053FA1F4E43F4BCC7F4028712C40933784405D4DDABF8EEC8E3D20866740680C5CBF3F781340D6DA15C0217A5B3FA56AE83FBC1C0EBF53E1EE3F2C076CC0849D164051C92D418E308D405103FEBFCC63A6BFF734514081B3D73FA06CABBF83352CC081F9E4BEA250873E323916BF47DDBEBF1C6BF43ED4EEC13FB45583C06C4EA5C0C42A6C403A19753FCAC9273F1031E0C08201103E7B3093C0566A67C0C785834093E43B40E8675FC02C3FD33FE304273F59F33A3FD7BEEFC03A368A4016A8E43DD45602BE920A42C00B4607405CE00BBFC32118C04F56B43DBB486BC001D2BE4056E29C3FE1352AC0921B2F405E566840710AC53F97DD16BFB064584062C261C0077DD140F7D60F40C79FE03FEEC1A33F5633853F7620683E39C3A7BBF95E4AC00D6DC5BF17AA2140A2F96E406AA787BE63CEC8C0C8B7B0BF6C22393F0FBC80402D360EBFC93860403F39E2C0F63DE33F68BF6DC0CEF006C0F9492A408F8B0840B7A929BF00B1B23F5F23A040852576BF88BD5FC0F94633BF34F85F4053A4B7BD06DC29BE0C42C3BFC132324030231BC09E78933E99E21E40EF92F5BF0579FC3ED04247BF8F12874055D70DBFF9D2C7C0316165C0AB016B4095F8B9C04279FF3F7C86AA3FE03D44BF479DDC3FE80E9F40CC9100C0E2F1A13FE13F3AC0D8A3E63FED565740EBC893C0AADD8B4042D588BF5102F13F3AC907BF03320140A1B36DC0B6211540DA453EBE363A89406F093C40A4FD07C0C1BA8640AFE4203EB3CE0EC0D5BF093F54DE4FC018DABC3E179C273FC308F5BDBBC482407E81ABBF57471F407D75DF3FC1FC32BF8EEB843FC296A6C0636B99400B2C844054A8ACBE817A37BF5F5A31C0B9A61FC005E20240A1F1BC3D2A9E46C04BCDBC3FE8025EBF2905043FD2403940C98E944068767140F9515B3FA30AFBBFDE050AC04FB06CC0CAA5943FB9B147C0EF4CB0BF7213D23F9ED29840B6268D409DE808C0A09E8D40344D9FBEC853113F621A09407A5752C050B9283FCBFFBEBF2924C94063541CC09FE68F40AA62253F468E0E40118D8CC017F1B2BF46450CC058200040C7D70E403375B940E92B9D3EAD3752BFD6DFB5BF6B5EACBFE1655340E68DD43F84E916C00FD88B3F884986BE22EA623D9A5B2B3F2D0CED3E21936B403A1DB2BF4201FD3FAB2D45BF28D6AB3F27EEB5BF1D9789C0515D2AC0D6AD26C0353C4BBF9FC519BE60167AC0960E81BF4956AB3E1B3418BE448C66C0F4978A40C87D0AC068A15D3EB5ED23C0534E4BC017BFB93F05811040316C7140AE148A40F6B79040822D6240512BD63F1F95B43FB232944060CAE6BF7B6379BE127A6EC0B09D143FE395A340598BBEBF731328C093B18BBF1164554033453E4087B9B13F093A4FC07447013FC27971C07C42EABCF63385BF49779C40EA77393FD3206DBF04653F3F2897A9C0626CF2BFC79A773F02C24FBF2AF2D6BDD4C4E7BFB48A323E3278C3C03E8EA540C0B500C1AD3F1ABF52C4BCBECE69B63F22790640DD6DDD406FDC943F1809C13E8EF879C040252E40458276C08D148C3FEED8C54020A2FF3F36843CBF103D57C013AD343FF9C46940E88EE2BFF8A758C03AA1873F0C0BBAC0D500904064B87C4045B16D3F04C67EBFD428033FB1B2A840C4F8BCBF56D69D40B9912940FA06B2C0144681BFE8607540164174BFD85B073F142AAA3FF82792BEA0E618C02068CCBD3E3AC4BEA1DC913FC9C972BF362BF9BF56443EBFCABFC23F8825B93ECC3D16BE2613FB3F199680C0F29FBC3E0D7FB840311A5B400995FEBF18ACB7C0EC7EAE402CA96B3F866DDC40530456C01BBF4B40A3F0D7BF606012BFE23F3240E13B8ABFB6212940029838C06E1DFFBFF512A13E41D3BEBF709D9F3F029400BF2FF0D23F3F616140399103BEB9653E3FDA71904065DEE63FF46F0B401E4973BEF0E7C0C0F55D673FEFAB8B3FF6AD1FC03B1A8FBEEF18A83FAD271A4024ED88C02C7F6F3F6C900B4068832040F9B87EC0291FD1C0F9E1913B647A803FDDD256C032040940EC2207C1A30F5D400F230E40C44209C02B215EBE942FEF40127C874013DE9BC051528BC0E723E3BF2F14BD403D9462C08F78363F5FF817C0C919964099DACEBFC1FDDABFA71A1FC0FC7EFBBD0D32DABFA12B0440D1EAECBF93D4D1BF957B903FF2673BC02CEF803F472C5E3F00655E400B6CD03E1CAA00C1C291FC3FFE44CE3ECCAB43C0C9AB86BF3FF045C0642AD73FF5237D406057EC3FE865A8C09388B73FCC6B8940B4A44EC030D7CFBFB3082A4036FD533EAF22973FF21454C0616495C09DA2B240752F3BC0D3F6A8C083EE71406A1ACD3FEF265D3FD2764FC0D44F2240D0FC56409A6CB33F306F8CBF1D204E40CFE852BF9F444CC0A63840C075D41B3F0AD169BF07D428C0DE7992BDB8B2E8BE7937AB3E7BAA3AC0D6F5A8BFCFEECD40EE3C6640C3DBFABFA0C18840AAB93040CDE3C040901998C0E4554EBF211EE7BD40F836401C4565BF841313C06586A240926246BF0A4019C04E1F00C01C4C08C0D91A9D3F11A6F13F079B2F3E7BB9F74017630C3F29FD37BE50BCF73FAA802CC08D2B4E4067B8FD3F3F1DAA4051514DC003BB87C01E9D01BFEC4C453EE2F0DD3F63261A4071228DBED483AEC01E6D29C0198912402EA52BBE084D9740BAE18840CC563C3FCBC94940FC97B240DC1DD53F1B3F343F7B094FC0B73911405FEE8D4054DCB4C08FD8124027F24A40E99019BF686DC23F01D6C2394FD01FC0087A5140C1FAF9BE37423BBE2A968A3FA52257C0A6A6BFBF90F1143F49280540D12ECFC027C22740A40F933F885D9A40C00ED93DAF1BCE3F255D223ED581ACBFE403B5BF7B525740B14E4EBF4124AD3FCB389F40F43E02C052D62EC035EB10C00A8ED5BEE70A5CC001716EC04CA071BE682F69C04465D3409EFCF63FD82143C06F0C19C04E2D334057749D40A8FE5EBF6FE495406EA080BEE22173BF5882A9C08EE98DBF7284FB3F5B0D0EBF4E2D483F2182CE3FA5E8953F5EC49C3FFD0720C03506263F527D7340E6883B40"> : tensor<20x20xcomplex<f32>>
    return %0 : tensor<20x20xcomplex<f32>>
  }
  func.func private @expected() -> tensor<20x20xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0xB3DA80C02698E0418F7682BF76F0EABC96B144C180E7814089D279BFA69F0EBC922EA5BF3F67D73EAC50BA3F00E96D40C12C1941804F32C13C3F05BF537BC2BE9A1608BFBF4C453D2D5B82BF65CDE8BC213106C00C8307C0884F29BFE05948409F6668BF982ADA3C5DB71DBF543A80BEAFD68ABFAE703B3DDE8D0AC2BDC64640EEF100C068D6FDBD07BCADBF320A3C3F5F6A7FBFCE60103A168729BFD883A9BE2E00E6BF3FF9053FB423B2BEA2A356BFA78D7FBFE74E7E3CFC90DC3F56A89A3F58F7A5BFD298853EF60E0FBFA6E3054055F8ADBF0BF9D53E41487EBFC910B2BC7CF517BFE494AFBFF1DD85C2694A55C11F87D33E6E0114C1299B67C0619BC43F5B08D0BF3A0D243F60A57EBF9F2CBB3E90AC7FBFF902D0B9258482BF7BCF49BCBC55D3BF4B80E83D74EE14BFD8B2B83E7A3E9EBF133C1ABE6D88C341DBF2DA406C5835C19E3DBE3F1B2A0E42C1AF0FC2525642BF7B8004BEDF20903FB4E8CB409A9AAFBF9F5D83BEFFE983BF3F13CABC9987AABF39A232C01BD929BF3843EFBE0DA881BFDD26A83C6F4480BF0F1BB13B798A88BFAA5E1A3ED307F0C0DD5F8EC0906CCDC171AC983ED2D871BFDEB310BEAC56DFBF6BBEFDBE3EE584BF78FD10BC608B78437E1E1B4332F480BF48DF75BC0B5280BF904E863EAD1283BF30B905BD8C478FBF12AA8DBE07385ABFD43040BF36ACDDC0AF91E941D9316941F4038CC1B6DFCCBFA460853F073163404C0D53C02FF4493ECA92EFBF4F999F43E49E9343AB375DBF83B81EBF7A9687BF6B5DF5BD08DF7144F5EB60C4FFEE9CC074AE85BEF14AA9BF80EE1F3EFD5E7EBFAFE1C93AC9064BC0D53F08C0E82076BFBF0D17BE000EDBC0A2B7C440857D11C28210E8C1829F82BFFBB8EBBD3FDC7DBF9DCEAE3BBF15C3BF22B9B13FC4FA82BF42A00C3DE53586BFC283783EE28466BF6686DF3E0A202241EE9C9F401340D14035C51840C371E83EE436963F2BE16DBFD36EEFBD84157340856CF1BF5E246FBFDCFC8C3F790980C0EFF47A3F9B0677BF021A09BD502784BF7699B2BC480CD9C0425F1ABFBD26CEBFE74930BF0C0580BF83C95FBC3C8060BF1EED64BDDA9883BF9DA6E4BA72CF87C2FE0DF241FFDBB4C0E44C5EC082F222BF2D77C13F50ED00C201E89ABFBCB09BBF45CEA2BD960E4EBF05E53DBFE6EC85BFE9296D3DEEC0AEBFA367143D8DADC040AB70F6C1D58C85BFCCF65BBF4F9ECFC10402A541F5D58040341553C126B30142254685433218A6BFCFDDCCBF0A416ABF9686213D745180BF75E5A03AF688D73E02C37B3F89C3B8BF20B064BF4ACD75BFB539323DBCEFA3C086E01642C8A80ABF74C627C04B786DBF9FA729BD8B3817BF4EA223BFF44492BF7B1D653E7A694E3D5472673F491BA1BFACB6B53FE42EDBBF639C2DC10A327BBF94B74FBC29C808416FF843C135C8BBBF8367D43E5B0A80BF9ED1A238C9894EBF903C953CA37018C34D482CC2798EAEBFAC3FD23F0A4C48C2AE48BC414CD415C131E676C206FFF9BF138EFBBE3D46A4BF2C06A13E94E26FBF11F1943D77218740674D4FC0CDBCD0C09DF1564044E819C0BCD326C14F9208C2F1FC96C2CE9AA2BF5DB10DBDF15E733E30FBA7C0EE5770BFFD4BF0BCE946AD3DEAA038BF07D04CBFE5D3D33D196166C09AFA6E409C9F80BFE41043BBF957883FFC39CB3F2FC57FBF923905391F2581BF6C79963BCBA972C27D414741624F80BFE3E4F83C766CDC3EA4F4A33F000780BF2FFD06BA9B62DF3D775711BE4F3083BF2021293D877DB5BFAA36CDBE2445F8BF67770E3F0E8D0243E2F2B643213B88BF3E2AE23C98513D3E59D1164288F9C4BFCBC506BE03BD78BF4F94FA3B25DE2EC0A2DD1441304C513F357546405043823E5172D2BB31B17FBFBF592DBDA8F735C1C566DFC0C55D6EBE6EEAC83BCFBE4FBF44612A3E96DC39427BB3EBC13FCFB3415026BCC1A5AEBEC0F0DC4C40BEC78DBF5F09663D0905B540CF43A6C0725B263E059177C062E5ADBF05D7073E3E84BBBF151832BE7859C9BD81941ABEC2159ABFB2C89B3DEE426ABF1E18CE3C8C8EA2C07B1634C1B7EA293E8B2793BFD4A763422A540FC27B3980BF20DF593A80250842D0F08E41A72B3D3F66CFE440820C89BF881BEB3E628978C2C27F02C396608EC01F1950BF4E25DDC032A4ABBF566C80BF475D18BC867E8DBF894DA73E159DA0BFA8C3073FF62582BFDCD5903CE3BBABBF1BC341BFE6EE2EC1B27080C1DA1783428DAF2841116568BF6469613DEDB876BF4084653CCC2D693F693C6BBE20FE4941E1CE67C29D3546C0DACA3D41ED6B3FBFA23FDB3E93E27FBF7719B3BBC0CA66428BA4A4C1DE44BABFAD0B35BE0CD384BFBC5F963D5F2206C0A77F2BBDA7F6E93F3C5255C052EF27C096AAD13E1896A9C2BF0675C21DC1F2BF87660BC053938CBF386E7A3D333086C0C6C88BBD9A4582BF87D4803E28D410C2F97EE2C2654384BF5C4DE7BD83BCC3BEF6AEC93E77DD16C19C8B9D3F292359BF31A9F6BF6FF9CDC389B2ACC3F24E8D42AD195842BAD77AC0D2DF0C41AB6692BF74CB4DBED277B1C0D0FFBA403A3B9C439DBBC642602E6FBF94B6DEBE89DCA0BF17522BBDF14997C068B76DC0889EF03F0DEC45BF99382FBEBAE4273FE54017C00ADC50BF03918CBFDAFB693E5C2D65BF8CDBE63E81648CBFF36E623E59AD87BFA25612BDA9910DBF81838ABDDC417DBF7F348BBCCA99C33EC4E953BE724C81BF0847CFBC524763BFB016CA3C11E089BFEA5A313B46416DC0992C5340C8998EC1735B20C23E20ACC2E8D40CC2C19C1ABE0A5BA840D9FDC4C1A7C2C7C27DABD3BF7A7BDD3EB8FE9BBE73B5D2BF782299BFB1CFE3BDB12EAABF12A683BDF1DB2240B6CE99410D3577BFFC8A9B3C7C1E7ABF2C412CBA111070BF3F05B2BE89A1753EBE28D3BFC4662E3E5809E13F6D226ABFDEB9FD3D3BEB0EBFF3973EBDB5C456BFAE68E83C22BF7FBF888602BBACEE7FBF41B33EB9764B66BF4D2F2F3F0664B240E07E9B40FDC3FC3F03C5963FFD5982BFF69E063C87817DBFECA59A3C614849C3DC7ADC4334CBBBBF1474D73D261031C00B997DBFAE2895BF8A52283D6060C93F8D09A73FFD077DC28E0E82C2D449C13ED4DD07C05DCDE9BD5C16B5BF0D3C73BFAA5064BE4C331A412A3A154171E2A3BF2C1E6EBEA2DD2ABFEBF3463E900E2840DA3488BF52A368BF37BA15BCDA0E37BFD3851E3F013392BFAE7CB8BE81FF79BFCD32F33E2927D73E9BEB56BE488AB1C00FE3AE407E17833E321D37BF4D4B57C1CB43E0C1CF717FBFDE881BBB5B4A883F57D4B63F878384BFD4FEC1BA962858BF8819CDBD6164D44011A164C1756969C1AFD864C073D35EBF3EB82C3D67A86DBFCA9B5A3E88F885BFED751A3FAE3B024206BD8AC0B119B5BFB0FF03C087F98EC0237D9F40FBA072BED5934B3E6B950E3EC52E0C4047B36BBF5670BABC5FDD70C051361F401EE57DBFE6BE363C282602C135B5A7401A5E7BBFD99598BBCDBFEBBE28E5583F5F6882BFAA5CF03CB20680BF6C748AB8911AC0C09FA6F7C0FBAB33BFD66B3F3FB32E1E41F83D88423F5580BF734E4EBCC42CAAC303F50E432CFB1DC0F830B5BFB4AABDC0D99FD9C2DF5592BF369EE1BDC11B8FBF276460BFD28B4BC0A988F2C015BE6ABF81AB333E76AF78BF973A393DD50B50C0AA7F47BF38DBA3BFD9EFBCBF5CC4B34080803440AB087ABFAF3B27BD7FA380BFDEBA383D2D5973C182F74842A2D17FBF2C3CA83B4C0594C20645CC40F65396BF622DBC3D564508BF989D913F4C3480BFDBDC143DB93C82C3D4A164C2F88580BF2EE746BBE05F0E40559A71407A1D84BFB24DB63C14BB75403AD8E2418298AABF9B22D8BC5C16B8BF30F9B33C729475BF3969E83C7CF9ACBF584446BEB77427BE4941D1BEA42917C0938C9FBE7D673CBFF5E8253D248B6DC1821C07C2284F87C28A24D54136417E41F63ACF4310630EBF09044EBDDC701E415DD759C1C7C876BF3C10C0BDB044ABBF4949A0BED22A89BF348DEABD9D1704C0D47F4F4073DB5DBFB1FB963FB4DD333F67569EBE98F5E7C0D4543FC0445C30C116C7B741E0214CC3F26F58418CC77CBF3096E4BB183499BF7C23993F502D1B412AAC41C07C7B80BFC01905BB9AB90B4149CDD2BF318E42C21620CDC23F8E45C08404C2BC02A0CFC16516844333D840C0A706413E5C4A69C090D614C12D4C80BF426C2C3B9B5E95416B4357C1C1516440537ADE3A52728ABFA1D430BCBF05CBBE6E90E4BD886B78C037E1243F882550BFEDCEFB3D479EDB408DE6C2BF2934944071BF4841B681F54246A55241A93B7C4078434A3FF99F75BF246883BEAC3098419DE4A6C15516D93B22016FC048578FBFCD3955BDC2AE67BF40662CBD236E83BF731B913C54A0D8BFEF8DC23E085A82C362202D44767084BF436504BD875219406FAC80C1698381BF0530D6BE8D200CBFB0F121BFBD6D7FBF4DD392BB540EA240C39270C090E68BBF47C20B4076C8C03D76354240C4426FBFDAFB4A3D4B9E33C239BA1641"> : tensor<20x20xcomplex<f32>>
    return %0 : tensor<20x20xcomplex<f32>>
  }
}
