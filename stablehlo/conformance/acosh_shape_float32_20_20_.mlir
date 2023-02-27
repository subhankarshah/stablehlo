module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = chlo.acosh %0 : tensor<20x20xf32> -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x04D5963F01988CC06828783F741892C0495F81402C71503FC9FECF4068E8B5C058934FBF0339F43F2A82374022EC2E406021824012A485BF96E7A5C05F87BA3FF4F0D340423A2CC049E231C0E89BD9406900CFBEDCCC9D3C5D9921BF28B535C0B8D5F43F8815993F6A4C6B3E0F93E3BFD19F04C01558C1BF021EF0BFB127A5BF96B59640A92D453EF83C46C0957FA8C0085C1ABF0C160BC03EAB5DBF0496654006E999C05B6C5840D301B0BF194AD53E46CC3CBF6BC9993EF05694BE3FF3AB3FF5F407C04BD6E6BE989A24C0C2E44E3F89C14DC07837834001B227403A4933BE6B6FA33F569EA0C03A675BC001AC2040A74A553EB7A44E40DBC0C83F202A0CBB7A1D0340A0E18540C0C6AE406088E8BFD9384F406A6080C00A436740334E8C40FC480D400130723CF52B2840C39F92406A75D73FCFA52EC09A1134404FDBB63EC9609A3F507A8140DED524401D3BDB3DDC72154087051BBEF5DCC0C00CDAAB3F50B1623FA6DDEBC0A80139BDE41B3CC072E4B540CABB9DC0FD22C240C2C650C0EAD635409B064DBFDEC69DBF0A1D2A3F1DD4783F8DB509BF50CC42C078C45E404F6077BEE5E026413AE5AB4092E1CCBE530526BF515EEE3EC251F3BF128E98C045B2F73F40827AC00EA00440A2D904C002D3E53F9DBCB2C021DF37C0E80D8C40CF2D51C09EEDCBBFFCBB15BF4A0DEFBFB10FC1BE4662823FC669EE3EEC3740C074BF16C061886BC02836D93FD008213E58917540F05A07C002BCDB3FDD040B4031902A3E9634804050A002C0832DAD3E68290FC07BF0D1BFD22D2E4098895F40191D9840489D2E3F9853A2BF1CE8ABBEDAD3ABC0D9AA1FC0852EA73FD0AD4B3FF9F1633FB3D3A53FF68AE0BFBCAF25C0440742400848943E2102613F0499F5BE0DFA973E884D993F05E9053F2B1E0040D0AB6B40DCEFD7BF6A918F40B3E52840602D25C0757FDA3FC8A7E4BF56AAF73F49A91D40C772EBBFD1E9C13F4302734024F77DBF76B746C0C9C6EBBFCD2592BFF73BAFBF52F79940DD958B405E138BC0CC50033F3DCB74BF56AB2FC03DB550BF1F28D1BE5D04ABC00F3614403EFF8F4055ABD13FF7CAB3BF25CE183F13312740812736405B9BCF40DD1DDDBF91B0453FAECE053F1358453F3AAD5240133A403D49022DC0DCB73D3E6DDD983FA6F6064116A640C08AFDC43FE7316940E32754BF4408EEBE3BE29840090D194051832CC0D98E26C04166F2BF156EABBFE93CEA40AD9619407C8B48C0BFAF0FBF66EE8AC0F054513DC3621B404A7D124060861A40C9B81D40542356407BC6A5C000D3863FC4D7B13E6EDF6CC049347AC0A3FB45BFC015073CD5C20F40B0B1243FFF5D72C0A50C634061D61940E65D2740A7B663C0122D6B3F8DE528C0700B5C40F4C43FC0FF3F44C0C09E19402775BBBED08F30C0715F473EFF5137C0F1FB893F876728401704244017928E3FB75924C02C336F404C8A3BC0C00FB7BFBEE408BE8BF725C0FA6DCCC06A098DBFCE6AC2BF14D4793F5B3363C041BD8F40315C8240569E3CC002C911404008383E92C20540F7DAEFBFC983A340A5EF58BEB95920408F4B9340C96467C0D277183FB47E6240D016B43EA2FB33C0AC7ED0C019DB8ABF2BD87340C6D4A240D2A447408887BF3F4EFB3D40F2898DBE4E079F400191A63F64688ABF59E2683D959943BFCC9294C090F9983E90A9C5BFD684A940B391FD3EE0B5AE3FA62546C00ED7B7BFEC7D3AC0CDE27240EE23B7BE17CA1F40C5949240D0A3573EB79452BF56ED253F561BFBBF8B0651404E78CD409800793F6E34A33FE509C63FB0982BC0E47D4FBF9DD69DC0F57E2C4086487A402628F63EE79A86BFACA04FC00F4FB9C03AD090408F75BA3FB7B3064042975140B836E63F17CE2A4011ABAABF61631E400A0B6A3F0C0C22C0756DC3BFE546EBBFB07D5EBE204B483EB905C140E51BA04011EAE63FE3AAE7C0582CB7400246073F2E7C9DBF644624BF800A3340DF7B7DBF43285F40DE7DB4BD026D2040BE73ABBF1FD49CC0D4C0F33F10B0A8406E31FFBF057D4FC0432B8DC0EFE01ABE02727240D793AFC0BC1716404D0B23404C01C53D5F68D33FA0301CBFEE6B52C0A7B774C00012DE3DF0A8EFBDC5F7983FB893A6BE89CF553FC1BBB5BF6A983340D8738F3F5A6E88BE92320A402398F03FED29194048AA55400184F13FD72C8D3F0FE42F40F3C766C03D7472406D36ABBF567BB63F7BD20B40C7062A40A03D95BE98D7C7BF1941BCBE489EF83ECA33E33F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x1BB8163F0000C07FFFFFFFFF0000C07FBAC30440FFFFFFFFEFC523400000C07FFFFFFFFF628DA13FEE78DB3FC4E7D43F842605400000C07F0000C07FCA546C3F7DFD24400000C07F0000C07FC3B22640FFFFFFFFFFFFFFFFFFFFFFFF0000C07FA8EDA13FBCC21D3FFFFFFFFF0000C07F0000C07F0000C07F0000C07F0000C07F95CD0E40FFFFFFFF0000C07F0000C07FFFFFFFFF0000C07FFFFFFFFFBBAAF93F0000C07FF5C6F13F0000C07FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE97F4E3F0000C07FFFFFFFFF0000C07FFFFFFFFF0000C07FFEB205402515CF3FFFFFFFFFB8583A3F0000C07F0000C07FA422C93FFFFFFFFF2391EB3FD2B7823FFFFFFFFF561CAC3F6B060740BE7918400000C07F7DF1EB3F0000C07FD0A2FA3F241C0A4089F2B63FFFFFFFFFAF79CF3F86000D4040348E3F0000C07F7AE2D83FFFFFFFFFCFA6213F85D104406EB2CC3FFFFFFFFFF7F4BE3FFFFFFFFF0000C07FB0474E3FFFFFFFFF0000C07FFFFFFFFF0000C07F63121B400000C07F984D1F400000C07F4B39DA3FFFFFFFFF0000C07FFFFFFFFFFFFFFFFFFFFFFFFF0000C07FAFA4F53FFFFFFFFFEF454240A9641740FFFFFFFFFFFFFFFFFFFFFFFF0000C07F0000C07F59ABA33F0000C07F82C9AD3F0000C07FA44E983F0000C07F0000C07FFEFD09400000C07F0000C07FFFFFFFFF0000C07FFFFFFFFF0253453EFFFFFFFF0000C07F0000C07F0000C07F9A7D8F3FFFFFFFFF574E01400000C07F8A50913FF89EB43FFFFFFFFF7C2A04400000C07FFFFFFFFF0000C07F0000C07FC351D43FB71AF63F0F690F40FFFFFFFF0000C07FFFFFFFFF0000C07F0000C07FAC84433FFFFFFFFFFFFFFFFFE042403F0000C07F0000C07FB70EE33FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF106D1E3FFFFFFFFFE4B4A83F5926FD3F0000C07FB79E0B402D12D03F0000C07FB46C903F0000C07F8FA6A33FAB7DC63F0000C07FDDC5793F829C0040FFFFFFFF0000C07F0000C07F0000C07F0000C07FDC3310408CC509400000C07FFFFFFFFFFFFFFFFF0000C07FFFFFFFFFFFFFFFFF0000C07F2DC7BD3FDCD00B402ED6893F0000C07FFFFFFFFF7DAACE3FD675DA3FF1A623400000C07FFFFFFFFFFFFFFFFFFFFFFFFF9E2AEE3FFFFFFFFF0000C07FFFFFFFFF47171D3F259C34400000C07F381B7F3F90BEFB3FFFFFFFFFFFFFFFFFADBD0F405C52C23F0000C07F0000C07F0000C07F0000C07F8E752B40D1D0C23F0000C07FFFFFFFFF0000C07FFFFFFFFFA873C43F161EBC3FC3ABC33F708BC63FA45AF03F0000C07F5974A63EFFFFFFFF0000C07F0000C07FFFFFFFFFFFFFFFFFAB6FB93FFFFFFFFF0000C07F782FF83F290BC33F9CCFCE3F0000C07FFFFFFFFF0000C07FBDFFF33F0000C07F0000C07F37D8C23FFFFFFFFF0000C07FFFFFFFFF0000C07FF1EEC83EACAACF3F2501CC3FB109F23E0000C07F1220FF3F0000C07F0000C07FFFFFFFFF0000C07F0000C07F0000C07F0000C07FFFFFFFFF0000C07FC0B20B40544405400000C07F716EBB3FFFFFFFFFDC07AF3F0000C07F66221440FFFFFFFF18DBC83F2E4D0D400000C07FFFFFFFFFF4DBF73FFFFFFFFF0000C07F0000C07F0000C07FCCD6004065DC134056EBE63F6B89753F2A32E03FFFFFFFFF595112400A0C423F0000C07FFFFFFFFFFFFFFFFF0000C07FFFFFFFFF0000C07F727C1640FFFFFFFFE98B543F0000C07F0000C07F0000C07FEA930040FFFFFFFFC85DC83F9BFB0C40FFFFFFFFFFFFFFFFFFFFFFFF0000C07FE71BED3F5AFB2240FFFFFFFFDAC3393FF571803F0000C07FFFFFFFFF0000C07F66FBD23FE9900240FFFFFFFF0000C07F0000C07F0000C07FD72F0C4029336C3F770DB03FDA78ED3F6C91983F869FD13F0000C07F8922C73FFFFFFFFF0000C07F0000C07F0000C07FFFFFFFFFFFFFFFFFF7ED1E4096C212402109993F0000C07F2A871B40FFFFFFFF0000C07FFFFFFFFFBB19D83FFFFFFFFF7EE0F53FFFFFFFFFE0EBC83F0000C07F0000C07F5243A13F702A16400000C07F0000C07F0000C07FFFFFFFFF137500400000C07FD290BF3F872DCB3FFFFFFFFFF82A8B3FFFFFFFFF0000C07F0000C07FFFFFFFFFFFFFFFFFE3671D3FFFFFFFFFFFFFFFFF0000C07F2C86D83F0E1EF93EFFFFFFFF0AC4B33FCC4B9F3FF36CC23FBA0EF03F9ADF9F3F575BE63E16AAD53F0000C07FB07500400000C07F8088643F7573B53FFEFDD03FFFFFFFFF0000C07FFFFFFFFFFFFFFFFFC488963F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}
