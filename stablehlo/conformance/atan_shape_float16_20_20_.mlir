module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = chlo.atan %0 : tensor<20x20xf16> -> tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xB13D17C0FEC10A3C4B40423B6DBCF944973E9D37902D7542283946C567BBDD3C05B5654073B83341824851C067474141FCC0DE4047409AC425C4C94383C3E2C2F44571B9694190418AC0FBBD46403BBB2F2B3EC5193B11C38A443B3FD3C484B9BA4050410CC83E411F384640C6C5B59B3044D0C108C314C529C0EE47ADB8D5456C3AD03D71C03DC424408AC102C2743605C2A5C151424DC17A3E5FB94F361BBB8EC3B9B4CDC685B7A1C0DF37B9415ABBEA44654761C4C1A8A536BDACBB3E9AC510C203403F39A3C327BC95C81BC4E143C6411AC842C0783866C0123FD2402F3E2A4352454BC08642734088BCD140573B5CC01DC21341DFB1D3C153BD39353644A5424CB083BE33C346C41E2540C0383D39C09CC680C438BE8634A1B6AC41353E09BD9C39DE45033E22408DBC5445D7B9A7BFA93D58B6EC3F4AC476B850C3EBAC734066BC0D4319C4352AF8A8AFC26CBF0842B03760C2F4C404A55440A1337B44623C7A48F84577B994416CC158C3EEC45E3E24C065B70D3F952467457E3FB538B8C3324038BC73405EBD09BE04B6C6BF58BF2E3BC734B0C1824212316B43B8B7383AC0391CBE19C1D9BCA4C0F4BAC4C2673AE43F3DC07DC0C14253B936C205C1854066BAFDBD153DAEC2144206418B3FD4C374C5CFBD76403F3B88C22C40E5BE5BC40CC56E3B26C223C7E5C4BEC54A4176C3A0C3C136C5BA26C694453AC4D743BC40603DCCC46DB9DF3F504174466C3E04BE7438F1C2DC418C38873C894336C51FC20A40083961B9E33D58BF2C41A23E314386C1BC35EEC0EF3C1A3C4D3B1AC2BDC5694627C6AE3F1244D2458F44723EDEC4004265C25342D9B508445EB433C290C002BC4E3E29C4FEC6294364C556BE66C01BBC5E3DF23EA63D364333446EBBF343213E91329C321B342ABE123D47AE6BBE2D401A40DD3AD74048C77546053A4A3CB9C0423D85C3C3C0F3C070B9ACC1E7BCE63C82419FADA5C305C22A3CD9C0DEB86BB4BABC59C49A327A43EC3C963FC5C10A444B38A23A7E4473BE07408C33FAC102BCEA401AC56FC31FC0FF424EC3E7352FC4B6C68831344046BF62C2BE3975C66F42CFB523C3FF4069387DC1623BC139E6BE9033B1C05C3C0044F0BE9DBF"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xAA3B77BCFFBC523A8A3CE539B0BA7D3D1A3C1B378C2D153D953889BDF9B90F3BDDB4933C0FB8D03CD73D8CBCBF3DD43CC2BCB93C893C6DBD56BD473D3EBD27BD9E3DC7B8DE3CE73CA0BCDABB883CE1B92D2B87BDCE392EBD6A3D433C77BDD4B8AF3CD83CCBBDD33C9C37883C99BDB59B583DF5BC2DBD81BD7EBCC83D3BB89B3D6939BF3B97BC5BBD7C3CE6BCFFBC223600BDECBC0E3DD7BC123CBBB80236CFB93FBD98B4B3BD08B7A7BC5137F03CF2B97B3DBF3D63BDC0A84C36BBAC233C94BD02BD6F3CA53842BD6FBAD9BD54BD4A3DF33CCCBD87BC133893BC393CB63CF93B323D8A3D8ABC183D983CC8BAB53CF03990BC05BDC83CCEB1F6BC69BB0C355A3D1D3D45B014BC33BD5DBD1E2586BC563B84BCAFBD69BDFEBB693449B6ED3CFD3B32BBE4389C3DDF3B7B3CCCBA8B3D0BB95BBCA53B0AB66A3C5EBD12B837BDE9AC983CAABA2D3D53BD342AF7A81FBD4EBC013D2B3711BD7CBD04A58D3C7D33683DA63AD73D9F3DCBB8E83CDFBC38BD7CBD0A3C7CBCEDB6383C95248D3D523C413845BD813C7FBA983C71BBE2BBC1B562BC4ABCDA39A434EEBC173D07313B3D31B74939FC38EEBBCABC0CBBA8BCB9B922BD6639683C85BC9BBC223DB3B809BDC4BC9E3C66B9DBBB3B3B1FBD033DC53C553C48BD8FBDBEBB993CE33918BD7F3C2EBC61BD80BDFD3906BDBABD7ABD98BDD63C3CBD42BD64369EB9A3BD933D5BBD493DAF3C733B76BDC5B8673CD83CAB3D0E3CDFBB103829BDF83C2238C73A3F3D86BD05BD723C7E38BCB8CB3B4ABCCF3C1D3C333DE5BC8235BEBC1E3B623AEB3904BD98BDAA3DA3BD5D3C523D9A3D6B3D103C79BDFF3C12BD0F3D9BB5503D43B409BDA1BC4ABA063C57BDB7BD323D8DBD08BC93BC63BA713B313CA33B333D593DFDB94C3DF13B7A3285320534F6BB393B42AE0EBC7F3C783CAC39B73CBDBDAB3D2939903AAEBC5D3B3EBDB1BCBFBCC7B8EDBC17BB173BE43C9BAD42BD00BD723AB8BC5FB850B4F3BA61BD83323D3D1B3B583CF3BC503DE1378A39683D10BC713C6933FEBC4ABABD3C82BD3BBD7ABC2B3D37BDA83558BDB1BD7A31823C46BC12BDFB38ABBD143D92B531BDC33C0838E3BCF639FD382EBC6D33ACBCA03A4E3D31BC59BC"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}
