module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.minimum %0#0, %0#1 : tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf16>, tensor<20x20xf16>) {
    %0 = stablehlo.constant dense<"0x06BFD83DCF40D23D93BFEB4039C362C1D3BCF6C0DEC2B5BB54C70CBD0FC4C540BBC0753E95C0DB3BAEC0F93E4F3D1040CB42D9AC734111C380B094368BBCF7C099C0D446DD3888417EC04A42164463C7253E4C47ED3D304144BA363B16C26BBF2A42E0BC753932BC06C17D4080C71CB7133F5340B5C24F3BF4B41E4345446EB94839CC3DD9BE80B8DC42164677BB23422740723C58C2A43A2EC20CC4A9448BBCA1C4C9C2263EECBEA73F78399AB8E14046B294BA8EC1263F253A9E44A434CFC2C73C4340E641262A4D40893CCF469441EAB228445AC331B99440933593B94C3D78410D3C0648A5B3DB3C223609C3EF42A7BAA83D43B9D4BD9BC01E2E63412C3CAF42FFB9EA3DA342B8340F2AC9BCD7B7964090BF362AF0B85BB5DD3662444BBF9B451C40E4C30D41D33FDB43DDC26E43474138C42446A0C0F2B99F411948A7BDE0C4EB3C31409DC3DDC53D3A93C4B3B3A344813C2DC11CC4D43FFDC3ABBE3D370B3B64B5CF408BA497C1873610461348CC43483F2CC1E9BC58C12EBE914679C5B6C199BB17C4933CC4BD5FC6CC445F4275C2C9C6563C423C3143B030BEBD21463FB83A445CB560C739BBD6C0023DBD3CBC3DD8C00044ED4039C17F4308BC84374643214323C1E9C4B93F0EA7142ECD4468408CC145C0493D0541DD41CF4341B597BAAABFAC433EC2E942724342BE893EFAC5C5C115C34C409FB8E93F0A452541A93625C07839012CA5480AC5FA3D0830A9BF9E40D031EFBFDAB793BEEE29BF311744A1C16F346EC0A245224011C084426A30FABF02BC76BFAD42AB3C8E3F52C6613D302EBCC354C1DEC32D3DA4C3E93FD53C7745929A0DC82ABC1BBD95C45238CA406CC45D3F1FA4603D29C5DB3CD0C0553F06BF01C1A143D1BCBE3459363535B9392F419843A1B1D7BDD7C0F940E2407FC236C2E5C6E1BDAAC3C3BEFFBAC23D9845B1C473C095BCD0405D4089BBC9402DB3F9C379BB6FBB733A7EBF67B545C62EBD1F4253410FB7E0C0C632183F1C43FFC2C5410F41A42C81B7C2B6EF3681C118362E4314C491B973BB8F40DAC3FAAFD33D483DEDC52ABC17C03AC17FC22AC627BBE8427A44D84582438734D6406FC4E13F1C445A3CBCC01840753BD24199C2"> : tensor<20x20xf16>
    %1 = stablehlo.constant dense<"0x0CAE0EBCAE43583D2344F7BAF8BCD13CB2C0F1397044BDC053C15E4483C0CC390BC4ED3AE934DDC349B4B6BD134459C28BC38DBD9041C642A4352D401DBCD8405ABE0A4115412F439F3C04C077C1E63CF334BA40FD44BF40BB3C46C2D4BD19456A3C313CE2477D3EEB3CD2C0803818C37EC7EABEDD476B3D2AC53E46B73D163F3F40403CD1BE37B4713CAC45E3BAE841B7BF59406FC1DCBDD5B51741053FEA3F622FAFC5F9BCE3B9A34082C5A1C1D144EEC02B405443173A7640D13920C62BC0E4B98AB8403C77AB4834253EDFB403C060BE33C46842E844EDBF12C10240D22593430B3F593AC2B0263E2FB499BFA0BC60B918485D3B26C8A84066BC4644D33C82B6BEC26AC299C28B3B03C5CEB5B6B64E45C81C39C422C07BB990BF14BC6AC05E437136CDC5943F19BCD24066B4DABDC04244BFAEB564BE014870BD63BF00C57D402338B642A7BAD944B43F4EB9DA461D3EB040CB448840913F5E419E437EB924C2D83A63B8DD45F040A543FEC41DC1963E843997C0E0C228BE2041B5C70BB49AC4ABBDFAC0BA38BAC3F733544398420C3C6DBC19C110C019BA8340D545714538C0803D9945293CC13ED7B9B64271BE0D2966C6BEBFBAC04EB403435B3B4CBC4BBE3431DAC129BB1937ED33F33F88443DBEC6BE18C09DAE1E460DBF2F3057C5B9342DB975BFD1C1FEC15A46B4BFA9BCFCC0453E7CC57EBE9ABFECBCA43A6A427BC2162EFBC0843F7EAF3B3F10C04F40CEBC774623C45E41B646DF427846A444C1BC833E7F42A0C110B9A3354A3E4B39803AC3BD1D422E3DC240E84148441E3F193E83456A410140E9C006C6EDC392365A39AD3BD4421A418C41A2B9C5C1FDC1DE38A3B58F40D541CAB82D3DC4C45240D5B60B450F375145ED3EF1BF6543B0C1AC4012BB2039793D6CC1F63965C444C711C0F52D28427E3BDCAF1EBD05C001BEA1C2B2B8CE429C417444303EAD453A42E3C1B642FA40CD3E6CBCD8C5D4467C4390BEE042613AE83A83BD6CBA44C6F34001B7B7C2A9AFFC2EF430953CE2BA5639DC389B448FB895C6863ABDB46F3BE9C2B8BCD3BE533ED83E6840A63C2C3703463F2EDB2EB2BC3DBDD3480BBB3ABC23C5D9BEFDC082BB2A44ACC091C265C42546"> : tensor<20x20xf16>
    return %0, %1 : tensor<20x20xf16>, tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x06BF0EBCCF40583D93BFF7BA39C362C1B2C0F6C0DEC2BDC054C70CBD0FC4CC390BC4ED3A95C0DDC3AEC0B6BD4F3D59C28BC38DBD734111C380B094368BBCF7C099C00A41DD3888417EC004C077C163C7F334BA40ED3DBF4044BA46C216C26BBF6A3CE0BC753932BC06C1D2C080C718C37EC7EABEB5C24F3B2AC51E43B73D6EB94839403CD9BE80B8713CAC4577BBE841B7BF723C58C2DCBD2EC20CC4053F8BBCA1C4AFC5F9BCECBEA73F82C5A1C1E140EEC094BA8EC1173A253AD13920C6CFC2E4B98AB8403C77AB4834893CDFB403C060BE33C45AC331B9EDBF12C193B9D22578410D3C593AA5B3DB3C2FB409C3A0BCA7BAA83D43B926C89BC066BC63412C3C82B6BEC26AC299C2B83403C5C9BCD7B7964090BF39C422C07BB990BF14BC6AC05E437136CDC5943F19BCD240DDC2DABD474138C4AEB5A0C0F2B970BD63BF00C5E0C4233831409DC3DDC53D3A93C4B3B31D3E813C2DC11CC4913FFDC3ABBE7EB924C264B563B88BA497C18736FEC41DC1963E84392CC1E0C258C12EBEB5C779C59AC4ABBD17C4BA38BAC35FC654435F4275C2C9C619C110C019BAB030BEBD714538C0803D5CB560C739BBD6C0023D71BE0D2966C6BEBFBAC039C1034308BC4CBC4BBE3431DAC1E9C419370EA7142E88443DBE8CC145C09DAE05410DBF2F3057C597BAAABF75BF3EC2FEC17243B4BFA9BCFAC5C5C17CC57EBE9ABFECBCA43A25417BC225C0FBC0012C7EAF0AC510C00830A9BF9E4023C4EFBFDAB793BEEE29BF31C1BCA1C16F34A0C110B9A33511C04B396A30FABF02BC76BFC240AB3C8E3F52C6613D302EBCC354C1DEC306C6EDC392365A39AD3B929A0DC82ABC1BBD95C4FDC1DE386CC45D3F1FA4CAB829C5C4C4D0C0D5B606BF01C1A143D1BCF1BF5936B0C1B93912BB2039A1B16CC1D7C065C444C77FC236C2E5C6E1BDAAC3C3BE05C001BEA1C2B1C473C095BCD040303E89BBC940E3C1F9C379BB6FBB6CBCD8C567B545C690BE1F42613A0FB7E0C06CBA44C6F340FFC2B7C2A9AFA42C81B7C2B6E2BA81C118362E4314C495C673BBBDB4DAC3E9C2B8BCD3BEEDC52ABC17C03AC17FC22AC627BBDB2EB2BC3DBD82430BBB3ABC23C5D9BEFDC082BBBCC0ACC091C265C499C2"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}
