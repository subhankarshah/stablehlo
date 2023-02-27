module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3x9xf32>, tensor<12x1x3xf32>)
    %1 = call @expected() : () -> tensor<2x12x5xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [2], reverse = [0]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<2x3x9xf32>, tensor<12x1x3xf32>) -> tensor<2x12x5xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x12x5xf32>, tensor<2x12x5xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3x9xf32>, tensor<12x1x3xf32>) {
    %0 = stablehlo.constant dense<[[[3.04400682, -0.134076446, -2.55740047, -2.981350e+00, -0.906828224, -6.28386545, -0.458992362, 3.95152402, 1.34437251], [0.0944827869, -1.43317401, -3.83725524, -2.58242583, -4.04143763, -2.19238949, -5.4026041, 1.93202007, -0.701777756], [-2.47733283, 2.40808129, 2.46476054, -5.41994286, -2.891675, 5.00586939, 0.848040282, 3.33549261, -0.215870857]], [[-2.64300203, -4.74133492, -4.72930193, -5.26105642, 4.83494329, -0.951084971, 6.54493856, -1.95984232, 1.91990817], [-7.57104682, -0.500148356, 0.822786211, 2.20762134, 1.54983664, 1.16287351, 1.49611044, 0.0396747701, -2.67182302], [3.78783298, -6.108675, 5.55271769, 1.72712553, 0.353400081, 2.71722293, 2.98243332, 4.45062542, 4.88345146]]]> : tensor<2x3x9xf32>
    %1 = stablehlo.constant dense<[[[-1.47825599, -0.894643604, 1.18653953]], [[6.431360e+00, 4.53192329, -2.681674]], [[1.89863122, 2.70629096, -2.54629207]], [[-3.08360124, 1.87289381, 1.76727319]], [[-0.382819504, 4.32886314, 1.77006853]], [[1.20504749, -4.96435595, 0.746465981]], [[-3.16886234, -0.335680306, -0.630066276]], [[-3.96170807, -2.03329706, -1.07532191]], [[-2.56098819, -8.310980e-01, 4.91082191]], [[-4.34932232, 3.03853345, 0.281442106]], [[4.92129517, 2.89361811, -2.78656626]], [[-2.02973628, -0.578914702, -1.83752525]]]> : tensor<12x1x3xf32>
    return %0, %1 : tensor<2x3x9xf32>, tensor<12x1x3xf32>
  }
  func.func private @expected() -> tensor<2x12x5xf32> {
    %0 = stablehlo.constant dense<"0x156C52C047E692C067828140877B6B41F129564022B4264135931E40699C9AC1CDFE68C2604738C13B6E953FDCAEF540D982C4C0AAE902C2D462CCC032767CC1783482C1D70BAC40EA0B8D4037008A40EC67BEC1102D68C1EEB5CCC1EBA0A2C052A8B8C1282C8141CC4D1741CC7F3641B86C13416A69AB41733E6240A445D940E85C8741B976F640E5FF70418A613C413294544121D5E941DFC649417F01DE41FD781EC1EB5CB74156E0823EE8CCD0411B80B440E2998B414644CCC17C249AC1AAE31E42667C714111E33F409F408EC1F51BB33F70DCABC135CE32C1BAA30E410B2D2FC1ED629CC012ADFC3F50CDB840B1FF5D411465294100E726413BAFC94010982BC131964DC24A244FC2F471D0C1B48F03C2716E5E426A06F1C1CA8BA6C1C9F448C1FF50F2C07107B041EFC9FA40A18345403DD20C4278A62F41BBEF3D3F16411341B3E63C4123AD1041F349884006B1933F9CD140C1EE1A2BC1A2BDB2C0195045C08EBAF1C04BE9B5414DA7E33D093F82C0E826EDC075B86EC0E73CD541547F70C0C35000C1077332C110E3C9C0884749C1CA6BDC41AA36073EFDCB7241BBC8A4416480FE3E375302421CE6B1C14BA2FF3FB0640E4116E50642F48B02C2AF4EA0412F767D4059494FC0F3D538C164FFCC40BDA487C1CC1B54C176AD36C1"> : tensor<2x12x5xf32>
    return %0 : tensor<2x12x5xf32>
  }
}
