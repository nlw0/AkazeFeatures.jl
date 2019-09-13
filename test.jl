include("akaze-config.jl")
include("akaze.jl")
include("fed.jl")

opt = AKAZEOptions(omin=3, img_width=1024, img_height=780)
aaa = AKAZE(opt)
