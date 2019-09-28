using Images
using FileIO

using Plots
pyplot()

include("akaze-config.jl")
include("akaze.jl")
include("fed.jl")
include("demo.jl")
include("nonlinear-diffusion.jl")

# imagename = "images/square.png"
# imagename = "images/concave.png"
# imagename = "images/wiggly.png"
# imagename = "images/wiggly-blur.png"
# imagename = "images/pirate.png"
# imagename = "images/cameraman.png"
imagename = "images/zezao.png"
# imagename = "images/polly.png"
# imagename = "images/polly-small.png"

oriimg = load(imagename)

img = rawview(channelview(oriimg)) / 255

img_height, img_width = size(img)

opt = AKAZEOptions(
    omin = 5,
    # nsublevels=5,
    img_width = img_width,
    img_height = img_height,
    # diffusivity = PM_G1,
    # diffusivity = PM_G2,
    # diffusivity = WEICKERT,
    diffusivity = CHARBONNIER,
    # dthreshold = 4e-5,
    dthreshold = 16e-5,
)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)

pp = vcat([[p.pt.x p.pt.y] for p in kpts]...)

origpt = original_akaze_features(imagename, Int(opt.diffusivity), 0.002)

plot(size=(800,800))
plot!([-0.5, size(img)[2]-0.5], [-0.5,size(img)[1]-0.5], RGB.(oriimg), yflip = false)
plot!(pp[:, 1], pp[:, 2], m = 5, l = 0, color = :red, ratio = 1, shape = :x, label = "this")
plot!(origpt[1, :], origpt[2, :], m = 5, l = 0, color = :yellow, ratio = 1, shape = :+, label = "original")
aa = plot!(xticks = :native, yticks = :native)
