using Plots
using Images, TestImages

include("akaze-config.jl")
include("akaze.jl")
include("fed.jl")
include("nonlinear-diffusion.jl")

img = rawview(channelview(testimage("cameraman")))/256
img_height, img_width = size(img)

opt = AKAZEOptions(omin=3, img_width=img_width, img_height=img_height)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)

plotly()
# pyplot()

for (n,x) in enumerate(akaze.evolution_)
    save("cam-$(lpad(n,2,'0')).png", Gray.(x.Lt))
end
