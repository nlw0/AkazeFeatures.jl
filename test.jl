using Images, TestImages
using FileIO

include("akaze-config.jl")
include("akaze.jl")
include("fed.jl")
include("nonlinear-diffusion.jl")

# imagename="cameraman"
# imagename="pirate"
# imagename="woman_blond"
img = rawview(channelview(testimage(imagename))) / 256

img_height, img_width = size(img)

opt = AKAZEOptions(omin=3, img_width=img_width, img_height=img_height, diffusivity=WEICKERT)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)

filenames = map(1:length(akaze.evolution_)) do n "akazetest-$(lpad(n,2,'0')).png" end

for (fn, x) in zip(filenames, akaze.evolution_)
    save(fn, Gray.(x.Lt))
end

run(`montage -filter point -geometry 512x512 -tile 4x3 $filenames akazetest.png`)
