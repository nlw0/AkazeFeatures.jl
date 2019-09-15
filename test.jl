using Plots
# pyplot()
# plotly()

using Images, TestImages
using FileIO

include("akaze-config.jl")
include("akaze.jl")
include("fed.jl")
include("nonlinear-diffusion.jl")

# imagename="cameraman"
imagename="pirate"
img = rawview(channelview(testimage(imagename))) / 256

img_height, img_width = size(img)

opt = AKAZEOptions(omin=3, img_width=img_width, img_height=img_height, diffusivity=WEICKERT)
akaze = AKAZE(opt)

# evolution1.Create_Nonlinear_Scale_Space(img1_32);
# evolution1.Feature_Detection(kpts1);
# evolution1.Compute_Descriptors(kpts1, desc1);

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)


mkcolorimage(img) = colorview(RGB, img, img, img)



# function demo_scalespace()
# filenames = map(1:length(akaze.evolution_)) do n "akazetest-$(lpad(n,2,'0')).png" end
# for (fn, x) in zip(filenames, akaze.evolution_)
    # save(fn, Gray.(x.Lt))
# end
# run(`montage -filter point -geometry 512x512 -tile 4x3 $filenames akazetest.png`)
# imshow(load("akazetest.png"))
# end


pp = vcat([[p.pt.x p.pt.y] for p in kpts]...)

save("pirate.png", testimage(imagename))

# plot(mkcolorimage(testimage(imagename))[end:-1:1,:])
# plot!(pp[:,1], 513-pp[:,2], m=3, l=0, color=:red, ratio=1)
