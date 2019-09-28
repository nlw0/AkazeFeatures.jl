using Images, TestImages
using FileIO

using Plots
pyplot()

include("akaze-config.jl")
include("akaze.jl")
include("fed.jl")
include("nonlinear-diffusion.jl")

# imagename="cameraman"
# imagename="pirate"
# img = rawview(channelview(testimage(imagename))) / 255
imagename = "images/square.png"
# imagename = "images/concave.png"
# imagename = "images/wiggly.png"
# imagename = "images/wiggly-blur.png"
oriimg = load(imagename)
img = rawview(channelview(oriimg)) / 255

img_height, img_width = size(img)

opt = AKAZEOptions(
    omin = 3,
    img_width = img_width,
    img_height = img_height,
    # diffusivity = WEICKERT,
    diffusivity = PM_G2,
    dthreshold = 4e-5,
    # dthreshold = 0#1e-5,
)
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

# [k for k in kpts if k.pt.x>200 && k.pt.x<300 && k.pt.y<300]
[k for k in kpts if k.pt.x<200 && k.pt.y<200]

run(`/home/user/src/akaze/build/bin/akaze_features $imagename --diffusivity $(Int(opt.diffusivity)) --show_results 0`)

orig = open("keypoints.txt") do x
    el = eachline(x)
    iterate(el)
    iterate(el)
    reshape([parse(Float64, st) for r in el for st in split(r)[1:2]], 2, :)
end

# save("pirate.png", testimage(imagename))

plot(size=(800,800))
# plot(RGB.(testimage(imagename)), xticks=:native, yticks=:native, yflip=false)
plot([-0.5, 511.5], [-0.5,511.5], RGB.(oriimg), yflip = false)
plot!(pp[:, 1], pp[:, 2], m = 5, l = 0, color = :red, ratio = 1, shape = :x, label = "this")
plot!(orig[1, :], orig[2, :], m = 5, l = 0, color = :yellow, ratio = 1, shape = :+, label = "original")
aa = plot!(xticks = :native, yticks = :native)
