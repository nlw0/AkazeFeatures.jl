using Images
using FileIO
using JLD2

using GLMakie; using GLMakie: Axis
# using Plots
# pyplot()

using Revise
using AkazeFeatures

# imagename = "images/dirac.png"
# imagename = "images/square.png"
# imagename = "images/concave.png"
# imagename = "images/wiggly.png"
# imagename = "images/wiggly-blur.png"
imagename = "images/pirate.png"
# imagename = "/home/user/src/data/mars/NLB_629553760EDR_F0780612CCAM05613M_.jpeg"
# imagename = "../vggaffine/boat/img1.pgm"
# imagename = "images/cameraman.png"
# imagename = "images/zezao.png"
# imagename = "images/polly.png"
# imagename = "images/polly-small.png"

oriimg = load(imagename)
# img = rawview(channelview(oriimg)) / 255

img = AkazeFeatures.load_image_as_grayscale(imagename)

img_height, img_width = size(img)

opt = AKAZEOptions(
    omin = 0,
    # omin = 3,
    omax = 5,
    nsublevels=3,
    img_width = img_width,
    img_height = img_height,
    # descriptor = AkazeFeatures.MLDB,
    descriptor = AkazeFeatures.MLDB_UPRIGHT,
    # diffusivity = PM_G1,
    # diffusivity = PM_G2,
    # diffusivity = WEICKERT,
    diffusivity = AkazeFeatures.CHARBONNIER,
    # dthreshold = 16e-3,
    # dthreshold = 8e-3,
    # dthreshold = 4e-3,
    # dthreshold = 2e-3,
    dthreshold = 1.75e-3,
    # dthreshold = 1.7e-3,
    # dthreshold = 2e-4,
    # dthreshold = 1e-5,
    # harris_coefficient=0.06
)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)
println("Extracted $(length(kpts)) points")
desc = Compute_Descriptors(akaze, kpts)

origpt, origdesc = AkazeFeatures.original_akaze_features(imagename, diffusivity=Int(opt.diffusivity), nsublevels=opt.nsublevels, omax=opt.omax, dthreshold=opt.dthreshold, descriptor=opt.descriptor)

println("Original: $(length(origpt)) points")


# plot(size=(800,800))
# plot!([-0.5, size(img)[2]-0.5], [-0.5,size(img)[1]-0.5], RGB.(oriimg), yflip = true)
# plot_features(origpt, "Blues", :solid);
# plot_features(kpts[1:length(kpts)], "Oranges");
# plot!(xticks = :native, yticks = :native)
# plot!()
# plot!(yflip = false)

ff=Figure()
ax = pltimg(ff[1,1], img)
scatter!(ax, [x.pt.x for x in origpt], [x.pt.y for x in origpt], marker=:diamond, markersize=30)
scatter!(ax, [x.pt.x for x in kpts], [x.pt.y for x in kpts], markersize=20, color=:red)

myord = sortperm(kpts, by=x->(x.pt.x,x.pt.y))
origord = sortperm(origpt, by=x->(x.pt.x,x.pt.y))

kpts[myord][100]
origpt[origord][100]

desc[:,myord[100]]'
origdesc[:,origord[100]]'

pltimg((desc[:,myord] .!= origdesc[:,origord]))

# desc[:,myord] == origdesc[:,origord]

# measure_error

# save("original-akaze-pirate.jld2", Dict("keypoints"=>origpt, "descriptors"=>origdesc, "akazeoptions"=>opt))

# load("original-akaze-pirate.jld2")


origpt
origdesc
