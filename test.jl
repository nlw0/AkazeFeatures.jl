using Images
using FileIO

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
    omin = 3,
    omax = 4,
    nsublevels=2,
    img_width = img_width,
    img_height = img_height,
    # diffusivity = PM_G1,
    # diffusivity = PM_G2,
    # diffusivity = WEICKERT,
    diffusivity = CHARBONNIER,
    # dthreshold = 16e-3,
    # dthreshold = 8e-3,
    # dthreshold = 4e-3,
    dthreshold = 2e-3,
)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)
# for i in 1:length(kpts)
#     Compute_Main_Orientation(akaze, Ref(kpts,i)) ## actually done within Compute_Descriptors already
# end
println("Extracted $(length(kpts)) points")
desc = Compute_Descriptors(akaze, kpts)

origpt, origdesc = AkazeFeatures.original_akaze_features(imagename, diffusivity=Int(opt.diffusivity), nsublevels=opt.nsublevels, omax=opt.omax, dthreshold=opt.dthreshold)
println("Original: $(length(origpt)) points")


# plot(size=(800,800))
# plot!([-0.5, size(img)[2]-0.5], [-0.5,size(img)[1]-0.5], RGB.(oriimg), yflip = true)
# plot_features(origpt, "Blues", :solid);
# plot_features(kpts[1:length(kpts)], "Oranges");
# plot!(xticks = :native, yticks = :native)
# plot!()
# plot!(yflip = false)
