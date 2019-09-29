using Images
using FileIO

using Plots
pyplot()

using YAML


include("akaze-config.jl")
include("akaze-descriptor.jl")
include("akaze.jl")
include("fed.jl")
include("demo.jl")
include("nonlinear-diffusion.jl")

imagename = "images/square.png"
# imagename = "images/concave.png"
# imagename = "images/wiggly.png"
# imagename = "images/wiggly-blur.png"
# imagename = "images/pirate.png"
# imagename = "images/cameraman.png"
# imagename = "images/zezao.png"
# imagename = "images/polly.png"
# imagename = "images/polly-small.png"

oriimg = load(imagename)

img = rawview(channelview(oriimg)) / 255

img_height, img_width = size(img)

opt = AKAZEOptions(
    omin = 3,
    omax = 2,
    nsublevels=3,
    img_width = img_width,
    img_height = img_height,
    # diffusivity = PM_G1,
    # diffusivity = PM_G2,
    # diffusivity = WEICKERT,
    diffusivity = CHARBONNIER,
    dthreshold = 1e-5,
    # dthreshold = 2e-5,
    # dthreshold = 1.5e-5,
    # dthreshold = 4e-5,
    # dthreshold = 16e-5,
    # dthreshold = 32e-5,
    # dthreshold = 100.0,
)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)
Compute_Main_Orientation.([akaze], kpts)
println("Extracted $(length(kpts)) points")

origpt = original_akaze_features(imagename, Int(opt.diffusivity); nsublevels=opt.nsublevels, omax=opt.omax, dthreshold=0.001)
println("Original: $(length(origpt)) points")

mycirc = hcat([[cos(t), sin(t)] for t in (2 * π * (0:53) / 53)]...)

function plot_features(kpts, cmap="Blues")
    data = map(kpts) do kp
        xy = [kp.pt.x; kp.pt.y]
        xx = xy .+ kp.size * [0 cos(kp.angle); 0 sin(kp.angle)]
        circ = xy .+ kp.size * mycirc
        [circ[1,:], circ[2,:], xx[1,:], xx[2,:], xx[1,1:1], xx[2,1:1]]
    end
    cx = [d[1] for d in data]
    cy = [d[2] for d in data]
    rx = [d[3] for d in data]
    ry = [d[4] for d in data]
    px = [d[5] for d in data]
    py = [d[6] for d in data]

    plot!(cx, cy, l=2, color = Colors.colormap(cmap)[90], label = "", style=:dash, alpha=0.85)
    plot!(rx,ry, l=2, color = Colors.colormap(cmap)[75], label = "", alpha=0.85)
    scatter!(px,py, color = Colors.colormap(cmap)[50], shape = :o, m=3, label = "", msw=0)
end

plot(size=(800,800))
plot!([-0.5, size(img)[2]-0.5], [-0.5,size(img)[1]-0.5], RGB.(oriimg), yflip = false)
plot_features(origpt, "Blues");
plot_features(kpts[1:min(length(kpts),500)], "Oranges");
plot!(xticks = :native, yticks = :native)


evs = map(0:5) do ee
    open("/home/user/src/AKAZE.jl/evLdet-000$ee.ext") do ff
        datas = read(ff, String)[40:end]
        data = YAML.load(datas)
        (sz,) = size(data["data"])
        reshape(data["data"], round(Int, sqrt(sz)), :)'
    end
end

# heatmap(evs[1][119:140,119:140], subplot=1, layout=2)
# heatmap!(akaze.evolution_[1].Ldet[119:140,119:140], subplot=2, layout=2)

# aa = evs[1][119:140,119:140]
# oo = akaze.evolution_[1].Ldet[119:140,119:140]
a=374
o=391
q=3
aa = evs[q][a:o,a:o]
oo = akaze.evolution_[q].Ldet[a:o,a:o]
heatmap(aa, subplot=1, layout=2)
heatmap!(oo, subplot=2, layout=2)

# heatmap((aa .+ 1e-20) ./ (oo .+ 1e-18))
