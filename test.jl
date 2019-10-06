using Images
using FileIO

using Plots
pyplot()

using YAML


include("akaze-config.jl")
include("akaze-descriptor.jl")
include("akaze.jl")
include("fed.jl")
include("nonlinear-diffusion.jl")
include("utils.jl")

# imagename = "images/dirac.png"
# imagename = "images/square.png"
# imagename = "images/concave.png"
# imagename = "images/wiggly.png"
# imagename = "images/wiggly-blur.png"
# imagename = "images/pirate.png"
imagename = "../vggaffine/boat/img1.pgm"
# imagename = "images/cameraman.png"
# imagename = "images/zezao.png"
# imagename = "images/polly.png"
# imagename = "images/polly-small.png"

oriimg = load(imagename)

img = rawview(channelview(oriimg)) / 255

img_height, img_width = size(img)

opt = AKAZEOptions(
    omin = 3,
    omax = 4,
    nsublevels=5,
    img_width = img_width,
    img_height = img_height,
    # diffusivity = PM_G1,
    # diffusivity = PM_G2,
    # diffusivity = WEICKERT,
    diffusivity = CHARBONNIER,
    dthreshold = 16e-3,
    # dthreshold = 4e-3,
    # dthreshold = 2e-3,
    # dthreshold = 1e-8,
    # dthreshold = 1e-5,
    # dthreshold = 2e-5,
    # dthreshold = 1.5e-5,
    # dthreshold = 4e-5,
    # dthreshold = 16e-5,
    # dthreshold = 32e-5,
    # dthreshold = 200.0,
)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)
Compute_Main_Orientation.([akaze], kpts)
println("Extracted $(length(kpts)) points")
desc = Compute_Descriptors(akaze, kpts)

origpt = original_akaze_features(imagename, Int(opt.diffusivity); nsublevels=opt.nsublevels, omax=opt.omax, dthreshold=opt.dthreshold)
println("Original: $(length(origpt)) points")

mycirc = hcat([[cos(t), sin(t)] for t in (2 * π * (0:53) / 53)]...)

function plot_features(kpts, cmap="Blues", style=:dash)
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

    plot!(cx, cy, l=2, color = Colors.colormap(cmap)[90], label = "", style=style, alpha=0.85)
    plot!(rx,ry, l=2, color = Colors.colormap(cmap)[75], label = "", alpha=0.85)
    scatter!(px,py, color = Colors.colormap(cmap)[50], shape = :o, m=5, label = "", msw=0)
end

plot(size=(800,800))
plot!([-0.5, size(img)[2]-0.5], [-0.5,size(img)[1]-0.5], RGB.(oriimg), yflip = false)
plot_features(origpt, "Blues", :solid);
# plot_features(kpts[1:min(length(kpts),500)], "Oranges");
plot_features(kpts[1:length(kpts)], "Oranges");
plot!(xticks = :native, yticks = :native)

print_keypoints(kpts, desc)

# evs = map(0:14) do ee
#     open("/home/user/src/AKAZE.jl/evLdet-$(lpad(ee,4,'0')).ext") do ff
#         datas = read(ff, String)[40:end]
#         data = YAML.load(datas)
#         (sz,) = size(data["data"])
#         reshape(data["data"], round(Int, sqrt(sz)), :)'
#     end
# end

# # heatmap(evs[1][119:140,119:140], subplot=1, layout=2)
# # heatmap!(akaze.evolution_[1].Ldet[119:140,119:140], subplot=2, layout=2)

# # aa = evs[1][119:140,119:140]
# # oo = akaze.evolution_[1].Ldet[119:140,119:140]
# mev = [ev.Lx for ev in akaze.evolution_]

# a=374
# o=391
# q=5
# aa = evs[q][a:o,a:o]
# oo = akaze.evolution_[q].Ldet[a:o,a:o]
# heatmap(aa, subplot=1, layout=2, xticks=:native, yticks=:native, colorbar=false, ratio=1)
# heatmap!(oo, subplot=2, layout=2, xticks=:native, yticks=:native, colorbar=false, ratio=1)

# plot(size=(1500,900), layout=(3,5))
# for q in 1:15
#     # a=374 ÷ 2^akaze.evolution_[q].octave
#     # o=391 ÷ 2^akaze.evolution_[q].octave
#     a=330 ÷ 2^akaze.evolution_[q].octave
#     o=410 ÷ 2^akaze.evolution_[q].octave
#     aa = evs[q][a:o,a:o]
#     oo = akaze.evolution_[q].Ldet[a:o,a:o]
#     # heatmap!(aa, subplot=q, layout=(5,3), xticks=:native, yticks=:native, colorbar=false, ratio=1)
#     # heatmap!(aa, subplot=q, layout=(3,5), colorbar=false, ratio=1)
#     heatmap!(oo, subplot=q, layout=(3,5), colorbar=false, ratio=1)
# end
# plot!()

# heatmap((aa .+ 1e-20) ./ (oo .+ 1e-18))

# o = hcat([[minimum(ev), maximum(ev)] for ev in evs]...)
# m = hcat([[minimum(ev), maximum(ev)] for ev in mev]...)

# aa=sort(abs.(evs[4][:]))
# oo=sort(abs.(mev[4][:]))
# plot(aa/aa[end], 1:length(aa))
# plot!(oo/oo[end], 1:length(oo))


# map(akaze.evolution_) do ev
#     ratio = 2.0 ^ ev.octave
#     round(Int, ev.esigma * opt.derivative_factor/ratio)
#     # ev.esigma * opt.derivative_factor/ratio
# end



# evs = map(0:14) do ee
#     open("/home/user/src/AKAZE.jl/evLdet-$(lpad(ee,4,'0')).ext") do ff
#         datas = read(ff, String)[40:end]
#         data = YAML.load(datas)
#         (sz,) = size(data["data"])
#         reshape(data["data"], round(Int, sqrt(sz)), :)'
#     end
# end
# plot(size=(1500,900), layout=(3,5))
# for q in 1:15
#     a=330 ÷ 2^akaze.evolution_[q].octave
#     o=410 ÷ 2^akaze.evolution_[q].octave
#     aa = evs[q][a:o,a:o]
#     heatmap!(aa, subplot=q, layout=(3,5), ratio=1)
# end
# plot!()
# savefig("akaze-Ldet-ori.png")

# mev = [ev.Ldet for ev in akaze.evolution_]

# plot(size=(1500,900), layout=(3,5))
# for q in 1:15
#     a=330 ÷ 2^akaze.evolution_[q].octave
#     o=410 ÷ 2^akaze.evolution_[q].octave
#     oo = mev[q][a:o,a:o]
#     heatmap!(oo, subplot=q, layout=(3,5), ratio=1)
# end
# plot!()
# savefig("akaze-Ldet-new.png")

# @show [maximum(abs.(ev)) for ev in evs]'
# @show [maximum(abs.(ev)) for ev in mev]'

# [maximum(abs.(ev)) for ev in evs] ./ [maximum(abs.(ev)) for ev in mev][:]
