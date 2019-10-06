using Images: Gray, channelview, rawview

using FileIO
import Printf

function load_image_as_grayscale(imagename)
    oriimg = load(imagename)
    grayimg = Gray.(oriimg)
    rawview(channelview(oriimg)) / 255
end

function demo_scalespace(akaze)
    filenames = map(1:length(akaze.evolution_)) do n "akazetest-$(lpad(n,2,'0')).png" end
    for (fn, x) in zip(filenames, akaze.evolution_)
        save(fn, Gray.(x.Lt))
    end
    (j,k) = size(akaze.evolution_)
    run(`montage -filter point -geometry 512x512 -tile $(j)x$k $filenames akazetest.png`)
    imshow(load("akazetest.png"))
end

function plot_features(kpts, cmap="Blues", style=:dash)
    mycirc = hcat([[cos(t), sin(t)] for t in (2 * π * (0:53) / 53)]...)

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

function original_akaze_features(imagename, diffusivity; nsublevels=4, omax=4, dthreshold=0.001)
    dthreshold_str = Printf.@sprintf("%.15f", dthreshold)

    cmd = `/home/user/src/akaze/build/bin/akaze_features $imagename --diffusivity $(Int(opt.diffusivity)) --show_results 0 --dthreshold $dthreshold_str --descriptor 5 --nsublevels $nsublevels --omax $omax`
    display(cmd)
    run(cmd)

    data = open("keypoints.txt") do x
        el = eachline(x)
        iterate(el)
        iterate(el)
        ncols = 4
        reshape([parse(Float64, st) for r in el for st in split(r)[1:ncols]], ncols, :)
    end
    keypoints = map(eachcol(data)) do (x,y,size,angle)
        KeyPoint(pt=Point(x,y), size=size, angle=angle)
    end
    desc = open("keypoints.txt") do x
        el = eachline(x)
        rows = parse(Int, iterate(el)[1])
        cols = parse(Int, iterate(el)[1])
        reshape([parse(UInt8, st) for r in el for st in split(r)[end-rows+1:end]], rows, cols)
    end

    keypoints, desc
end

function dump_keypoints_text(kpts, desc)
    thedata = map(kpts, eachcol(desc)) do kp, dd
        @Printf.sprintf("%f %f %f %f %s", kp.pt.x, kp.pt.y, kp.size, kp.angle, join(dd, " "))
    end
    [[@Printf.sprintf("%d",x) for x in size(desc)]...,thedata...]
end

function print_keypoints_for_benchmark(kpts, desc)
    @Printf.printf("%d\n%d", size(desc)...)
    foreach(kpts, eachcol(desc)) do kp, dd
        sc = 1.0 / (kp.size)^2
        @Printf.printf("%f %f %f 0 %f %s\n", kp.pt.x, kp.pt.y, sc, sc, join(dd, " "))
    end
end
