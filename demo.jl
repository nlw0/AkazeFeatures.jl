import Printf
function demo_scalespace(akaze)
    filenames = map(1:length(akaze.evolution_)) do n "akazetest-$(lpad(n,2,'0')).png" end
    for (fn, x) in zip(filenames, akaze.evolution_)
        save(fn, Gray.(x.Lt))
    end
    (j,k) = size(akaze.evolution_)
    run(`montage -filter point -geometry 512x512 -tile $(j)x$k $filenames akazetest.png`)
    imshow(load("akazetest.png"))
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
    map(eachcol(data)) do (x,y,size,angle)
        KeyPoint(pt=Point(x,y), size=size, angle=angle)
    end
end
