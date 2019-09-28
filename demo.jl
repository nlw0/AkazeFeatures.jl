function demo_scalespace(akaze)
    filenames = map(1:length(akaze.evolution_)) do n "akazetest-$(lpad(n,2,'0')).png" end
    for (fn, x) in zip(filenames, akaze.evolution_)
        save(fn, Gray.(x.Lt))
    end
    (j,k) = size(akaze.evolution_)
    run(`montage -filter point -geometry 512x512 -tile $(j)x$k $filenames akazetest.png`)
    imshow(load("akazetest.png"))
end

function original_akaze_features(imagename, diffusivity, dthreshold=0.001)
    run(`/home/user/src/akaze/build/bin/akaze_features $imagename --diffusivity $(Int(opt.diffusivity)) --show_results 0 --dthreshold $dthreshold`)

    open("keypoints.txt") do x
        el = eachline(x)
        iterate(el)
        iterate(el)
        reshape([parse(Float64, st) for r in el for st in split(r)[1:2]], 2, :)
    end
end
