using Test
using JLD2
using Statistics: median, quantile
using AkazeFeatures

refdata = load("original-akaze-pirate.jld2")

imagename = "../images/pirate.png"
img = AkazeFeatures.load_image_as_grayscale(imagename)

akaze = AKAZE(refdata["akazeoptions"])

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)
desc = Compute_Descriptors(akaze, kpts)

ord = sortperm(kpts, by=x->(x.pt.x, x.pt.y))

refkpts = refdata["keypoints"]
refdesc = refdata["descriptors"]
reford = sortperm(refkpts, by=x->(x.pt.x,x.pt.y))

ord = sortperm(collect(eachcol(desc)))
reford = sortperm(collect(eachcol(refdesc)))
rr = hcat(
    map(1:size(desc,2)) do n
        err = sum(Int.(refdesc .⊻ desc[:, n]), dims=1)[:]
        descerr = minimum(err)
        refn = findfirst(err .== descerr)
        [[kpts[n].pt.x, kpts[n].pt.y] - [refkpts[refn].pt.x, refkpts[refn].pt.y]; descerr]
    end...)

@testset "Pirate image stats" begin
    Nkp = length(refkpts)
    @test size(refdesc) == (61, Nkp)
    @test size(desc) == size(refdesc)
    @test maximum(abs.(rr[1:2,:])) < 6e-4
    @test median(abs.(rr[1:2,:])) < 2.5e-4
    @test maximum(abs.(rr[3,:])) < 130 # largest error is a keypoint right over the pirate's ring, high frequency
    @test count(x->x==0, rr[3,:]) / Nkp < 0.93
end

## Produce test data
# origpt, origdesc = AkazeFeatures.original_akaze_features(imagename, diffusivity=Int(opt.diffusivity), nsublevels=opt.nsublevels, omax=opt.omax, dthreshold=opt.dthreshold)
# save("original-akaze-pirate.jld2", Dict("keypoints"=>origpt, "descriptors"=>origdesc, "akazeoptions"=>opt))

## Investigate residues
# scatterlines(sort(abs.(rr[1,:])))
# scatterlines!(sort(abs.(rr[2,:])))
# scatterlines(sort(abs.(rr[3,:])))

# lines(rr[1,:])
# lines!(rr[2,:])
# lines!(rr[3,:])
# n = findfirst(rr[3,:] .== maximum(rr[3,:]))
# kpts[n].pt
# kpts[n]
# err = sum(Int.(refdesc .⊻ desc[:, n]), dims=1)[:]
# # lines(err)
# descerr = minimum(err)
# refn = findfirst(err .== descerr)
# refkpts[refn].pt
# refkpts[refn]
