using ImageFiltering: Kernel, imfilter
using ImageView: imshow
using Images: rawview, channelview



function nld_step_scalar(Ld, c, stepsize)
    dx = 0.5*stepsize*(c[1:end, 1:end-1] + c[1:end, 2:end]).*(Ld[1:end, 2:end] - Ld[1:end, 1:end-1])
    dy = 0.5*stepsize*(c[1:end-1, 1:end] + c[2:end, 1:end]).*(Ld[2:end, 1:end] - Ld[1:end-1, 1:end])

    Ld[1:end,1:end-1] .+= dx
    Ld[1:end,2:end] .-= dx
    Ld[1:end-1,1:end] .+= dy
    Ld[2:end,1:end] .-= dy
end

pm_g1_diffusivity(Lx, Ly, k) = calculate_diffusivity′(Lx, Ly, k) do dL
    -dL
end

pm_g2_diffusivity(Lx, Ly, k) = calculate_diffusivity′(Lx, Ly, k) do dL
    1.0 / (1.0 + dL)
end

weickert_diffusivity(Lx, Ly, k) = calculate_diffusivity′(Lx, Ly, k) do dL
    1.0 - exp(-3.315/(dL*dL*dL*dL))
end

charbonnier_diffusivity(Lx, Ly, k) = calculate_diffusivity′(Lx, Ly, k) do dL
    1.0 / sqrt(1.0 + dL)
end

function calculate_diffusivity′(func, Lx, Ly, k)
    dst = zeros(size(Lx))
    invk2 = 1.0 / (k * k)
    nrow, ncol = size(Lx)
    @inbounds begin
        @simd for j in 1:nrow
            @simd for k in 1:ncol
                lx,ly = Lx[j,k],Ly[j,k]
                dL = (lx*lx+ly*ly) * invk2
                dst[j,k] = func(dL)
            end
        end
    end
    dst
end

function compute_k_percentile(img, perc; gscale=1.0, nbins = 300)
    Lsmooth = imfilter(img, Kernel.gaussian(gscale))
    fx,fy = Kernel.ando3()
    fx .*= 1.0940889041865054
    fy .*= 1.0940889041865054

    Lx = @view imfilter(Lsmooth, fx)[2:end-1,2:end-1]
    Ly = @view imfilter(Lsmooth, fy)[2:end-1,2:end-1]
    compute_k_percentile′(Lx, Ly, perc, nbins)
end

function compute_k_percentile′(Lx, Ly, perc, nbins = 300)
    hist = zeros(Int32, nbins)

    hmax = sqrt(maximum(y->sum(x->x^2, y), zip(Lx[:], Ly[:])))

    for (lx, ly) in zip(Lx[:], Ly[:])
        if (lx, ly) != (0.0, 0.0)
            modg = sqrt(lx^2 + ly^2)
            nbin = ceil(Int, nbins*(modg/hmax))  # limiting to nbins should not be necessary
            hist[nbin] += 1
        end
    end

    nthreshold = floor(Int, sum(hist) * perc)

    k = findfirst(hx -> hx > nthreshold, cumsum(hist))

    return if k == nothing
        0.03
    else
        (hmax * (k-1)) / nbins
    end
end

function demo_diffusivity_functions()
    imshow(Lsmooth)
    aa = pm_g1_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = pm_g2_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = weickert_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = charbonnier_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
end

function demo_k_percentile()
    gthresh = compute_k_percentile(img, 0.95)
    imshow(sqrt.(Lx.^2+Ly.^2))
    imshow(sqrt.(Lx.^2+Ly.^2) .< gthresh)
end

function demo_nld()
    Lflow = charbonnier_diffusivity(Lx, Ly, 0.01)

    imshow(copy(Lt))
    for _ in 1:3
        for _ in 1:100
            nld_step_scalar(Lt, Lflow, 0.05)
        end
        imshow(copy(Lt))
    end
end

# demo_diffusivity_functions()
# demo_k_percentile()
# demo_nld()
