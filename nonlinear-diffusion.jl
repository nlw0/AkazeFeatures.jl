using ImageFiltering: Kernel, imfilter
using TestImages: testimage
using ImageView: imshow
using Images: rawview, channelview


img = rawview(channelview(testimage("pirate"))) / 255

Lt = img
Lsmooth = imfilter(img, Kernel.gaussian(1.0))

fx,fy = Kernel.ando3()

Lx = imfilter(Lsmooth, fx)
Ly = imfilter(Lsmooth, fy)
Lxx = imfilter(Lx, fx)
Lxy = imfilter(Lx, fy)
Lyy = imfilter(Ly, fy)


function nld_step_scalar(Ld, c, stepsize)
    Lstep = zeros(size(Ld))

    dx = 0.5*stepsize*(c[1:end, 1:end-1] + c[1:end, 2:end]).*(Ld[1:end, 2:end] - Ld[1:end, 1:end-1])
    dy = 0.5*stepsize*(c[1:end-1, 1:end] + c[2:end, 1:end]).*(Ld[2:end, 1:end] - Ld[1:end-1, 1:end])

    Ld[1:end,1:end-1] .+= dx
    Ld[1:end,2:end] .-= dx
    Ld[1:end-1,1:end] .+= dy
    Ld[2:end,1:end] .-= dy
end

pm_g1_diffusivity(Lx, Ly, k) = calculate_diffusivity(Lx, Ly, k) do dL
    -dL
end

pm_g2_diffusivity(Lx, Ly, k) = calculate_diffusivity(Lx, Ly, k) do dL
    1.0 / (1.0 + dL)
end

weickert_diffusivity(Lx, Ly, k) = calculate_diffusivity(Lx, Ly, k) do dL
    1.0 - exp(-3.315/(dL*dL*dL*dL))
end

charbonnier_diffusivity(Lx, Ly, k) = calculate_diffusivity(Lx, Ly, k) do dL
    1.0 / sqrt(1.0 + dL)
end

function calculate_diffusivity(func, Lx, Ly, k)
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

function compute_k_percentile(Lx, Ly; nbins = 300, perc=0.2)
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


# imshow(Lsmooth)
# aa = pm_g1_diffusivity(Lx, Ly, 0.01)
# imshow(aa)
# aa = pm_g2_diffusivity(Lx, Ly, 0.01)
# imshow(aa)
# aa = weickert_diffusivity(Lx, Ly, 0.01)
# imshow(aa)
# aa = charbonnier_diffusivity(Lx, Ly, 0.01)
# imshow(aa)


# compute the k contrast

# gthresh = compute_k_percentile((@view Lx[2:end-1,2:end-1]), (@view Ly[2:end-1,2:end-1]); perc=0.95)

# imshow(sqrt.(Lx.^2+Ly.^2))
# imshow(sqrt.(Lx.^2+Ly.^2) .< gthresh)

# Lflow = pm_g2_diffusivity(Lx, Ly, 0.01)
Lflow = charbonnier_diffusivity(Lx, Ly, 0.01)

imshow(copy(Lt))
for _ in 1:3
    for _ in 1:400
        nld_step_scalar(Lt, Lflow, 0.05)
    end
    imshow(copy(Lt))
end
