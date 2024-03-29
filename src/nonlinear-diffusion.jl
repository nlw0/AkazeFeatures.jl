using ImageFiltering: Kernel, imfilter, kernelfactors, centered


# function nld_step_scalar(Ld, c, stepsize)
#     dx = 0.5 * stepsize * (c[1:end, 1:end-1] + c[1:end, 2:end]) .* (Ld[1:end, 2:end] - Ld[1:end, 1:end-1])
#     dy = 0.5 * stepsize * (c[1:end-1, 1:end] + c[2:end, 1:end]) .* (Ld[2:end, 1:end] - Ld[1:end-1, 1:end])

#     Ld[1:end, 1:end-1] .+= dx
#     Ld[1:end, 2:end] .-= dx
#     Ld[1:end-1, 1:end] .+= dy
#     Ld[2:end, 1:end] .-= dy
# end

function nld_step_scalar(Ld, c, stepsize, dx, dy) @inbounds begin
    rows::Int64=size(Ld,1)
    cols::Int64=size(Ld,2)

    # for k in 1:cols-1
    #     for j in 1:rows
    #         dx[j,k+1] = 0.5 * stepsize * (c[j,k] + c[j,k+1]) .* (Ld[j,k+1] - Ld[j,k])
    #     end
    # end
    # for k in 1:cols
    #     for j in 1:rows-1
    #         dy[j+1,k] = 0.5 * stepsize * (c[j,k] + c[j+1, k]) .* (Ld[j+1, k] - Ld[j,k])
    #     end
    # end

    for k in 1:cols-1
        for j in 1:rows-1
            cjk=c[j,k]
            Ldjk = Ld[j,k]
            dx[j,k+1] = 0.5 * stepsize * (cjk + c[j,k+1]) * (Ld[j,k+1] - Ldjk)
            dy[j+1,k] = 0.5 * stepsize * (cjk + c[j+1, k]) * (Ld[j+1, k] - Ldjk)
        end
    end
    j=rows
    for k in 1:cols-1
        dx[j,k+1] = 0.5 * stepsize * (c[j,k] + c[j,k+1]) .* (Ld[j,k+1] - Ld[j,k])
    end
    k=cols
    for j in 1:rows-1
        dy[j+1,k] = 0.5 * stepsize * (c[j,k] + c[j+1, k]) .* (Ld[j+1, k] - Ld[j,k])
    end

    for k in 1:cols
        for j in 1:rows
            Ld[j,k] += dx[j,k+1] - dx[j,k] + dy[j+1,k] - dy[j,k]
        end
    end
end end

pm_g1_diffusivity(Lx, Ly, k) =
    calculate_diffusivity′(Lx, Ly, k) do dL
        exp(-dL)
    end

pm_g2_diffusivity(Lx, Ly, k) =
    calculate_diffusivity′(Lx, Ly, k) do dL
        1.0 / (1.0 + dL)
    end

weickert_diffusivity(Lx, Ly, k) =
    calculate_diffusivity′(Lx, Ly, k) do dL
        1.0 - exp(-3.315 / (dL * dL * dL * dL))
    end

charbonnier_diffusivity(Lx, Ly, k) =
    calculate_diffusivity′(Lx, Ly, k) do dL
        1.0 / sqrt(1.0 + dL)
    end

function calculate_diffusivity′(func, Lx, Ly, k)
    dst = zeros(size(Lx))
    invk2 = 1.0 / (k * k)
    nrow, ncol = size(Lx)
    @inbounds begin
        @simd for j = 1:nrow
            @simd for k = 1:ncol
                lx, ly = Lx[j, k], Ly[j, k]
                dL = (lx * lx + ly * ly) * invk2
                dst[j, k] = func(dL)
            end
        end
    end
    dst
end

function compute_k_percentile(img, perc; gscale = 1.0, nbins = 300)
    Lsmooth = imfilter(img, Kernel.gaussian(gscale))
    fy, fx = Kernel.scharr()
    fy .*= 32
    fx .*= 32
    # fx,fy = Kernel.ando3()

    w = (size(fx)[1] + 1) ÷ 2
    Lx = @view imfilter(Lsmooth, fx)[w:end-w+1, w:end-w+1]
    Ly = @view imfilter(Lsmooth, fy)[w:end-w+1, w:end-w+1]
    compute_k_percentile′(Lx, Ly, perc, nbins)
end

function compute_k_percentile′(Lx, Ly, perc, nbins = 300) @inbounds begin
    rows::Int64 = size(Lx, 1)
    cols::Int64 = size(Lx, 2)

    hist = zeros(Int32, nbins)

    hmax = 0.0

    for k in 1:cols
        for j in 1:rows
            lx = Lx[j,k]
            ly = Ly[j,k]
            hmax = max(hmax, lx^2 + ly^2)
        end
    end
    hmax = sqrt(hmax)
    indexscale = nbins / hmax

    for k in 1:cols
        for j in 1:rows
            lx = Lx[j,k]
            ly = Ly[j,k]
            modg = sqrt(lx^2 + ly^2)
            nbin = ceil(Int, modg * indexscale)  # limiting to nbins should not be necessary
            hist[max(1, nbin)] += if nbin > 0 && modg > 1e-10 1 else 0 end
        end
    end
    nthreshold = floor(Int, sum(hist) * perc)

    k = findfirst(hx -> hx > nthreshold, cumsum(hist))

    return if k == nothing
        0.03
    else
        # (hmax * k) / nbins
        k / indexscale
    end
end end

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
    imshow(sqrt.(Lx .^ 2 + Ly .^ 2))
    imshow(sqrt.(Lx .^ 2 + Ly .^ 2) .< gthresh)
end

function demo_nld()
    Lflow = charbonnier_diffusivity(Lx, Ly, 0.01)

    imshow(copy(Lt))
    for _ = 1:3
        for _ = 1:100
            nld_step_scalar(Lt, Lflow, 0.05)
        end
        imshow(copy(Lt))
    end
end

function compute_derivative_kernels(scale)

    ksize = 3 + 2*(scale-1)

    # ## The usual Scharr kernel
    # if (scale == 1)
    #     return KernelFactors.scharr()
    # end

    kx = centered(zeros(ksize))
    ky = centered(zeros(ksize))

    w = 10.0/3.0
    norm = 1.0/(2.0*scale*(w+2.0))

    ky[-ksize÷2] = norm
    ky[0] = w*norm
    ky[ksize÷2] = norm

    kx[-ksize÷2] = -1
    kx[0] = 0
    kx[ksize÷2] = 1

    (kernelfactors((ky, kx)), kernelfactors((kx, ky)))
end
