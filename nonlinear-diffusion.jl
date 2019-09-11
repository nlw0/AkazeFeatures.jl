using ImageFiltering
using TestImages
using ImageView

img = testimage("pirate")

Lsmooth = imfilter(img, Kernel.gaussian(1.0))

fx,fy = Kernel.ando3()

Lx = imfilter(Lsmooth, fx)
Ly = imfilter(Lsmooth, fy)
Lxx = imfilter(Lx, fx)
Lxy = imfilter(Lx, fy)
Lyy = imfilter(Ly, fy)

# imshow(Lx)
# imshow(Ly)
# imshow(Lxx)
# imshow(Lxy)
# imshow(Lyy)



# function weickert_diffusivity(Lx, Ly, k)
#     dst = zeros(size(Lx))
#     invk2 = 1.0 / (k * k)
#     weickert_diffusivity_calc(Lx, Ly, dst, invk2)
#     dst
# end

# function weickert_diffusivity_calc(Lx, Ly, dst, invk2)
#     sz = size(Lx)
#     # @inbounds for y in 1:sz[1], x in 1:sz[2]
#         # dst[y,x] = weickert_expression(Lx[y,x], Ly[y,x], invk2)
#     # end
#     @inbounds dst .= weickert_expression.(Lx, Ly, [invk2])
# end

# @inline function weickert_expression(lx, ly, invk2)
#     dL = invk2 * (lx*lx + ly*ly)
#     1.0 - exp(-3.315/(dL*dL*dL*dL))
# end








function pm_g1_diffusivity(Lx, Ly, k)
    dst = zeros(size(Lx))
    invk2 = 1.0 / (k * k)
    sz = size(Lx)
    @inbounds begin
        @simd for y in 1:sz[1]
            @simd for x in 1:sz[2]
                lx,ly = Lx[y,x], Ly[y,x]

                dst[y,x] = (-invk2*(lx*lx + ly*ly))
            end
        end
    end
    dst
end

function pm_g2_diffusivity(Lx, Ly, k)
    dst = zeros(size(Lx))
    invk2 = 1.0 / (k * k)
    sz = size(Lx)
    @inbounds begin
        @simd for y in 1:sz[1]
            @simd for x in 1:sz[2]
                lx, ly = Lx[y,x], Ly[y,x]

                dst[y,x] = 1.0 / (1.0+invk2*(lx*lx + ly*ly));
            end
        end
    end
    dst
end

function weickert_diffusivity(Lx, Ly, k)
    dst = zeros(size(Lx))
    invk2 = 1.0 / (k * k)
    sz = size(Lx)
    @inbounds begin
        @simd for y in 1:sz[1]
            @simd for x in 1:sz[2]
                lx,ly = Lx[y,x], Ly[y,x]

                dL = invk2 * (lx*lx + ly*ly)
                dst[y,x] = 1.0 - exp(-3.315/(dL*dL*dL*dL))
            end
        end
    end
    dst
end

function charbonnier_diffusivity(Lx, Ly, k)
    dst = zeros(size(Lx))
    invk2 = 1.0 / (k * k)
    sz = size(Lx)
    @inbounds begin
        @simd for y in 1:sz[1]
            @simd for x in 1:sz[2]
                lx,ly = Lx[y,x], Ly[y,x]

                den = sqrt(1.0+invk2*(lx*lx + ly*ly))
                dst[y,x] = 1.0 / den
            end
        end
    end
    dst
end

# aa = pm_g1_diffusivity(Lx, Ly, 0.01)
# aa = pm_g2_diffusivity(Lx, Ly, 0.01)
# aa = weickert_diffusivity(Lx, Ly, 0.01)
# aa = charbonnier_diffusivity(Lx, Ly, 0.01)


# imshow(Lsmooth)
# imshow(aa)

# compute the k contrast

hmax = maximum(zip(Lx[:], Ly[:])) do (lx, ly)
    lx^2+ly^2
end |> sqrt

Nbin = 300

hist = zeros(Int32, Nbin)

for (lx, ly) in zip(Lx[:], Ly[:])
    hist[convert(Int, 1+round(sqrt(lx^2 + ly^2) * (Nbin-1) / hmax))] +=1
end

imshow(sqrt.(Lx.^2+Ly.^2))

using Plots
# pyplot()
plot(hist)
