using Images: Kernel

using ImageTransformations: imresize
using ImageFiltering: imfilter!

fy, fx = Kernel.scharr()
fy .*=32
fx .*=32

# fy, fx = Kernel.ando3()
# fy, fx = Kernel.ando5()

################################################################
struct AKAZE

    options_::AKAZEOptions                      ###< Configuration options for AKAZE
    evolution_::Vector{TEvolution}              ###< Vector of nonlinear diffusion evolution

    ### FED parameters
    ncycles_::Int32                             ###< Number of cycles
    reordering_::Bool                           ###< Flag for reordering time steps
    tsteps_::Vector{Vector{Float32}}            ###< Vector of FED dynamic time steps
    nsteps_::Vector{Int32}                      ###< Vector of number of steps per cycle

    # ### Matrices for the M-LDB descriptor computation
    # descriptorSamples_
    # descriptorBits_
    # bitMask_

    ### Computation times variables in ms
    timing_::AKAZETiming

    function AKAZE(options::AKAZEOptions)

        ## Smallest possible octave and allow one scale if the image is small
        octavemax = min(options.omax, 1+mylog2(options.img_width ÷ 80), 1+mylog2(options.img_height ÷ 40))

        evolution =
            map(Iterators.product(0:options.nsublevels-1, 0:octavemax-1)) do (j, i)
                construct_tevolution(
                    image_width = options.img_width >> i,
                    image_height = options.img_height >> i,
                    esigma = options.soffset * 2.0^(j / options.nsublevels + i),
                    octave = i,
                    sublevel = j,
                )
            end[:]

        ncycles = length(evolution) - 1
        reordering = true

        ## Allocate memory for the number of cycles and time steps
        ntau = map(2:length(evolution)) do i
            ttime = evolution[i].etime - evolution[i-1].etime

            fed_tau_by_process_time(Float64(ttime), 1, 0.25, reordering)
        end
        nsteps, tsteps = [[x...] for x in zip(ntau...)]

        new(
            options,
            evolution,
            ncycles,
            reordering,
            tsteps,
            nsteps,
            # 0,
            # 0,
            # 0,
            AKAZETiming(0, 0, 0, 0, 0, 0, 0),
        )
    end
end


mylog2(i::Int, acc = 0) =
    if i <= 1
        acc
    else
        mylog2(i >> 1, acc + 1)
    end

@with_kw struct Point
    x::Float64
    y::Float64
end

@with_kw struct KeyPoint
    pt::Point
    size::Float64
    angle::Float64 = 0.0
    response::Float64 = 0.0
    octave::Int64 = 0
    class_id::Int64 = -1
end

KeyPoint(pt, size) = KeyPoint(pt, size, 0.0, 0.0, 0, -1)

################################################################
function Create_Nonlinear_Scale_Space(akaze, img)

    t1 = time_ns()

    ## Copy the original image to the first level of the evolution
    imfilter!(akaze.evolution_[1].Lt, img, AkazeGauss(akaze.options_.soffset))
    akaze.evolution_[1].Lsmooth .= akaze.evolution_[1].Lt
    imfilter!(akaze.evolution_[1].Lx, akaze.evolution_[1].Lsmooth, fx)
    imfilter!(akaze.evolution_[1].Lx, akaze.evolution_[1].Lx, AkazeGauss(akaze.options_.soffset))
    imfilter!(akaze.evolution_[1].Ly, akaze.evolution_[1].Lsmooth, fy)
    imfilter!(akaze.evolution_[1].Ly, akaze.evolution_[1].Ly, AkazeGauss(akaze.options_.soffset))

    ## First compute the kcontrast factor
    # akaze.options_.kcontrast =
    mykcontrast = compute_k_percentile(
        img,
        akaze.options_.kcontrast_percentile,
        gscale = 1.0,
        nbins = akaze.options_.kcontrast_nbins,
    )

    t2 = time_ns()
    akaze.timing_.kcontrast = t2 - t1

    ## Now generate the rest of evolution levels
    for i = 2:length(akaze.evolution_)
        if akaze.evolution_[i].octave > akaze.evolution_[i-1].octave
            akaze.evolution_[i].Lt .= halfsample_image(akaze.evolution_[i-1].Lt)
            # akaze.options_.kcontrast = akaze.options_.kcontrast * 0.75
            mykcontrast = mykcontrast * 0.75
        else
            akaze.evolution_[i].Lt .= akaze.evolution_[i-1].Lt
        end

        imfilter!(akaze.evolution_[i].Lsmooth, akaze.evolution_[i].Lt, AkazeGauss(1.0))

        ## Compute the Gaussian derivatives Lx and Ly
        imfilter!(akaze.evolution_[i].Lx, akaze.evolution_[i].Lsmooth, fx)
        imfilter!(akaze.evolution_[i].Ly, akaze.evolution_[i].Lsmooth, fy)

        calculate_diffusivity = select_diffusivity(akaze.options_.diffusivity)

        akaze.evolution_[i].Lflow .= calculate_diffusivity(
            akaze.evolution_[i].Lx,
            akaze.evolution_[i].Ly,
            # akaze.options_.kcontrast,
            mykcontrast,
        )

        ## Perform FED n inner steps
        for j = 1:akaze.nsteps_[i-1]
            nld_step_scalar(
                akaze.evolution_[i].Lt,
                akaze.evolution_[i].Lflow,
                akaze.tsteps_[i-1][j],
                akaze.evolution_[i].dx,
                akaze.evolution_[i].dy,
            )
        end
    end

    t2 = time_ns()
    akaze.timing_.scale = t2 - t1
end


halfsample_image(img) = imresize(img, size(img) .÷ 2)


################################################################
function Feature_Detection(akaze::AKAZE) #::kpts

    t1 = time_ns()

    Compute_Multiscale_Derivatives(akaze)
    Compute_Determinant_Hessian_Response(akaze)
    allkpts = Find_Scale_Space_Extrema(akaze)
    kpts = Do_Subpixel_Refinement(akaze, allkpts)

    t2 = time_ns()
    akaze.timing_.detector = t2 - t1
    kpts
end


################################################################
function Compute_Multiscale_Derivatives(akaze::AKAZE)

    t1 = time_ns()

    for ev in akaze.evolution_
        ratio = 2.0 ^ ev.octave
        sigma_size_ = round(Int, ev.esigma * akaze.options_.derivative_factor / ratio)
        fx, fy = compute_derivative_kernels(sigma_size_)

        imfilter!(ev.Lx, ev.Lsmooth, fx)
        imfilter!(ev.Ly, ev.Lsmooth, fy)
        imfilter!(ev.Lxx, ev.Lx, fx)
        imfilter!(ev.Lxy, ev.Lx, fy)
        imfilter!(ev.Lyy, ev.Ly, fy)
    end

    t2 = time_ns()
    akaze.timing_.derivatives = t2 - t1
end


################################################################
function Compute_Determinant_Hessian_Response(akaze::AKAZE; k=0.06)

    for ev in akaze.evolution_

        if akaze.options_.verbosity
            @info "Computing detector response. Determinant of Hessian. Evolution time: $ev.etime"
        end

        ratio = 2.0 ^ ev.octave
        sigma_size = round(Int, ev.esigma*akaze.options_.derivative_factor/ratio)
        sigma_size_quat = sigma_size*sigma_size*sigma_size*sigma_size

        # ev.Ldet .= (ev.Lxx .* ev.Lyy - ev.Lxy .^ 2 - k * (ev.Lxx + ev.Lyy).^2) * sigma_size_quat
        for x in eachindex(ev.Ldet)
            @inbounds ev.Ldet[x] = (ev.Lxx[x] * ev.Lyy[x] - ev.Lxy[x] ^ 2 - k * (ev.Lxx[x] + ev.Lyy[x]).^2) * sigma_size_quat
        end
    end
end


################################################################
function Find_Scale_Space_Extrema(akaze::AKAZE) @inbounds begin
    value = 0.0
    dist = 0.0
    ratio = 0.0
    smax = 0.0
    id_repeated = 0
    sigma_size_ = 0
    left_x = 0
    right_x = 0
    up_y = 0
    down_y = 0

    is_extremum = false
    is_repeated = false
    is_out = false

    point = KeyPoint(Point(0.0, 0.0), 0)
    kpts_aux = KeyPoint[]
    kpts = KeyPoint[]

    ## Set maximum size
    smax = if akaze.options_.descriptor in (SURF_UPRIGHT, SURF, MLDB_UPRIGHT, MLDB)
        10.0 * sqrt(2.0)
    else  # (MSURF_UPRIGHT, MSURF)
        12.0 * sqrt(2.0)
    end

    t1 = time_ns()

    mydthreshold = akaze.options_.dthreshold
    mymin_dthreshold = akaze.options_.min_dthreshold

    for i in 1:length(akaze.evolution_)
        ev = akaze.evolution_[i]
        myscale = ev.esigma * akaze.options_.derivative_factor
        ev_Ldet = ev.Ldet
        rows::Int64 = size(ev_Ldet,1)
        cols::Int64 = size(ev_Ldet,2)

        # for k in 2:cols-1, j in 2:rows-1
        for k in 2:cols-1
            for j in 2:rows-1

                is_extremum = false
                is_repeated = false
                is_out = false
                value = ev_Ldet[j, k]

                ## Filter the points with the detector threshold
                # mypredicate = value > akaze.options_.dthreshold &&
                    # value >= akaze.options_.min_dthreshold &&
                mypredicate = value > mydthreshold &&
                    value >= mymin_dthreshold &&
                    value >= ev_Ldet[j, k-1] &&
                    value > ev_Ldet[j, k+1] &&
                    value >= ev_Ldet[j-1, k-1] &&
                    value >= ev_Ldet[j-1, k] &&
                    value >= ev_Ldet[j-1, k+1] &&
                    value > ev_Ldet[j+1, k-1] && value > ev_Ldet[j+1, k] && value > ev_Ldet[j+1, k+1]
                if mypredicate

                    is_extremum = true
                    # point.pt.x = k - 1
                    # point.pt.y = j - 1
                    # point.size = ev.esigma * akaze.options_.derivative_factor
                    # point.response = abs(value)
                    # point.octave = ev.octave
                    # point.class_id = i
                    point = KeyPoint(
                        Point(k - 1, j - 1),
                        myscale,
                        0.0,
                        abs(value),
                        ev.octave,
                        i
                    )

                    ratio = 2.0^point.octave
                    sigma_size_ = round(Int64, point.size / ratio)

                    ## Compare response with the same and lower scale
                    for (pki, otherpoint) in enumerate(kpts_aux)

                        if point.class_id - 1 == otherpoint.class_id || point.class_id == otherpoint.class_id

                            dist = ((point.pt.x * ratio - otherpoint.pt.x) *
                                (point.pt.x * ratio - otherpoint.pt.x) +
                                (point.pt.y * ratio - otherpoint.pt.y) *
                                (point.pt.y * ratio - otherpoint.pt.y))

                            if dist <= point.size * point.size
                                if point.response > otherpoint.response
                                    id_repeated = pki
                                    is_repeated = true
                                else
                                    is_extremum = false
                                end
                                break
                            end
                        end
                    end

                    ## Check out of bounds
                    if is_extremum
                        ## Check that the point is under the image limits for the descriptor computation
                        left_x = round(Int64, point.pt.x - smax * sigma_size_) - 1
                        right_x = round(Int64, point.pt.x + smax * sigma_size_) + 1
                        up_y = round(Int64, point.pt.y - smax * sigma_size_) - 1
                        down_y = round(Int64, point.pt.y + smax * sigma_size_) + 1

                        if left_x < 0 || right_x >= cols || up_y < 0 || down_y >= rows
                            is_out = true
                        end

                        if is_out == false
                            # point.pt.x = point.pt.x * ratio + 0.5*(ratio-1.0)
                            # point.pt.y = point.pt.y * ratio + 0.5*(ratio-1.0)
                            point = KeyPoint(
                                pt=Point(
                                    x=point.pt.x * ratio + 0.5*(ratio-1.0),
                                    y=point.pt.y * ratio + 0.5*(ratio-1.0)
                                ),
                                size=point.size,
                                angle=point.angle,
                                response=point.response,
                                octave=point.octave,
                                class_id=point.class_id
                            )
                            if is_repeated == false
                                # push!(kpts_aux, deepcopy(point))
                                push!(kpts_aux, point)
                            else
                                # kpts_aux[id_repeated] = deepcopy(point)
                                kpts_aux[id_repeated] = point
                            end
                        end ## if is_out
                    end ##if is_extremum
                end
            end end## for j, k
    end ## for i

    ## Now filter points with the upper scale level
    for (i, point) in enumerate(kpts_aux)

        is_repeated = false

        for j = i+1:length(kpts_aux)

            ## Compare response with the upper scale
            if (point.class_id + 1) == kpts_aux[j].class_id

                dist = ((point.pt.x - kpts_aux[j].pt.x) * (point.pt.x - kpts_aux[j].pt.x) +
                    (point.pt.y - kpts_aux[j].pt.y) * (point.pt.y - kpts_aux[j].pt.y))

                if dist <= point.size * point.size && point.response < kpts_aux[j].response
                    is_repeated = true
                    break
                end
            end
        end

        if is_repeated == false
            # push!(kpts, deepcopy(point))
            push!(kpts, point)
        end
    end

    t2 = time_ns()
    akaze.timing_.extrema = t2 - t1
    kpts
end
end


################################################################
function Do_Subpixel_Refinement(akaze, kpts)
    newkpts = KeyPoint[]

    t1 = time_ns()

    for kp in kpts
        ratio = 2.0^kp.octave
        k = 1 + round(Int, kp.pt.x/ratio)
        j = 1 + round(Int, kp.pt.y/ratio)

        Ldet = akaze.evolution_[kp.class_id].Ldet
        ## Compute the gradient
        Dx = 0.5 * (Ldet[j,k+1] - Ldet[j,k-1])
        Dy = 0.5 * (Ldet[j+1,k] - Ldet[j-1,k])

        ## Compute the Hessian
        Dxx = Ldet[j,k+1] + Ldet[j,k-1] - 2.0 * Ldet[j,k]
        Dyy = Ldet[j+1,k] + Ldet[j-1,k] - 2.0 * Ldet[j,k]
        Dxy = 0.25 * (Ldet[j+1,k+1] + Ldet[j-1,k-1] - Ldet[j-1,k+1] - Ldet[j+1,k-1])

        ## Solve the linear system
        # A = [Dxx Dxy
        #      Dxy Dyy]
        # b = -[Dx, Dy]
        A = SMatrix{2,2}(Dxx, Dxy, Dxy, Dyy)
        b = -SVector(Dx, Dy)

        dst = A \ b

        # if (fabs(dst(0)) <= 1.0 && fabs(dst(1)) <= 1.0) {
        power = 2 ^ akaze.evolution_[kp.class_id].octave
        newkp = KeyPoint(
            pt=Point(
                (k - 1 + dst[1])*power + 0.5*(power-1),
                (j - 1 + dst[2])*power + 0.5*(power-1)
            ),
            size=kp.size * 2.0,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id
        )
        # newkp.pt.x = k - 1 + dst[1]
        # newkp.pt.y = j - 1 + dst[2]
        # newkp.pt.x = newkp.pt.x*power + 0.5*(power-1)
        # newkp.pt.y = newkp.pt.y*power + 0.5*(power-1)
        # newkp.angle = 0.0
        ## In OpenCV the size of a keypoint is the diameter, not the radius
        # newkp.size *= 2.0;

        push!(newkpts, newkp)
        ## Delete the point since its not stable
        # else {
        # kpts.erase(kpts.begin()+i);
        # i--;
        # end
    end

    t2 = time_ns()
    akaze.timing_.subpixel = t2 - t1
    newkpts
end


function AkazeGauss(sigma)
    ksize_x = ceil(Int, 2.0*(1.0 + (sigma-0.8)/(0.3)))
    ## The kernel size must be and odd number
    if ((ksize_x % 2) == 0)
        ksize_x += 1
    end
    Kernel.gaussian((sigma,sigma), (ksize_x, ksize_x))
end
