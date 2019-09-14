using ImageTransformations: imresize


################################################################
mutable struct AKAZE

    options_::AKAZEOptions                      ###< Configuration options for AKAZE
    evolution_::Array{TEvolution}               ###< Vector of nonlinear diffusion evolution

    ### FED parameters
    ncycles_::Int32                             ###< Number of cycles
    reordering_::Bool                           ###< Flag for reordering time steps
    tsteps_::Array{Array{Float32}}              ###< Vector of FED dynamic time steps
    nsteps_::Array{Int32}                       ###< Vector of number of steps per cycle

    ### Matrices for the M-LDB descriptor computation
    descriptorSamples_
    descriptorBits_
    bitMask_

    ### Computation times variables in ms
    timing_::AKAZETiming

    function AKAZE(options::AKAZEOptions)

        ## Smallest possible octave and allow one scale if the image is small
        octavemax = min(
            options.omax,
            mylog2(options.img_width÷80),
            mylog2(options.img_height÷40)
        )

        evolution = map(Iterators.product(0:options.nsublevels-1, 0:octavemax)) do (j,i)
            construct_tevolution(
                image_width = options.img_width >> i,
                image_height = options.img_height >> i,
                esigma = options.soffset * 2.0^(j/options.nsublevels + i),
                octave = i,
                sublevel = j
            )
        end

        ncycles = length(evolution) - 1
        reordering = true

        ## Allocate memory for the number of cycles and time steps
        ntau = map(2:length(evolution)) do i
            ttime = evolution[i].etime - evolution[i-1].etime

            fed_tau_by_process_time(Float64(ttime), 1, 0.25, reordering)
        end
        nsteps, tsteps = [[x...] for x in zip(ntau...)]

        new(options, evolution, ncycles, reordering, tsteps, nsteps, 0,0,0,AKAZETiming(0,0,0,0,0,0,0))
    end
end


mylog2(i::Int, acc=0) = if i==1 acc else mylog2(i>>1,acc+1) end


################################################################
function Create_Nonlinear_Scale_Space(akaze, img)

    fx,fy = Kernel.ando3()

    t1 = time_ns()

    ## Copy the original image to the first level of the evolution
    imfilter!(akaze.evolution_[1].Lt, img, Kernel.gaussian(akaze.options_.soffset))
    akaze.evolution_[1].Lsmooth .= akaze.evolution_[1].Lt

    ## First compute the kcontrast factor
    akaze.options_.kcontrast = compute_k_percentile(img, akaze.options_.kcontrast_percentile,
                                                    gscale=1.0, nbins=akaze.options_.kcontrast_nbins)

    t2 = time_ns();
    akaze.timing_.kcontrast = t2-t1

    ## Now generate the rest of evolution levels
    for i in 2:length(akaze.evolution_)
        if akaze.evolution_[i].octave > akaze.evolution_[i-1].octave
            akaze.evolution_[i].Lt = halfsample_image(akaze.evolution_[i-1].Lt)
            akaze.options_.kcontrast = akaze.options_.kcontrast * 0.75
        else
            akaze.evolution_[i].Lt .= akaze.evolution_[i-1].Lt
        end

        imfilter!(akaze.evolution_[i].Lsmooth, akaze.evolution_[i].Lt, Kernel.gaussian(1.0))

        ## Compute the Gaussian derivatives Lx and Ly
        imfilter!(akaze.evolution_[i].Lx, akaze.evolution_[i].Lsmooth, fx)
        imfilter!(akaze.evolution_[i].Ly, akaze.evolution_[i].Lsmooth, fy)

        calculate_diffusivity = select_diffusivity(akaze.options_.diffusivity)

        akaze.evolution_[i].Lflow .= calculate_diffusivity(
            akaze.evolution_[i].Lx,
            akaze.evolution_[i].Ly,
            akaze.options_.kcontrast
        )

        ## Perform FED n inner steps
        for j in 1:akaze.nsteps_[i-1]
            nld_step_scalar(akaze.evolution_[i].Lt, akaze.evolution_[i].Lflow, akaze.tsteps_[i-1][j])
        end
    end

    t2 = time_ns()
    akaze.timing_.scale = t2-t1
end


halfsample_image(img) = imresize(img, size(img).÷2)


#=
################################################################
function Feature_Detection(akaze)#::kpoints

    t1 = time_ns()

    vector<cv::KeyPoint>().swap(kpts)
    Compute_Determinant_Hessian_Response()
    Find_Scale_Space_Extrema(kpts)
    Do_Subpixel_Refinement(kpts)

    t2 = time_ns()
    akaze.timing_.detector = t2-t1
end
=#
