using ImageTransformations: imresize


fx, fy = Kernel.ando3()


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

@with_kw mutable struct Point
    x
    y
end

@with_kw mutable struct KeyPoint
    Point2f 	pt
    float 	size
    float 	angle = -1
    float 	response = 0
    int 	octave = 0
    int 	class_id = -1
end


################################################################
function Create_Nonlinear_Scale_Space(akaze, img)

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


################################################################
function Feature_Detection(akaze) #::kpts

    t1 = time_ns()

    Compute_Multiscale_Derivatives(akaze)
    Compute_Determinant_Hessian_Response(akaze)
    kpts = Find_Scale_Space_Extrema(akaze)
    Do_Subpixel_Refinement(akaze, kpts)

    t2 = time_ns()
    akaze.timing_.detector = t2-t1
end


################################################################
function Compute_Multiscale_Derivatives()

    t1 = time_ns()

    for ev in akaze.evolution_
        # imfilter!(ev.Lx, ev.Lsmooth, fx)
        # imfilter!(ev.Ly, ev.Lsmooth, fy)
        imfilter!(ev.Lxx, ev.Lx, fx)
        imfilter!(ev.Lxy, ev.Lx, fy)
        imfilter!(ev.Lyy, ev.Ly, fy)
    end

    t2 = time_ns()
    akaze.timing_.derivatives = t2-t1
end


################################################################
function Compute_Determinant_Hessian_Response(akaze)

    for ev in akaze.evolution_

        if akaze.options_.verbosity
            @info "Computing detector response. Determinant of Hessian. Evolution time: $ev.etime"
        end

        ev.Ldet .= ev.Lxx .* ex.Lyy - ev.Lxy.^2
    end
end

#=
################################################################
function Find_Scale_Space_Extrema(akaze)

    value = 0.0
    dist = 0.0
    ratio = 0.0
    smax = 0.0
    npoints = 0
    id_repeated = 0
    sigma_size_ = 0
    left_x = 0
    right_x = 0
    up_y = 0
    down_y = 0

    is_extremum = false
    is_repeated = false
    is_out = false

    cv::KeyPoint point;
    vector<cv::KeyPoint> kpts_aux;

  // Set maximum size
  if (options_.descriptor == SURF_UPRIGHT || options_.descriptor == SURF ||
      options_.descriptor == MLDB_UPRIGHT || options_.descriptor == MLDB) {
    smax = 10.0*sqrtf(2.0f);
  }
  else if (options_.descriptor == MSURF_UPRIGHT || options_.descriptor == MSURF) {
    smax = 12.0*sqrtf(2.0f);
  }

  t1 = time_ns()

  for (size_t i = 0; i < evolution_.size(); i++) {
    for (int ix = 1; ix < evolution_[i].Ldet.rows-1; ix++) {

      float* ldet_m = evolution_[i].Ldet.ptr<float>(ix-1);
      float* ldet = evolution_[i].Ldet.ptr<float>(ix);
      float* ldet_p = evolution_[i].Ldet.ptr<float>(ix+1);

      for (int jx = 1; jx < evolution_[i].Ldet.cols-1; jx++) {

        is_extremum = false;
        is_repeated = false;
        is_out = false;
        value = ldet[jx];

        // Filter the points with the detector threshold
        if (value > options_.dthreshold && value >= options_.min_dthreshold &&
            value > ldet[jx-1] && value > ldet[jx+1] &&
            value > ldet_m[jx-1] && value > ldet_m[jx] && value > ldet_m[jx+1] &&
            value > ldet_p[jx-1] && value > ldet_p[jx] && value > ldet_p[jx+1]) {

          is_extremum = true;
          point.response = fabs(value);
          point.size = evolution_[i].esigma*options_.derivative_factor;
          point.octave = evolution_[i].octave;
          point.class_id = i;
          ratio = pow(2.0f, point.octave);
          sigma_size_ = fRound(point.size/ratio);
          point.pt.x = jx;
          point.pt.y = ix;

          // Compare response with the same and lower scale
          for (size_t ik = 0; ik < kpts_aux.size(); ik++) {

            if ((point.class_id-1) == kpts_aux[ik].class_id ||
                point.class_id == kpts_aux[ik].class_id) {

              dist = (point.pt.x*ratio-kpts_aux[ik].pt.x)*(point.pt.x*ratio-kpts_aux[ik].pt.x) +
                     (point.pt.y*ratio-kpts_aux[ik].pt.y)*(point.pt.y*ratio-kpts_aux[ik].pt.y);

              if (dist <= point.size*point.size) {
                if (point.response > kpts_aux[ik].response) {
                  id_repeated = ik;
                  is_repeated = true;
                }
                else {
                  is_extremum = false;
                }
                break;
              }
            }
          }

          // Check out of bounds
          if (is_extremum == true) {

            // Check that the point is under the image limits for the descriptor computation
            left_x = fRound(point.pt.x-smax*sigma_size_)-1;
            right_x = fRound(point.pt.x+smax*sigma_size_) +1;
            up_y = fRound(point.pt.y-smax*sigma_size_)-1;
            down_y = fRound(point.pt.y+smax*sigma_size_)+1;

            if (left_x < 0 || right_x >= evolution_[i].Ldet.cols ||
                up_y < 0 || down_y >= evolution_[i].Ldet.rows) {
              is_out = true;
            }

            if (is_out == false) {
              if (is_repeated == false) {
                point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                kpts_aux.push_back(point);
                npoints++;
              }
              else {
                point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                kpts_aux[id_repeated] = point;
              }
            } // if is_out
          } //if is_extremum
        }
      } // for jx
    } // for ix
  } // for i

  // Now filter points with the upper scale level
  for (size_t i = 0; i < kpts_aux.size(); i++) {

    is_repeated = false;
    const cv::KeyPoint& point = kpts_aux[i];
    for (size_t j = i+1; j < kpts_aux.size(); j++) {

      // Compare response with the upper scale
      if ((point.class_id+1) == kpts_aux[j].class_id) {

        dist = (point.pt.x-kpts_aux[j].pt.x)*(point.pt.x-kpts_aux[j].pt.x) +
            (point.pt.y-kpts_aux[j].pt.y)*(point.pt.y-kpts_aux[j].pt.y);

        if (dist <= point.size*point.size) {
          if (point.response < kpts_aux[j].response) {
            is_repeated = true;
            break;
          }
        }
      }
    }

    if (is_repeated == false)
      kpts.push_back(point);
  }

  t2 = time_ns()
      akaze.timing_.extrema = t2-t1

end
=#
