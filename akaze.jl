################################################################
struct AKAZE

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
            options_.omax,
            mylog2(options_.img_width÷80),
            mylog2(options_.img_height÷40)
        )

        evolution = map(0:octavemax) do i
            level_width = options_.img_width >> i
            level_height = options_.img_height >> i

            map(0:options_.nsublevels-1) do j
                esigma = options_.soffset * 2f0^(Float32(j)/Float32(options_.nsublevels)) + i)

                TEvolution(
                [zeros(level_height, level_width) for _ in 1:9]...,
                esigma = esigma,
                sigma_size = round(Int, esigma),
                etime = 0.5f0*(esigma*esigma),
                octave = i,
                sublevel = j
                )
            end
        end |> Iterators.flatten |> collect

        ncycles = evolution_.size()-1
        reordering = true

        ## Allocate memory for the number of cycles and time steps
        ntau = map(2:evolution_.size()) do i
            ttime = evolution[i].etime - evolution[i-1].etime
            fed_tau_by_process_time(ttime, 1, 0.25, reordering)
        end
        nsteps, tsteps = zip(ntau...)

        new(options, evolution, ncycles, reordering, tsteps, nsteps, 0,0,0,AKAZETiming(0,0,0,0,0,0))
    end
end

mylog2(i::Int) = if i==1 0 else 1+mylog2(i>>1) end



/* ************************************************************************* */
int AKAZE::Create_Nonlinear_Scale_Space(const cv::Mat& img) {

  double t1 = 0.0, t2 = 0.0;

  if (evolution_.size() == 0) {
    cerr << "Error generating the nonlinear scale space!!" << endl;
    cerr << "Firstly you need to call AKAZE::Allocate_Memory_Evolution()" << endl;
    return -1;
  }

  t1 = cv::getTickCount();

  // Copy the original image to the first level of the evolution
  img.copyTo(evolution_[0].Lt);
  gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);
  evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);

  // First compute the kcontrast factor
  options_.kcontrast = compute_k_percentile(img, options_.kcontrast_percentile,
                                            1.0, options_.kcontrast_nbins, 0, 0);

  t2 = cv::getTickCount();
  timing_.kcontrast = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution_.size(); i++) {

    if (evolution_[i].octave > evolution_[i-1].octave) {
      halfsample_image(evolution_[i-1].Lt, evolution_[i].Lt);
      options_.kcontrast = options_.kcontrast*0.75;
    }
    else {
      evolution_[i-1].Lt.copyTo(evolution_[i].Lt);
    }

    gaussian_2D_convolution(evolution_[i].Lt, evolution_[i].Lsmooth, 0, 0, 1.0);

    // Compute the Gaussian derivatives Lx and Ly
    image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0);
    image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1);

    // Compute the conductivity equation
    switch (options_.diffusivity) {
      case PM_G1:
        pm_g1(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
      break;
      case PM_G2:
        pm_g2(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
      break;
      case WEICKERT:
        weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
      break;
      case CHARBONNIER:
        charbonnier_diffusivity(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
      break;
      default:
        cerr << "Diffusivity: " << options_.diffusivity << " is not supported" << endl;
    }

    // Perform FED n inner steps
    for (int j = 0; j < nsteps_[i-1]; j++)
      nld_step_scalar(evolution_[i].Lt, evolution_[i].Lflow, evolution_[i].Lstep, tsteps_[i-1][j]);
  }

  t2 = cv::getTickCount();
  timing_.scale = 1000.0*(t2-t1) / cv::getTickFrequency();

      return 0;

      }
