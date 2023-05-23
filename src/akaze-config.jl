using Parameters
using StaticArrays: SizedMatrix, SMatrix, SVector

################################################################
### Lookup table for 2d gaussian (sigma = 2.5) where (1,1) is top left and (7,7) is bottom right
const gauss25 = [
    0.02546481f0 0.02350698f0 0.01849125f0 0.01239505f0 0.00708017f0 0.00344629f0 0.00142946f0
    0.02350698f0 0.02169968f0 0.01706957f0 0.01144208f0 0.00653582f0 0.00318132f0 0.00131956f0
    0.01849125f0 0.01706957f0 0.01342740f0 0.00900066f0 0.00514126f0 0.00250252f0 0.00103800f0
    0.01239505f0 0.01144208f0 0.00900066f0 0.00603332f0 0.00344629f0 0.00167749f0 0.00069579f0
    0.00708017f0 0.00653582f0 0.00514126f0 0.00344629f0 0.00196855f0 0.00095820f0 0.00039744f0
    0.00344629f0 0.00318132f0 0.00250252f0 0.00167749f0 0.00095820f0 0.00046640f0 0.00019346f0
    0.00142946f0 0.00131956f0 0.00103800f0 0.00069579f0 0.00039744f0 0.00019346f0 0.00008024f0
]


################################################################
### AKAZE Descriptor Type
@enum DESCRIPTOR_TYPE begin
    SURF_UPRIGHT = 0   ###< Upright descriptors, not invariant to rotation
    SURF = 1
    MSURF_UPRIGHT = 2  ###< Upright descriptors, not invariant to rotation
    MSURF = 3
    MLDB_UPRIGHT = 4   ###< Upright descriptors, not invariant to rotation
    MLDB = 5
end

################################################################
### AKAZE Diffusivities
@enum DIFFUSIVITY_TYPE begin
    PM_G1 = 0
    PM_G2 = 1
    WEICKERT = 2
    CHARBONNIER = 3
end

select_diffusivity(diffusivity::DIFFUSIVITY_TYPE) =
    if diffusivity == PM_G1
        pm_g1_diffusivity
    elseif diffusivity == PM_G2
        pm_g2_diffusivity
    elseif diffusivity == WEICKERT
        weickert_diffusivity
    elseif diffusivity == CHARBONNIER
        charbonnier_diffusivity
    end


################################################################
### AKAZE Timing structure
mutable struct AKAZETiming
    kcontrast::Float64       ###< Contrast factor computation time in ms
    scale::Float64           ###< Nonlinear scale space computation time in ms
    derivatives::Float64     ###< Multiscale derivatives computation time in ms
    detector::Float64        ###< Feature detector computation time in ms
    extrema::Float64         ###< Scale space extrema computation time in ms
    subpixel::Float64        ###< Subpixel refinement computation time in ms
    descriptor::Float64      ###< Descriptors computation time in ms
end

################################################################
### AKAZE configuration options structure
@with_kw struct AKAZEOptions
    omin::Int32                           ###< Initial octave level (-1 means that the size of the input image is duplicated)
    omax::Int32 = 4                       ###< Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    nsublevels::Int32 = 4                 ###< Default number of sublevels per scale level
    img_width::Int64                      ###< Width of the input image
    img_height::Int64                     ###< Height of the input image
    soffset::Float32 = 1.6f0              ###< Base scale offset (sigma units)
    derivative_factor::Float32 = 1.5f0    ###< Factor for the multiscale derivatives
    sderivatives::Float32 = 1f0           ###< Smoothing factor for the derivatives
    diffusivity::DIFFUSIVITY_TYPE = PM_G2 ###< Diffusivity type

    dthreshold::Float32 = 3f-5            ###< Detector response threshold to accept point
    min_dthreshold::Float32 = 1f-5        ###< Minimum detector threshold to accept a point

    descriptor::DESCRIPTOR_TYPE = MLDB    ###< Type of descriptor
    descriptor_size::Int32 = 0            ###< Size of the descriptor in bits. 0->Full size
    descriptor_channels::Int32 = 3        ###< Number of channels in the descriptor (1, 2, 3)
    descriptor_pattern_size::Int32 = 10   ###< Actual patch size is 2*pattern_size*point.scale

    # kcontrast::Float32 = 1f-3             ###< The contrast factor parameter # This is actually a transient variable...
    kcontrast_percentile::Float32 = 7f-1  ###< Percentile level for the contrast factor
    kcontrast_nbins::UInt32 = 300         ###< Number of bins for the contrast factor histogram

    save_scale_space::Bool = false        ###< Set to true for saving the scale space images
    save_keypoints::Bool = false          ###< Set to true for saving the detected keypoints and descriptors
    show_results::Bool = true             ###< Set to true for displaying results
    verbosity::Bool = false               ###< Set to true for displaying verbosity information
end

################################################################
### AKAZE nonlinear diffusion filtering evolution
# @with_kw struct TEvolution{T,R}
#     Lx::T            ###< First order spatial derivatives
#     Ly::T            ###< First order spatial derivatives
#     Lxx::T  ###< Second order spatial derivatives
#     Lxy::T  ###< Second order spatial derivatives
#     Lyy::T  ###< Second order spatial derivatives
#     Lflow::T                ###< Diffusivity image
#     Lt::T                   ###< Evolution image
#     Lsmooth::T              ###< Smoothed image
#     Lstep::T                ###< Evolution step update
#     Ldet::R                 ###< Detector response
#     etime::Float32 = 0f0      ###< Evolution time
#     esigma::Float32 = 0f0     ###< Evolution sigma. For linear diffusion t = sigma^2 / 2
#     octave::UInt32 = 0x0      ###< Image octave
#     sublevel::UInt32 = 0x0    ###< Image sublevel in each octave
#     sigma_size::UInt32 = 0x0  ###< Integer sigma. For computing the feature detector responses
# end
@with_kw struct TEvolution
    Lx::Matrix{Float64}            ###< First order spatial derivatives
    Ly::Matrix{Float64}            ###< First order spatial derivatives
    Lxx::Matrix{Float64}  ###< Second order spatial derivatives
    Lxy::Matrix{Float64}  ###< Second order spatial derivatives
    Lyy::Matrix{Float64}  ###< Second order spatial derivatives
    Lflow::Matrix{Float64}                ###< Diffusivity image
    Lt::Matrix{Float64}                   ###< Evolution image
    Lsmooth::Matrix{Float64}              ###< Smoothed image
    Lstep::Matrix{Float64}                ###< Evolution step update
    Ldet::Matrix{Float64}                 ###< Detector response
    dx::Matrix{Float64}                 ###< used in nld_step_scalar
    dy::Matrix{Float64}                 ###< used in nld_step_scalar
    etime::Float32 = 0f0      ###< Evolution time
    esigma::Float32 = 0f0     ###< Evolution sigma. For linear diffusion t = sigma^2 / 2
    octave::UInt32 = 0x0      ###< Image octave
    sublevel::UInt32 = 0x0    ###< Image sublevel in each octave
    sigma_size::UInt32 = 0x0  ###< Integer sigma. For computing the feature detector responses
end

construct_tevolution(; image_width::Int64, image_height::Int64, esigma, octave, sublevel) =
    TEvolution(
        Lx = zeros(image_height, image_width),
        Ly = zeros(image_height, image_width),
        Lxx = zeros(image_height, image_width),
        Lxy = zeros(image_height, image_width),
        Lyy = zeros(image_height, image_width),
        Lflow = zeros(image_height, image_width),
        Lt = zeros(image_height, image_width),
        Lsmooth = zeros(image_height, image_width),
        Lstep = zeros(image_height, image_width),
        Ldet = zeros(image_height, image_width),
        dx = zeros(image_height, image_width+2),
        dy = zeros(image_height+2, image_width),
        esigma = esigma,
        sigma_size = round(Int, esigma),
        etime = 0.5f0 * (esigma * esigma),
        octave = octave,
        sublevel = sublevel,
    )

        # evolution = map(Iterators.product(0:octavemax, 0:options.nsublevels-1)) do (i,j)
        #     construct_tevolution(
        #         image_width = options.img_width >> i,
        #         image_height = options.img_height >> i,
        #         esigma = options.soffset * 2.0^(j/options.nsublevels + i),
        #         octave = i,
        #         sublevel = j
        #     )
        # end

# Float32[1.2800001, 1.8101934, 2.5600002, 3.6203868, 5.1200004, 7.2407737, 10.240001, 14.481547, 20.480001, 28.963095, 40.960003, 57.92619, 81.920006, 115.85238, 163.84001, 231.70476]
