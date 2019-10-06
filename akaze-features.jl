#!/usr/bin/env julia

using ArgParse: ArgParseSettings, @add_arg_table, parse_args, parse_item
import ArgParse

using AkazeFeatures
# include("akaze-config.jl")
# include("akaze-descriptor.jl")
# include("akaze.jl")
# include("fed.jl")
# include("nonlinear-diffusion.jl")
# include("utils.jl")

function ArgParse.parse_item(::Type{DIFFUSIVITY_TYPE}, str::AbstractString)
    return DIFFUSIVITY_TYPE(parse(Int, str))
end

s = ArgParseSettings()
@add_arg_table s begin
    "--nsublevels"
    help = "Number of sublevels per scale level"
    arg_type = Int
    default = 4

    "--omax"
    help = "Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)"
    arg_type = Int
    default = 4

    "--soffset"
    help="Base scale offset (sigma units)"
    arg_type = Float32
    default = 1.6f0

    "--derivative_factor"
    help="Factor for the multiscale derivatives"
    arg_type = Float32
    default = 1.5f0

    "--sderivatives"
    help="Smoothing factor for the derivatives"
    arg_type=Float32
    default = 1f0

    "--diffusivity"
    # arg_type=Int
    # default = Int(PM_G2)
    arg_type=DIFFUSIVITY_TYPE
    default = PM_G2
    help="Diffusivity type"

    "--dthreshold"
    help="Detector response threshold to accept point"
    arg_type=Float32
    default = 1f-3
    # default = 3f-5

    "--min_dthreshold"
    help="Minimum detector threshold to accept a point"
    arg_type=Float32
    default = 1f-5

    "--output"
    help="output filename"
    arg_type=String
    default = "keypoints.txt"

    "imagename"
    help = "Image file to be analyzed"
    required = true
end

parsed_args = parse_args(ARGS, s)

img = load_image_as_grayscale(parsed_args["imagename"])

img_height, img_width = size(img)

opt = AKAZEOptions(
    omin = 3,
    omax = parsed_args["omax"],
    nsublevels = parsed_args["nsublevels"],
    img_width = img_width,
    img_height = img_height,
    diffusivity = parsed_args["diffusivity"],
    dthreshold = parsed_args["dthreshold"],
)
akaze = AKAZE(opt)

Create_Nonlinear_Scale_Space(akaze, img)
kpts = Feature_Detection(akaze)
Compute_Main_Orientation.([akaze], kpts)
println("Extracted $(length(kpts)) points")
desc = Compute_Descriptors(akaze, kpts)

open(parsed_args["output"],"w") do io
    for row in dump_keypoints_text(kpts, desc)
        println(io, row)
    end
end
