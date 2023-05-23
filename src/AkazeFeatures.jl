module AkazeFeatures

export
    AKAZEOptions,
    AKAZE,
    Create_Nonlinear_Scale_Space,
    Feature_Detection,
    Compute_Main_Orientation,
    Compute_Descriptors,

    DIFFUSIVITY_TYPE,
    PM_G1,
    PM_G2,
    WEICKERT,
    CHARBONNIER,

    load_image_as_grayscale,
    original_akaze_features,
    dump_keypoints_text,
    plot_features

include("akaze-config.jl")
include("akaze.jl")
include("akaze-descriptor.jl")
include("fed.jl")
include("nonlinear-diffusion.jl")
include("utils.jl")

end
