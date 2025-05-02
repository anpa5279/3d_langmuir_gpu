using CUDA
using Statistics
function fluctuation_xy(a)
    # Compute horizontal (x, y) average at each z level
    CUDA.@allowscalar a_avg_xy = Statistics.mean(a, dims=(1, 2))
    @show a_avg_xy
    # Create an output Field with the same grid
    CUDA.@allowscalar a_fluctuation = a .- a_avg_xy
    @show a_fluctuation
    return a_fluctuation
end