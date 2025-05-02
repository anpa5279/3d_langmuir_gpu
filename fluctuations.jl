function fluctuation_xy(a::Field)
    # Compute horizontal (x, y) average at each z level
    a_avg_xy = Field{Center, Center, Face}(Average(a, dims=(1, 2)))                     # Evaluate the average on the GPU
    @show a_avg_xy
    # Create an output Field with the same grid
    a_fluctuation = Field{Center, Center, Face}(a.grid)
    @show a_fluctuation
    # GPU-friendly broadcasting subtraction
    a_fluctuation .= a.data .- a_avg_xy
    @show a_fluctuation
    return a_fluctuation
end