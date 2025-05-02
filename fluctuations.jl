function fluctuation_xy(a::Field)
    # Compute horizontal (x, y) average at each z level
    a_avg_xy = Field(Average(a, dims=(1, 2)))  
    @show a_avg_xy
    compute!(a_avg_xy)                         # Evaluate the average on the GPU
    # Create an output Field with the same grid
    a_fluctuation = Field(a - a_avg_xy)
    @show a_fluctuation.data
    compute!(a_fluctuation)                     # Evaluate the fluctuation on the GPU
    @show a_fluctuation
    return a_fluctuation
end