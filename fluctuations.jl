function fluctuation_xy(a::Field)
    # Compute horizontal (x, y) average at each z level
    a_avg_xy = Field(Average(a, dims=(1, 2)))  
    @show a_avg_xy
    compute!(a_avg_xy)                         # Evaluate the average on the GPU
    # Create an output Field with the same grid
    a_fluctuation = Field{Center, Center, Face}(a.grid)
    @show a_fluctuation
    # GPU-friendly broadcasting subtraction
    @. a_fluctuation.data = a.data - a_avg_xy.data

    return a_fluctuation
end