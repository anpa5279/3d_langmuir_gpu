function fluctuation_xy(a::Field)
    # Compute horizontal (x, y) average at each z level
    a_avg_xy = Field(Average(a, dims=(1, 2)))  
    compute!(a_avg_xy)                         # Evaluate the average on the GPU

    # Create an output Field with the same grid
    a_fluctuation = Field(a.grid)

    # GPU-friendly broadcasting subtraction
    @. a_fluctuation.data = a.data - a_avg_xy.data

    return a_fluctuation
end