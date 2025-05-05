function fluctuation_xy(a::Field)
    # Create average field (wrapped around lazy Average)
    a_avg_xy = Field(Average(a, dims=(1, 2)))  # Now a_avg_xy.data is ready
    compute!(a_avg_xy)  # Compute the average field
    # Create fluctuation field with same grid and architecture
    a_fluctuation = Field{Center, Center, Face}(a.grid)
    CUDA.allowscalar() a_fluctuation.= a .- a_avg_xy

    # GPU-friendly broadcast subtraction
    compute!(a_fluctuation)
    return a_fluctuation
end