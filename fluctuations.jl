function fluctuation_xy(a::Field)
    # Create average field (wrapped around lazy Average)
    a_avg_xy = Field{Nothing, Nothing, Center}(a.grid)
    set!(a_avg_xy, Average(a, dims=(1, 2)))  # Now a_avg_xy.data is ready
    @show a_avg_xy
    # Create fluctuation field with same grid and architecture
    a_fluctuation = Field{Center, Center, Face}(a.grid)
    set!(a_fluctuation, a .- a_avg_xy)

    # GPU-friendly broadcast subtraction
    compute!(a_fluctuation)
    return a_fluctuation
end