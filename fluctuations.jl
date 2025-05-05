function fluctuation_xy(a::Field)
    # Create average field (wrapped around lazy Average)
    a_avg_xy = Field{Nothing, Nothing, Center}(a.grid, Average(a, dims=(1, 2)))
    compute!(a_avg_xy)  # Now a_avg_xy.data is ready
    @show a_avg_xy
    # Create fluctuation field with same grid and architecture
    a_fluctuation = Field{Center, Center, Face}(a.grid; architecture = architecture(a))

    # GPU-friendly broadcast subtraction
    set!(a_fluctuation, a.data .- a_avg_xy.data)
    return a_fluctuation
end