function fluctuation_xy(a::Field)
    # Compute horizontal (x, y) average at each z level
    @show a
    # Create average field (wrapped around lazy Average)
    a_avg_xy = Field(Average(a, dims=(1, 2)); architecture = architecture(a))
    compute!(a_avg_xy)  # Now a_avg_xy.data is ready
    @show a_avg_xy
    # Create fluctuation field with same grid and architecture
    a_fluctuation = Field{Center, Center, Face}(a.grid; architecture = architecture(a))

    # GPU-friendly broadcast subtraction
    set!(a_fluctuation, a.data .- a_avg_xy.data)
    return a_fluctuation
end