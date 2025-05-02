function fluctuation_xy(a::Field)
    # Compute horizontal (x, y) average at each z level
    a_avg_xy = Field(Average(a, dims=(1, 2)))  
    @show a_avg_xy
    compute!(a_avg_xy)                         # Evaluate the average on the GPU
    # Step 2: Allocate fluctuation field with correct location and architecture
    a_fluctuation = Field(a.grid, loc=a.location, architecture=a.architecture)

    # Step 3: GPU-safe kernel to subtract average
    @kernel function subtract_mean!(f, a, avg)
        i, j, k = @index(Global, NTuple)
        f[i, j, k] = a[i, j, k] - avg[i, j, k]
    end

    launch!(subtract_mean!, a.grid, a_fluctuation, a, a_avg_xy)

    return a_fluctuation
end