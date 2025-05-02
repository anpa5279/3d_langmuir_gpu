function stokes_velocity(z, u₁₀)
    u = Array{Float64}(undef, length(z))
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ = a + 0.5 * df
        u_temp = 0.0
        for k in 1:nf
            u_temp = u_temp + (2.0 * α * g_Earth / (fₚ * σ) * exp(2.0 * σ^2 * z[i] / g_Earth - (fₚ / σ)^4))
            σ = σ + df
        end 
        u[i] = df * u_temp
    end
    return u
end
function dstokes_dz(z, u₁₀)
    dudz = Array{Float64}(undef, length(z))
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ = a + 0.5 * df
        du_temp = 0.0
        for k in 1:nf
            du_temp = du_temp + (4.0 * α * σ/ (fₚ) * exp(2.0 * σ^2 * z[i] / g_Earth - (fₚ / σ)^4))
            σ = σ + df
        end 
        dudz[i] = df * du_temp
    end
    return dudz
end 