using Pkg
using Statistics
using Printf
using Oceananigans

# simulation update function
function progress(simulation)
    u, v, w = simulation.model.velocities
    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                iteration(simulation),
                prettytime(time(simulation)),
                prettytime(simulation.Δt),
                maximum(abs, u), maximum(abs, v), maximum(abs, w),
                prettytime(simulation.run_wall_time))
    @info msg
    return nothing
end

# what to save to output manually 
function save_grid!(file, model) 
    grid = model.grid
    file["grid/Nx"] = grid.Nx
    file["grid/Ny"] = grid.Ny
    file["grid/Nz"] = grid.Nz
    file["grid/Lx"] = grid.Lx
    file["grid/Ly"] = grid.Ly
    file["grid/Lz"] = grid.Lz
    file["grid/Hx"] = grid.Hx
    file["grid/Hy"] = grid.Hy
    file["grid/Hz"] = grid.Hz
    file["grid/xᶜᵃᵃ"] = grid.xᶜᵃᵃ
    file["grid/yᵃᶜᵃ"] = grid.yᵃᶜᵃ
    file["grid/zᵃᵃᶜ"] = grid.z.cᵃᵃᶜ
    file["grid/Δxᶜᵃᵃ"] = grid.Δxᶜᵃᵃ
    file["grid/Δyᵃᶜᵃ"] = grid.Δyᵃᶜᵃ
    file["grid/z/Δᵃᵃᶜ"] = grid.z.Δᵃᵃᶜ
    file["grid/grid"] = grid
    return nothing
end

# BC functions
@inline function sflux(x, y, t) 
    if (x^2+y^2)^(1/2)<=rp
        return wp*Sj
    else
        return 0.0
    end
end

# stokes functions
function stokes_velocity(z, u₁₀)
    g = Oceananigans.defaults.gravitational_acceleration
    α = 0.00615
    fₚ = 2π * 0.13 * g / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    σ = a + 0.5 * df
    u_temp = 0.0
    for k in 1:nf
        u_temp = u_temp + (2.0 * α * g / (fₚ * σ) * exp(2.0 * σ^2 * z / g - (fₚ / σ)^4))
        σ = σ + df
    end 
    return df * u_temp
end
function dstokes_dz(z, u₁₀)
    g = Oceananigans.defaults.gravitational_acceleration
    α = 0.00615
    fₚ = 2π * 0.13 * g / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    σ = a + 0.5 * df
    du_temp = 0.0
    for k in 1:nf
        du_temp = du_temp + (4.0 * α * σ/ (fₚ) * exp(2.0 * σ^2 * z / g - (fₚ / σ)^4))
        σ = σ + df
    end 
    return df * du_temp
end 