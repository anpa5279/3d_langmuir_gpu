using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃         
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃² 

@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) = tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

function update_aux_fields!(sim)
    model = sim.model
    grid = model.grid
    velocities = model.velocities
    νₑ = model.auxiliary_fields.νₑ

    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        νₑ[i, j, k] = smagorinsky_visc(i, j, k, grid, velocities, 0.1)
    end

    return nothing
end

function smagorinsky_visc(i, j, k, grid, velocities, C)
    u = velocities.u
    v = velocities.v
    w = velocities.w
    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    #@show Σ², Σ²_tensor
    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    cˢ² = C^2

    return cˢ² * Δᶠ^2 * sqrt(2Σ²)
end

# Horizontal viscous fluxes for isotropic diffusivities

@inline function viscous_flux_ux(i, j, k, grid, ν, u)
    return @inbounds -2 * ν[i, j, k] * Σ₁₁(i, j, k, grid, u)
end
@inline function viscous_flux_vx(i, j, k, grid, ν, u, v)
    return @inbounds -2 * ν[i, j, k] * Σ₁₂(i, j, k, grid, u, v)
end
@inline function viscous_flux_wx(i, j, k, grid, ν, u, w)
    return @inbounds -2 * ν[i, j, k] * Σ₁₃(i, j, k, grid, u, w)
end
@inline function viscous_flux_uy(i, j, k, grid, ν, u, v)
    return @inbounds -2 * ν[i, j, k] * Σ₁₂(i, j, k, grid, u, v)
end
@inline function viscous_flux_vy(i, j, k, grid, ν, v)
    return @inbounds -2 * ν[i, j, k] * Σ₂₂(i, j, k, grid, v)
end
@inline function viscous_flux_wy(i, j, k, grid, ν, v, w)
    return @inbounds -2 * ν[i, j, k] * Σ₂₃(i, j, k, grid, v, w)
end

# Vertical viscous fluxes for isotropic diffusivities
@inline function viscous_flux_uz(i, j, k, grid, ν, u, w)
    return @inbounds -2 * ν[i, j, k] * Σ₁₃(i, j, k, grid, u, w)
end
@inline function viscous_flux_vz(i, j, k, grid, ν, v, w)
    return @inbounds -2 * ν[i, j, k] * Σ₂₃(i, j, k, grid, v, w)
end
@inline function viscous_flux_wz(i, j, k, grid, ν, w)
    return @inbounds -2 * ν[i, j, k] * Σ₃₃(i, j, k, grid, w)
end

#diffusivity
@inline function diffusive_flux_x(i, j, k, grid, C, c)
    return @inbounds - C[i, j, k]  * ∂xᶠᶜᶜ(i, j, k, grid, c)
end 
@inline function diffusive_flux_y(i, j, k, grid, C, c)
    return @inbounds - C[i, j, k]  * ∂yᶜᶠᶜ(i, j, k, grid, c)
end
@inline function diffusive_flux_z(i, j, k, grid, C, c)
    return @inbounds - C[i, j, k]  * ∂zᶜᶜᶠ(i, j, k, grid, c)
end

#####
##### Base difference operators
#####

@inline δxᶜᵃᵃ(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δxᶠᵃᵃ(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δyᵃᶜᵃ(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δyᵃᶠᵃ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δzᵃᵃᶜ(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δzᵃᵃᶠ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]

#these are the discrete forcing functions
@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ
    viscous_flux_ux = model_fields.flux_ux
    viscous_flux_uy = model_fields.flux_uy
    viscous_flux_uz = model_fields.flux_uz
    dflux_ux = δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux)
    dflux_uy = δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy)
    dflux_uz = δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz)
    return @inbounds -(dflux_ux[i, j, k] + dflux_uy[i, j, k] + dflux_uz[i, j, k])
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ
    viscous_flux_vx = model_fields.flux_vx
    viscous_flux_vy = model_fields.flux_vy
    viscous_flux_vz = model_fields.flux_vz
    dflux_vx = δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx)
    dflux_vy = δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy)
    dflux_vz = δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz)
    return @inbounds -(dflux_vx[i, j, k] + dflux_vy[i, j, k] + dflux_vz[i, j, k])
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ
    viscous_flux_wx = model_fields.flux_wx
    viscous_flux_wy = model_fields.flux_wy
    viscous_flux_wz = model_fields.flux_wz
    dflux_wx = δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx)
    dflux_wy = δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy)
    dflux_wz = δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz)
    return @inbounds -(dflux_wx[i, j, k] + dflux_wy[i, j, k] + dflux_wz[i, j, k])
end

@inline function ∇_dot_qᶜ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    scalar = model_fields.T
    ν = model_fields.νₑ
    diff_flux_x = model_fields.flux_Tx
    diff_flux_y = model_fields.flux_Ty
    diff_flux_z = model_fields.flux_Tz
    dflux_x = δxᶜᵃᵃ(i, j, k, grid, diff_flux_x)
    dflux_y = δyᵃᶜᵃ(i, j, k, grid, diff_flux_y)
    dflux_z = δzᵃᵃᶜ(i, j, k, grid, diff_flux_z)
    return @inbounds -(dflux_x[i, j, k] + dflux_y[i, j, k] + dflux_z[i, j, k])
end 