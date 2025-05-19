using Oceananigans.Operators
using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ
using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃         
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃² 

@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) = tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

function update_visc!(sim)
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

#these are the discrete forcing functions
@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ

    visc_flux_ux = viscous_flux_ux(i, j, k, grid, ν, u)
    visc_flux_uy = viscous_flux_uy(i, j, k, grid, ν, u, v)
    visc_flux_uz = viscous_flux_uz(i, j, k, grid, ν, u, w)
    visc_flux_ux_f = viscous_flux_ux(i - 1, j, k, grid, ν, u)
    visc_flux_uy_c = viscous_flux_uy(i, j + 1, k, grid, ν, u, v)
    visc_flux_uz_c = viscous_flux_uz(i, j, k + 1, grid, ν, u, w)
    ux_flux = (visc_flux_ux - visc_flux_ux_f)/grid.Δxᶠᵃᵃ
    uy_flux = (visc_flux_uy - visc_flux_uy_c)/grid.Δyᵃᶜᵃ
    uz_flux = (visc_flux_uz - visc_flux_uz_c)/grid.z.Δᵃᵃᶜ
    return @inbounds - (ux_flux + uy_flux + uz_flux)
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ
    
    visc_flux_vx = viscous_flux_vx(i, j, k, grid, ν, u, v)
    visc_flux_vy = viscous_flux_vy(i, j, k, grid, ν, v)
    visc_flux_vz = viscous_flux_vz(i, j, k, grid, ν, v, w)
    visc_flux_vx_f = viscous_flux_vx(i + 1, j, k, grid, ν, u, v)
    visc_flux_vy_c = viscous_flux_vy(i, j - 1, k, grid, ν, v)
    visc_flux_vz_c = viscous_flux_vz(i, j, k +1, grid, ν, v, w)
    vx_flux = (visc_flux_vx - visc_flux_vx_f)/grid.Δxᶜᵃᵃ
    vy_flux = (visc_flux_vy - visc_flux_vy_c)/grid.Δyᵃᶠᵃ
    vz_flux = (visc_flux_vz - visc_flux_vz_c)/grid.z.Δᵃᵃᶜ
    return @inbounds - (vx_flux + vy_flux + vz_flux)
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ

    visc_flux_wx = viscous_flux_wx(i, j, k, grid, ν, u, w)
    visc_flux_wy = viscous_flux_wy(i, j, k, grid, ν, v, w)
    visc_flux_wz = viscous_flux_wz(i, j, k, grid, ν, w)
    visc_flux_wx_f = viscous_flux_wx(i + 1, j, k, grid, ν, u, w)
    visc_flux_wy_c = viscous_flux_wy(i, j + 1, k, grid, ν, v, w)
    visc_flux_wz_c = viscous_flux_wz(i, j, k - 1, grid, ν, w)
    wx_flux = (visc_flux_wx - visc_flux_wx_f)/grid.Δxᶜᵃᵃ
    wy_flux = (visc_flux_wy - visc_flux_wy_c)/grid.Δyᵃᶜᵃ
    wz_flux = (visc_flux_wz - visc_flux_wz_c)/grid.z.Δᵃᵃᶠ
    return @inbounds - (wx_flux + wy_flux + wz_flux)
end

@inline function ∇_dot_qᶜ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    scalar = model_fields.T
    ν = model_fields.νₑ

    diff_flux_x = diffusive_flux_x(i, j, k, grid, ν, scalar)
    diff_flux_y = diffusive_flux_y(i, j, k, grid, ν, scalar)
    diff_flux_z = diffusive_flux_z(i, j, k, grid, ν, scalar)
    diff_flux_x_f = diffusive_flux_x(i + 1, j, k, grid, ν, scalar)
    diff_flux_y_c = diffusive_flux_y(i, j + 1, k, grid, ν, scalar)
    diff_flux_z_c = diffusive_flux_z(i, j, k + 1, grid, ν, scalar)
    δxᶜᵃᵃ = (diff_flux_x - diff_flux_x_f)/grid.Δxᶜᵃᵃ
    δyᵃᶜᵃ = (diff_flux_y - diff_flux_y_c)/grid.Δyᵃᶜᵃ
    δzᵃᵃᶜ = (diff_flux_z - diff_flux_z_c)/grid.z.Δᵃᵃᶜ
    return @inbounds -(δxᶜᵃᵃ + δyᵃᶜᵃ + δzᵃᵃᶜ)
end