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

@inline viscous_flux_ux(i, j, k, grid, ν, u) = - 2 * ν * Σ₁₁(i, j, k, grid, u)
@inline viscous_flux_vx(i, j, k, grid, ν, u, v) = - 2 * ν * Σ₁₂(i, j, k, grid, u, v)
@inline viscous_flux_wx(i, j, k, grid, ν, u, w) = - 2 * ν * Σ₁₃(i, j, k, grid, u, w)
@inline viscous_flux_uy(i, j, k, grid, ν, u, v) = - 2 * ν * Σ₁₂(i, j, k, grid, u, v)
@inline viscous_flux_vy(i, j, k, grid, ν, v) = - 2 * ν * Σ₂₂(i, j, k, grid, v)
@inline viscous_flux_wy(i, j, k, grid, ν, v, w) = - 2 * ν * Σ₂₃(i, j, k, grid, v, w)

# Vertical viscous fluxes for isotropic diffusivities
@inline viscous_flux_uz(i, j, k, grid, ν, u, w) = - 2 * ν * Σ₁₃(i, j, k, grid, u, w)
@inline viscous_flux_vz(i, j, k, grid, ν, v, w) = - 2 * ν * Σ₂₃(i, j, k, grid, v, w)
@inline viscous_flux_wz(i, j, k, grid, ν, w) = - 2 * ν * Σ₃₃(i, j, k, grid, w)

#diffusivity
@inline diffusive_flux_x(i, j, k, grid, C, c)= - C * ∂xᶠᶜᶜ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, C, c)= - C * ∂yᶜᶠᶜ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, C, c)= - C * ∂zᶜᶜᶠ(i, j, k, grid, c)

#these are the discrete forcing functions
@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = copy(model_fields.νₑ)
    visc_flux_ux = viscous_flux_ux(i, j, k, grid, ν)
    visc_flux_uy = viscous_flux_uy(i, j, k, grid, ν)
    visc_flux_uz = viscous_flux_uz(i, j, k, grid, ν)
    visc_flux_ux_f = viscous_flux_ux(i - 1, j, k, grid, ν)
    visc_flux_uy_c = viscous_flux_uy(i, j + 1, k, grid, ν)
    visc_flux_uz_c = viscous_flux_uz(i, j + 1, k, grid, ν)
    return @inbounds - (finite_diff(i, j, k, grid, ux_flux, "x", "c") +
                                      finite_diff(i, j, k, grid, uy_flux, "y", "f") +
                                      finite_diff(i, j, k, grid, uz_flux, "z", "f"))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = copy(model_fields.νₑ)
    
    return @inbounds 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, viscous_flux_vx, ν, u, v) +
                                      δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, viscous_flux_vy, ν, v) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, viscous_flux_vz, ν, v, w))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = copy(model_fields.νₑ)

    return @inbounds 1 / Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, viscous_flux_wx, ν, u, w) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, viscous_flux_wy, ν, v, w) +
                                      δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, viscous_flux_wz, ν, w))
end

@inline function ∇_dot_qᶜ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    scalar = model_fields.T
    ν = copy(model_fields.νₑ)

    return @inbounds 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, diffusive_flux_x, scalar, ν) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, diffusive_flux_y, scalar, ν) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, diffusive_flux_z, scalar, ν))
end