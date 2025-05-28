using KernelAbstractions: @kernel, @index

using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃         
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃² 
using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ
using Oceananigans.Operators: volume
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Utils: launch!

# strain tensor squared and summed 
@inline function ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    s = tr_Σ²(i, j, k, grid, u, v, w)
    s .+ 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w)
    s .+ 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w)
    s .+ 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)
    return s
end

# viscosity
@kernel function _smagorinsky_visc!(grid, velocities, νₑ)
    i, j, k = @index(Global, NTuple)

    u = velocities.u
    v = velocities.v
    w = velocities.w
    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    C = 0.1
    cˢ² = C^2

    @inbounds νₑ[i, j, k] = cˢ² * Δᶠ^2 * sqrt(2Σ²)
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
@inline function diffusive_flux_x(i, j, k, grid, ν, c)
    return @inbounds - ν[i, j, k]  * ∂xᶠᶜᶜ(i, j, k, grid, c)
end 
@inline function diffusive_flux_y(i, j, k, grid, ν, c)
    return @inbounds - ν[i, j, k]  * ∂yᶜᶠᶜ(i, j, k, grid, c)
end
@inline function diffusive_flux_z(i, j, k, grid, ν, c)
    return @inbounds - ν[i, j, k]  * ∂zᶜᶜᶠ(i, j, k, grid, c)
end

#these are the discrete forcing functions
@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    ν = model_fields.νₑ
    return -1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, viscous_flux_ux, ν, u) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, viscous_flux_uy, ν, u, v) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, viscous_flux_uz, ν, u, w))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, model_fields)
    ν = model_fields.νₑ
    launch!(arch, grid, :xyz, _smagorinsky_visc!, grid, velocities, ν)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    return -1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, viscous_flux_vx, ν, u, v) +
                                      δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, viscous_flux_vy, ν, v) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, viscous_flux_vz, ν, v, w))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, clock, model_fields)
    u = model_fields.u 
    v = model_fields.v
    w = model_fields.w
    return -1 / Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, viscous_flux_wx, ν, u, w) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, viscous_flux_wy, ν, v, w) +
                                      δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, viscous_flux_wz, ν, w))
end

@inline function ∇_dot_qᶜ(i, j, k, grid, clock, model_fields)
    ν = model_fields.νₑ
    launch!(arch, grid, :xyz, _smagorinsky_visc!, grid, velocities, ν)
    scalar = model_fields.T
    return -1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, diffusive_flux_x, ν, scalar) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, diffusive_flux_y, ν, scalar) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, diffusive_flux_z, ν, scalar))
end 
