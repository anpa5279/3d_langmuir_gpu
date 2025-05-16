using Oceananigans.Operators
using Oceananigans.Operators: Δzᶜᶜᶜ, Δyᶜᶜᶜ, Δxᶜᶜᶜ, Δxᶠᶠᶠ, Δyᶠᶠᶠ, Δzᶠᶠᶠ, Δxᶠᶠᶜ, Δyᶠᶠᶜ, Δzᶠᶠᶜ, Δxᶠᶜᶠ, Δyᶠᶜᶠ, Δzᶠᶜᶠ
using Oceananigans.AbstractOperations: ∂x, ∂y, ∂z
using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ, ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ, ℑxyz
using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃         
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃²  

const center = Center()
const face = Center()
#####
##### Base difference operators
#####

@inline δxᶜᵃᵃ(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δxᶠᵃᵃ(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δyᵃᶜᵃ(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δyᵃᶠᵃ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δzᵃᵃᶜ(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δzᵃᵃᶠ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]

#####
##### 3D differences
#####

for ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    δx = Symbol(:δx, ℓx, ℓy, ℓz)
    δy = Symbol(:δy, ℓx, ℓy, ℓz)
    δz = Symbol(:δz, ℓx, ℓy, ℓz)

    δxᵃ = Symbol(:δx, ℓx, :ᵃ, :ᵃ)
    δyᵃ = Symbol(:δy, :ᵃ, ℓy, :ᵃ)
    δzᵃ = Symbol(:δz, :ᵃ, :ᵃ, ℓz)

    @eval begin
        @inline $δx(i, j, k, grid, c) = $δxᵃ(i, j, k, grid, c)
        @inline $δy(i, j, k, grid, c) = $δyᵃ(i, j, k, grid, c)
        @inline $δz(i, j, k, grid, c) = $δzᵃ(i, j, k, grid, c)
    end
end

#####
##### First derivative operators
#####

for LX in (:ᶜ, :ᶠ, :ᵃ), LY in (:ᶜ, :ᶠ, :ᵃ), LZ in (:ᶜ, :ᶠ, :ᵃ)

    x_derivative = Symbol(:∂x, LX, LY, LZ)
    x_spacing    = Symbol(:Δx, LX, LY, LZ)
    x_difference = Symbol(:δx, LX, LY, LZ)

    y_derivative = Symbol(:∂y, LX, LY, LZ)
    y_spacing    = Symbol(:Δy, LX, LY, LZ)
    y_difference = Symbol(:δy, LX, LY, LZ)

    z_derivative = Symbol(:∂z, LX, LY, LZ)
    z_spacing    = Symbol(:Δz, LX, LY, LZ)
    z_difference = Symbol(:δz, LX, LY, LZ)

    @eval begin
        @inline $x_derivative(i, j, k, grid, c) = $x_difference(i, j, k, grid, c) / $x_spacing(i, j, k, grid)
        @inline $y_derivative(i, j, k, grid, c) = $y_difference(i, j, k, grid, c) / $y_spacing(i, j, k, grid)
        @inline $z_derivative(i, j, k, grid, c) = $z_difference(i, j, k, grid, c) / $z_spacing(i, j, k, grid)

        @inline $x_derivative(i, j, k, grid, c::Number) = zero(grid)
        @inline $y_derivative(i, j, k, grid, c::Number) = zero(grid)
        @inline $z_derivative(i, j, k, grid, c::Number) = zero(grid)

        @inline $x_derivative(i, j, k, grid, f::Function, args...) = $x_difference(i, j, k, grid, f, args...) / $x_spacing(i, j, k, grid)
        @inline $y_derivative(i, j, k, grid, f::Function, args...) = $y_difference(i, j, k, grid, f, args...) / $y_spacing(i, j, k, grid)
        @inline $z_derivative(i, j, k, grid, f::Function, args...) = $z_difference(i, j, k, grid, f, args...) / $z_spacing(i, j, k, grid)

        export $x_derivative
        export $x_difference
        export $y_derivative
        export $y_difference
        export $z_derivative
        export $z_difference
    end
end

@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

function compute_smagorinsky_visc(i, j, k, grid, velocities, C)
    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    cˢ² = C^2

    @inbounds νₑ = cˢ² * Δᶠ^2 * sqrt(2Σ²)
end

# Horizontal viscous fluxes for isotropic diffusivities
@inline ν_σᶜᶜᶜ(i, j, k, grid, ν, fields, σᶜᶜᶜ, args...) = ν * σᶜᶜᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶠᶜ(i, j, k, grid, ν, fields, σᶠᶠᶜ, args...) = ν * σᶠᶠᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶜᶠ(i, j, k, grid, ν, fields, σᶠᶜᶠ, args...) = ν * σᶠᶜᶠ(i, j, k, grid, args...)
@inline ν_σᶜᶠᶠ(i, j, k, grid, ν, fields, σᶜᶠᶠ, args...) = ν * σᶜᶠᶠ(i, j, k, grid, args...)

@inline viscous_flux_ux(i, j, k, grid, ν, fields) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, ν, fields, Σ₁₁, fields.u, fields.v, fields.w)
@inline viscous_flux_vx(i, j, k, grid, ν, fields) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, ν, fields, Σ₁₂, fields.u, fields.v, fields.w)
@inline viscous_flux_wx(i, j, k, grid, ν, fields) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, ν, fields, Σ₁₃, fields.u, fields.v, fields.w)
@inline viscous_flux_uy(i, j, k, grid, ν, fields) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, ν, fields, Σ₁₂, fields.u, fields.v, fields.w)
@inline viscous_flux_vy(i, j, k, grid, ν, fields) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, ν, fields, Σ₂₂, fields.u, fields.v, fields.w)
@inline viscous_flux_wy(i, j, k, grid, ν, fields) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, ν, fields, Σ₂₃, fields.u, fields.v, fields.w)

# Vertical viscous fluxes for isotropic diffusivities
@inline viscous_flux_uz(i, j, k, grid, ν, fields) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, ν, fields, Σ₁₃, fields.u, fields.v, fields.w)
@inline viscous_flux_vz(i, j, k, grid, ν, fields) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, ν, fields, Σ₂₃, fields.u, fields.v, fields.w)
@inline viscous_flux_wz(i, j, k, grid, ν, fields) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, ν, fields, Σ₃₃, fields.u, fields.v, fields.w)


#diffusivity
@inline diffusive_flux_x(i, j, k, grid, C, field)= - C * ∂xᶠᶜᶜ(i, j, k, grid, field)
@inline diffusive_flux_y(i, j, k, grid, C, field)= - C * ∂yᶜᶠᶜ(i, j, k, grid, field)
@inline diffusive_flux_z(i, j, k, grid, C, field)= - C * ∂zᶜᶜᶠ(i, j, k, grid, field)

#these are the discrete forcing functions
@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, C)
    ν = model_fields.νₑ[i, j, k]
    area_ccc = Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    area_ffc = Δyᶠᶠᶜ(i, j, k, grid) * Δzᶠᶠᶜ(i, j, k, grid)
    area_fcf = Δyᶠᶜᶠ(i, j, k, grid) * Δzᶠᶜᶠ(i, j, k, grid)
    visc_flux_ux = viscous_flux_ux(i, j, k, grid, ν, model_fields) * area_ccc
    visc_flux_uy = viscous_flux_uy(i, j, k, grid, ν, model_fields) * area_ffc
    visc_flux_uz = viscous_flux_uz(i, j, k, grid, ν, model_fields) * area_fcf
    return @inbounds - 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, visc_flux_ux) +
                                      δyᵃᶜᵃ(i, j, k, grid, visc_flux_uy) +
                                      δzᵃᵃᶜ(i, j, k, grid, visc_flux_uz))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, C)
    ν = model_fields.νₑ[i, j, k]
    area_ffc = Δyᶠᶠᶜ(i, j, k, grid) * Δzᶠᶠᶜ(i, j, k, grid)
    area_ccc = Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    area_cff = Δyᶜᶠᶠ(i, j, k, grid) * Δzᶜᶠᶠ(i, j, k, grid)
    visc_flux_vx = viscous_flux_vx(i, j, k, grid, ν, model_fields) * area_ffc
    visc_flux_vy = viscous_flux_vy(i, j, k, grid, ν, model_fields) * area_ccc
    visc_flux_vz = viscous_flux_vz(i, j, k, grid, ν, model_fields) * area_cff
    return @inbounds - 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, visc_flux_vx) +
                                      δyᵃᶠᵃ(i, j, k, grid, visc_flux_vy) +
                                      δzᵃᵃᶜ(i, j, k, grid, visc_flux_vz))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, C)
    ν = model_fields.νₑ[i, j, k]
    area_fcf = Δyᶠᶜᶠ(i, j, k, grid) * Δzᶠᶜᶠ(i, j, k, grid)
    area_cff = Δyᶜᶠᶠ(i, j, k, grid) * Δzᶜᶠᶠ(i, j, k, grid)
    area_ccc = Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    visc_flux_wx = viscous_flux_wx(i, j, k, grid, ν, model_fields) * area_fcf
    visc_flux_wy = viscous_flux_wy(i, j, k, grid, ν, model_fields) * area_cff
    visc_flux_wz = viscous_flux_wz(i, j, k, grid, ν, model_fields) * area_ccc
    return @inbounds - 1 / Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, visc_flux_wx) +
                                      δyᵃᶜᵃ(i, j, k, grid, visc_flux_wy) +
                                      δzᵃᵃᶠ(i, j, k, grid, visc_flux_wz))
end

@inline function ∇_dot_qᶜ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    scalar = model_fields[keys(model_fields)[4]]
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, C)
    ν = model_fields.νₑ[i, j, k]
    area_ffc = Δyᶠᶜᶜ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
    area_cfc = Δyᶜᶠᶜ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
    area_ccf = Δyᶜᶜᶠ(i, j, k, grid) * Δzᶜᶜᶠ(i, j, k, grid)
    diff_flux_x = diffusive_flux_x(i, j, k, grid, ν, scalar) * area_ffc
    diff_flux_y = diffusive_flux_y(i, j, k, grid, ν, scalar) * area_cfc
    diff_flux_z = diffusive_flux_z(i, j, k, grid, ν, scalar) * area_ccf
    return @inbounds - 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diff_flux_x) +
                                    δyᵃᶜᵃ(i, j, k, grid, diff_flux_y) +
                                    δzᵃᵃᶜ(i, j, k, grid, diff_flux_z))
end