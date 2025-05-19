using Oceananigans.Operators
using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ
using Oceananigans.Operators: Δzᶜᶜᶜ, Δyᶜᶜᶜ, Δxᶜᶜᶜ, Δxᶠᶠᶠ, Δyᶠᶠᶠ, Δzᶠᶠᶠ, Δxᶠᶠᶜ, Δyᶠᶠᶜ, Δzᶠᶠᶜ, Δxᶠᶜᶠ, Δyᶠᶜᶠ, Δzᶠᶜᶠ
include("local_op.jl")

function compute_smagorinsky_visc(i, j, k, grid, velocities, Σ, C)
    # Strain tensor dot product
    Σ²_tensor = Σ.^2
    Σ² = sum(Σ²_tensor)
    #@show Σ², Σ²_tensor
    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    cˢ² = C^2

    @inbounds νₑ = cˢ² * Δᶠ^2 * sqrt(2Σ²)
end

# Horizontal viscous fluxes for isotropic diffusivities

@inline viscous_flux_ux(i, j, k, grid, ν, Σ) = - 2 * ν * Σ[1, 1]
@inline viscous_flux_vx(i, j, k, grid, ν, Σ) = - 2 * ν * Σ[1, 2]
@inline viscous_flux_wx(i, j, k, grid, ν, Σ) = - 2 * ν * Σ[1, 3]
@inline viscous_flux_uy(i, j, k, grid, ν, Σ) = - 2 * ν * Σ[2, 1]
@inline viscous_flux_vy(i, j, k, grid, ν, Σ) = - 2 * ν * Σ[2, 2]
@inline viscous_flux_wy(i, j, k, grid, ν, Σ) = - 2 * ν * Σ[2, 3]

# Vertical viscous fluxes for isotropic diffusivities
@inline viscous_flux_uz(i, j, k, grid, ν, Σ) = - 2 * Σ[3, 1]
@inline viscous_flux_vz(i, j, k, grid, ν, Σ) = - 2 * Σ[3, 2]
@inline viscous_flux_wz(i, j, k, grid, ν, Σ) = - 2 * Σ[3, 3]

#diffusivity
@inline diffusive_flux_x(i, j, k, grid, C, field)= - C * finite_diff(i, j, k, grid, field, "x", "f")
@inline diffusive_flux_y(i, j, k, grid, C, field)= - C * finite_diff(i, j, k, grid, field, "y", "f")
@inline diffusive_flux_z(i, j, k, grid, C, field)= - C * finite_diff(i, j, k, grid, field, "z", "f")

#these are the discrete forcing functions
@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    strain_tensor = strain(i, j, k, grid, velocities)
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, strain_tensor, C)
    ν = model_fields.νₑ[i, j, k]
    #@show ν
    area_ccc = Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    area_ffc = Δyᶠᶠᶜ(i, j, k, grid) * Δzᶠᶠᶜ(i, j, k, grid)
    area_fcf = Δyᶠᶜᶠ(i, j, k, grid) * Δzᶠᶜᶠ(i, j, k, grid)
    visc_flux_ux = viscous_flux_ux(i, j, k, grid, ν, strain_tensor) * area_ccc
    visc_flux_uy = viscous_flux_uy(i, j, k, grid, ν, strain_tensor) * area_ffc
    visc_flux_uz = viscous_flux_uz(i, j, k, grid, ν, strain_tensor) * area_fcf
    visc_flux_ux_f = viscous_flux_ux(i - 1, j, k, grid, ν, strain_tensor) * area_ccc
    visc_flux_uy_c = viscous_flux_uy(i, j + 1, k, grid, ν, strain_tensor) * area_ffc
    visc_flux_uz_c = viscous_flux_uz(i, j + 1, k, grid, ν, strain_tensor) * area_fcf
    v_f_ux = finite_diff_nofield(visc_flux_ux, visc_flux_ux_f, grid.Δxᶠᵃᵃ)
    v_f_uy = finite_diff_nofield(visc_flux_uy, visc_flux_uy_c, grid.Δyᵃᶜᵃ)
    v_f_uz = finite_diff_nofield(visc_flux_uz, visc_flux_uz_c, grid.z.Δᵃᵃᶜ)
    #@show (v_f_ux + v_f_uy + v_f_uz)
    return @inbounds - 1 / Vᶠᶜᶜ(i, j, k, grid) * (v_f_ux + v_f_uy + v_f_uz)
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    strain_tensor = strain(i, j, k, grid, velocities)
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, strain_tensor, C)
    ν = model_fields.νₑ[i, j, k]
    area_ffc = Δyᶠᶠᶜ(i, j, k, grid) * Δzᶠᶠᶜ(i, j, k, grid)
    area_ccc = Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    area_cff = Δyᶜᶠᶠ(i, j, k, grid) * Δzᶜᶠᶠ(i, j, k, grid)
    visc_flux_vx = viscous_flux_vx(i, j, k, grid, ν, strain_tensor) * area_ffc
    visc_flux_vy = viscous_flux_vy(i, j, k, grid, ν, strain_tensor) * area_ccc
    visc_flux_vz = viscous_flux_vz(i, j, k, grid, ν, strain_tensor) * area_cff
    visc_flux_vx_c = viscous_flux_vx(i + 1, j, k, grid, ν, strain_tensor) * area_ffc
    visc_flux_vy_f = viscous_flux_vy(i, j - 1, k, grid, ν, strain_tensor) * area_ccc
    visc_flux_vz_c = viscous_flux_vz(i, j, k + 1, grid, ν, strain_tensor) * area_cff
    v_f_vx = finite_diff_nofield(visc_flux_vx, visc_flux_vx_c, grid.Δxᶜᵃᵃ)
    v_f_vy = finite_diff_nofield(visc_flux_vy, visc_flux_vy_f, grid.Δyᵃᶠᵃ)
    v_f_vz = finite_diff_nofield(visc_flux_vz, visc_flux_vz_c, grid.z.Δᵃᵃᶜ)
    ##@show visc_flux_vx
    return @inbounds - 1 / Vᶜᶠᶜ(i, j, k, grid) * ((v_f_vx + v_f_vy + v_f_vz))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    strain_tensor = strain(i, j, k, grid, velocities)
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, strain_tensor, C)
    ν = model_fields.νₑ[i, j, k]
    area_fcf = Δyᶠᶜᶠ(i, j, k, grid) * Δzᶠᶜᶠ(i, j, k, grid)
    area_cff = Δyᶜᶠᶠ(i, j, k, grid) * Δzᶜᶠᶠ(i, j, k, grid)
    area_ccc = Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    visc_flux_wx = viscous_flux_wx(i, j, k, grid, ν, strain_tensor) * area_fcf
    visc_flux_wy = viscous_flux_wy(i, j, k, grid, ν, strain_tensor) * area_cff
    visc_flux_wz = viscous_flux_wz(i, j, k, grid, ν, strain_tensor) * area_ccc
    visc_flux_wx_c = viscous_flux_wx(i + 1, j, k, grid, ν, strain_tensor) * area_fcf
    visc_flux_wy_c = viscous_flux_wy(i, j + 1, k, grid, ν, strain_tensor) * area_cff
    visc_flux_wz_f = viscous_flux_wz(i, j, k - 1, grid, ν, strain_tensor) * area_ccc
    v_f_wx = finite_diff_nofield(visc_flux_wx, visc_flux_wx_c, grid.Δxᶜᵃᵃ)
    v_f_wy = finite_diff_nofield(visc_flux_wy, visc_flux_wy_c, grid.Δyᵃᶜᵃ)
    v_f_wz = finite_diff_nofield(visc_flux_wz, visc_flux_wz_f, grid.z.Δᵃᵃᶠ)
    ##@show visc_flux_wz
    return @inbounds - 1 / Vᶜᶜᶠ(i, j, k, grid) * (v_f_wx + v_f_wy + v_f_wz)
end

@inline function ∇_dot_qᶜ(i, j, k, grid, clock, model_fields, C)
    velocities = model_fields[keys(model_fields)[1:3]]
    scalar = model_fields[keys(model_fields)[4]]
    strain_tensor = strain(i, j, k, grid, velocities)
    model_fields.νₑ[i, j, k] = compute_smagorinsky_visc(i, j, k, grid, velocities, strain_tensor, C)
    ν = model_fields.νₑ[i, j, k]
    area_ffc = Δyᶠᶜᶜ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
    area_cfc = Δyᶜᶠᶜ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
    area_ccf = Δyᶜᶜᶠ(i, j, k, grid) * Δzᶜᶜᶠ(i, j, k, grid)
    diff_flux_x = diffusive_flux_x(i, j, k, grid, ν, scalar) * area_ffc
    diff_flux_y = diffusive_flux_y(i, j, k, grid, ν, scalar) * area_cfc
    diff_flux_z = diffusive_flux_z(i, j, k, grid, ν, scalar) * area_ccf
    diff_flux_x_c = diffusive_flux_x(i + 1, j, k, grid, ν, scalar) * area_ffc
    diff_flux_y_c = diffusive_flux_y(i, j + 1, k, grid, ν, scalar) * area_cfc
    diff_flux_z_c = diffusive_flux_z(i, j, k + 1, grid, ν, scalar) * area_ccf
    d_f_x = finite_diff_nofield(diff_flux_x, diff_flux_x_c, grid.Δxᶜᵃᵃ)
    d_f_y = finite_diff_nofield(diff_flux_y, diff_flux_y_c, grid.Δyᵃᶜᵃ)
    d_f_z = finite_diff_nofield(diff_flux_z, diff_flux_z_c, grid.z.Δᵃᵃᶜ)
    ##@show diff_flux_z
    return @inbounds - 1/Vᶜᶜᶜ(i, j, k, grid) * (d_f_x + d_f_y + d_f_z)
end