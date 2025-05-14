using Oceananigans
using Oceananigans.Operators
using Oceananigans.Fields

using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index
import Oceananigans.Operators: ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑyᶜᵃᵃ, ℑzᵃᶜᶜ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
import Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂zᶜᶜᶜ

using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃         
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃² 

function stress(i, j, k, grid, u, v, w, C)
    Δx = grid.Δxᶜᵃᵃ
    Δy = grid.Δyᵃᶜᵃ
    Δz = grid.z.Δᵃᵃᶜ
    #calcualte the resolved strain rate 0.5*(u_i,j+u_j,i)
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    #calculating filter width
    Δ³ = Δx * Δy * Δz
    Δᶠ = cbrt(Δ³)
    #calculate the eddy viscosity
    νₑ[i, j, k] = sqrt(2*Σ²)*(C *  Δᶠ)^2
    #calcaulate subgrid stress
    return @inbounds τ = -2 * νₑ[i, j, k] * Σ²
end

function smag_u(i, j, k, grid, clock, model_fields, C)
    u = model_fields.u
    v = model_fields.v
    w = model_fields.w
    τ1 = stress(i, j, k, grid, u, v, w, C)
    ∂1_τ11 = ℑxᶠᵃᵃ(i, j, k, grid, ∂xᶜᶜᶜ, τ1)
    ∂2_τ12 = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶜᶜ, τ1)
    ∂3_τ13 = ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶜᶜᶜ, τ1)
    return @inbounds ∂j_τ1j = - ∂1_τ11 - ∂2_τ12 - ∂3_τ13
end
function smag_v(i, j, k, grid, clock, model_fields, C)
    u = model_fields.u
    v = model_fields.v
    w = model_fields.w
    τ2 = stress(i, j, k, grid, clock, model_fields, C)
    ∂1_τ21 = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶜᶜᶜ, τ2)
    ∂2_τ22 = ℑyᵃᶠᵃ(i, j, k, grid, ∂yᶜᶜᶜ, τ2)
    ∂3_τ23 = ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶜᶜᶜ, τ2)
    return @inbounds ∂j_τ2j = - ∂1_τ21 - ∂2_τ22 - ∂3_τ23
end
function smag_w(i, j, k, grid, clock, model_fields, C)
    u = model_fields.u
    v = model_fields.v
    w = model_fields.w
    τ3 = stress(i, j, k, grid, clock, model_fields, C)
    ∂1_τ31 = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶜᶜᶜ, τ3)
    ∂2_τ32 = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶜᶜ, τ3)
    ∂3_τ33 = ℑzᵃᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, τ3)
    return @inbounds ∂j_τ3j = - ∂1_τ31 - ∂2_τ32 - ∂3_τ33
end
function smag_c(i, j, k, grid, clock, model_fields, C)
    u = model_fields.u
    v = model_fields.v
    w = model_fields.w
    τ3 = stress(i, j, k, grid, clock, model_fields, C)
    ∂1_τ31 = ℑxᶜᵃᵃ(i, j, k, grid, ∂x, )
    ∂2_τ32 = ℑyᵃᶜᵃ(i, j, k, grid, ∂y, )
    ∂3_τ33 = ℑzᵃᵃᶜ(i, j, k, grid, ∂z, )
    return @inbounds ∇_dot_qᶜ = - ∂1_τ31 - ∂2_τ32 - ∂3_τ33
end
@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

