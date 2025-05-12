using Oceananigans
using Oceananigans.Operators
using Oceananigans.Fields

using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.AbstractOperations: ∂x, ∂y, ∂z

@kernel function smagorinksy_forcing!(i, j, k, grid, clock, model_fields, p)
    C = p.C
    grid = w.grid
    Δyᶜᶜᶜ = grid.Δyᶜᶜᶜ
    Δxᶜᶜᶜ = grid.Δxᶜᶜᶜ
    Δzᶜᶜᶜ = grid.Δzᶜᶜᶜ
    Δx = Δxᶜᶜᶜ(i, j, k, grid)
    Δy = Δyᶜᶜᶜ(i, j, k, grid)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    u = model_fields.u
    v = model_fields.v
    w = model_fields.w
    #calcualte the resolved strain rate 0.5*(u_i,j+u_j,i)
    strain_rate2 = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    #calculating filter width
    Δ³ = Δx * Δy * Δz
    Δᶠ = cbrt(Δ³)
    #calculate the eddy viscosity
    νₑ = sqrt(2*strain_rate2)*(C *  Δᶠ)^2
    #calcaulate subgrid stress
    τ = -2 * νₑ * strain_rate2
    #calculate the subgrid forcing
    ∂τ∂x = Field(∂x(τ))
    ∂τ∂y= Field(∂y(τ))
    ∂τ∂z = Field(∂z(τ))
    @inbounds sqrt(∂τ∂x^2 + ∂τ∂y^2 + ∂τ∂z^2)
end

@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)


@inline tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2

@inline Σ₁₂²(i, j, k, grid, u, v, w) = Σ₁₂²(i, j, k, grid, u, v)
@inline Σ₁₃²(i, j, k, grid, u, v, w) = Σ₁₃²(i, j, k, grid, u, w)
@inline Σ₂₃²(i, j, k, grid, u, v, w) = Σ₂₃²(i, j, k, grid, v, w)

@inline Σ₁₁(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, u)
@inline Σ₂₂(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, v)
@inline Σ₃₃(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, w)


