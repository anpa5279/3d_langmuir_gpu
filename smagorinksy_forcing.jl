using Oceananigans
using Oceananigans.Operators
using Oceananigans.Fields

using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.AbstractOperations: ∂x, ∂y, ∂z

include("scale_invariant_operators.jl")

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
    strain_rate = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    #calculating filter width
    Δ³ = Δx * Δy * Δz
    Δᶠ = cbrt(Δ³)
    #calculate the eddy viscosity
    νₑ = sqrt(2*strain_rate^2)*(C *  Δᶠ)^2
    #calcaulate subgrid stress
    τ = -2 * νₑ * strain_rate
    #calculate the subgrid forcing
    ∂τ∂x = Field(∂x(τ))
    ∂τ∂y= Field(∂y(τ))
    ∂τ∂z = Field(∂z(τ))
    return @inbounds sqrt(∂τ∂x^2 + ∂τ∂y^2 + ∂τ∂z^2)
end