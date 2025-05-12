using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: FieldBoundaryConditions
using Oceananigans.Utils: launch!, IterationInterval
using Oceananigans.TurbulenceClosures: tr_Σ, Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃

using Adapt

using ..TurbulenceClosures:
    AbstractScalarDiffusivity,
    ThreeDimensionalFormulation,
    ExplicitTimeDiscretization,
    convert_diffusivity

import Oceananigans.Utils: with_tracers

import ..TurbulenceClosures:
    viscosity,
    diffusivity,
    κᶠᶜᶜ,
    κᶜᶠᶜ,
    κᶜᶜᶠ,
    compute_diffusivities!,
    build_diffusivity_fields,
    tracer_diffusivities

struct MoninObukhovCoefficient{FT}
    smagorinsky :: FT
    reduction_factor :: FT
end

MoninObukhovCoefficient(FT=Oceananigans.defaults.FloatType; smagorinsky=0.16, reduction_factor=1) =
    MoninObukhovCoefficient(convert(FT, smagorinsky), convert(FT, reduction_factor))

const MoninObukhov = Smagorinsky{<:Any, <:MoninObukhovCoefficient}

function MoninObukhov(time_discretization=ExplicitTimeDiscretization(), FT=Oceananigans.defaults.FloatType; C=0.1, Pr=1)
    coefficient = MoninObukhovCoefficient(FT, smagorinsky=C)
    TD = typeof(time_discretization)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return Smagorinsky{TD}(coefficient, Pr)
end

MoninObukhov(FT::DataType; kwargs...) = MoninObukhov(ExplicitTimeDiscretization(), FT; kwargs...)


@inline function square_smagorinsky_coefficient(i, j, k, grid, closure::MoninObukhov,
                                                diffusivity_fields, Σ²)
    i, j, k = @index(Global, NTuple)
    # smagorinsky coefficient
    c₀ = closure.coefficient.smagorinsky
    # average of strain rate tensor
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)
    S_bar = sqrt(2Σ²)
    # prime of strain rate tensor
    prime_Σ² = prime_Σᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w) 
                * prime_Σᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)
    S_prime = sqrt(2 * prime_Σ²)
    γ = S_prime / (S_bar + S_prime)
    # calculating isotropy factor
    return (sqrt(γ) * c₀)^2
end

"average strain rate"
@inline Σᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =    tr_Σ(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂, u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃, u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃, u, v, w)
"fluctuation strain rate tensor"
@inline prime_Σᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ(i, j, k, grid, u, v, w) +
                                                    2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂ - Σᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w), u, v, w) +
                                                    2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃ - Σᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w), u, v, w) +
                                                    2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃ - Σᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w), u, v, w)


Base.summary(dc::MoninObukhovCoefficient) = string("MoninObukhovCoefficient(smagorinsky = $(dc.smagorinsky), reduction_factor = $(dc.reduction_factor))")
Base.show(io::IO, dc::MoninObukhovCoefficient) = print(io, "MoninObukhovCoefficient with\n",
                                                    "├── Smagorinsky coefficient = ", dc.smagorinsky, "\n",
                                                    "└── reduction_factor = ", dc.reduction_factor)


