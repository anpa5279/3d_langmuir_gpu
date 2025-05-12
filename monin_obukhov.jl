using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: FieldBoundaryConditions
using Oceananigans.Utils: launch!, IterationInterval

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

#####
##### The turbulence closure proposed by Sullivan 1994.
#####

struct SmagorinskyMoninObukhov{TD, C, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 2}
    coefficient :: C
    Pr :: P

    function Smagorinsky{TD}(coefficient, Pr) where TD
        P = typeof(Pr)
        C = typeof(coefficient)
        return new{TD, C, P}(coefficient, Pr)
    end
end

@inline viscosity(::SmagorinskyMoninObukhov, K) = K.νₑ
@inline diffusivity(closure::SmagorinskyMoninObukhov, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

const ConstantMoninObukhov = SmagorinskyMoninObukhov{<:Any, <:Number}

""
function SmagorinskyMoninObukhov(time_discretization::TD = ExplicitTimeDiscretization(), FT=Oceananigans.defaults.FloatType;
                     coefficient = 0.1, Pr = 1.0) where TD
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return SmagorinskyMoninObukhov{TD}(coefficient, Pr)
end

SmagorinskyMoninObukhov(FT::DataType; kwargs...) = SmagorinskyMoninObukhov(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::SmagorinskyMoninObukhov{TD}) where TD
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return Smagorinsky{TD}(closure.coefficient, Pr)
end

@kernel function _compute_moninobukhov_viscosity!(diffusivity_fields, grid, closure, buoyancy, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    cˢ² = square_smagorinskymoninobukhov_coefficient(i, j, k, grid, closure, diffusivity_fields, Σ², buoyancy, tracers)

    νₑ = diffusivity_fields.νₑ

    @inbounds νₑ[i, j, k] = cˢ² * Δᶠ^2 * sqrt(2Σ²)
end

@inline function square_smagorinskymoninobukhov_coefficient(i, j, k, grid, closure::SmagorinskyMoninObukhov,
                                                            diffusivity_fields, Σ²)
    i, j, k = @index(Global, NTuple)
    # smagorinsky coefficient
    c₀ = closure.coefficient
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

compute_coefficient_fields!(diffusivity_fields, closure, model; parameters) = nothing

function compute_diffusivities!(diffusivity_fields, closure::SmagorinskyMoninObukhov, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    buoyancy = model.buoyancy
    velocities = model.velocities
    tracers = model.tracers

    compute_coefficient_fields!(diffusivity_fields, closure, model; parameters)

    launch!(arch, grid, parameters, _compute_smagorinsky_viscosity!,
            diffusivity_fields, grid, closure, buoyancy, velocities, tracers)

    return nothing
end

allocate_coefficient_fields(closure, grid) = NamedTuple()

function build_diffusivity_fields(grid, clock, tracer_names, bcs, closure::SmagorinskyMoninObukhov)
    coefficient_fields = allocate_coefficient_fields(closure, grid)

    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(grid, boundary_conditions=bcs.νₑ)
    viscosity_nt = (; νₑ)

    return merge(viscosity_nt, coefficient_fields)
end

@inline κᶠᶜᶜ(i, j, k, grid, c::SmagorinskyMoninObukhov, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / c.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, c::SmagorinskyMoninObukhov, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / c.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, c::SmagorinskyMoninObukhov, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / c.Pr[id]

Base.summary(closure::SmagorinskyMoninObukhov) = string("Smagorinsky with coefficient = ", summary(closure.coefficient), ", Pr=$(closure.Pr)")
function Base.show(io::IO, closure::SmagorinskyMoninObukhov)
    coefficient_summary = closure.coefficient isa Number ? closure.coefficient : summary(closure.coefficient)
    print(io, "Smagorinsky Monin-Obukhov closure with\n",
              "├── coefficient = ", coefficient_summary, "\n",
              "└── Pr = ", closure.Pr)
end

