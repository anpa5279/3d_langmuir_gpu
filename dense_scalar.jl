using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition, getbc
using Oceananigans.Operators
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
import Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ,  ∂x_b,  ∂y_b,  ∂z_b, top_buoyancy_flux, bottom_buoyancy_flux, 
                                        with_float_type, AbstractBuoyancyFormulation, required_tracers
using Oceananigans.Utils: tupleit
import SeawaterPolynomials: thermal_expansion, haline_contraction, with_float_type
struct TracerConcentrationBuoyancy{FT} <: AbstractBuoyancyFormulation{FT} 
    densities :: P
    molar_masses :: P
    reference_density :: FT
    thermal_expansion :: FT
    haline_contraction :: FT
    constant_temperature :: T
    constant_salinity :: S
end

function  TracerConcentrationBuoyancy(FT = Oceananigans.defaults.FloatType;
                          densities = (), 
                          molar_masses = (),
                          reference_density = 1026.0, # kg m⁻³, average density at the surface of the world ocean
                          thermal_expansion = 2.0e-4, # 1/K, thermal
                          haline_contraction = 7.6e-4, # 1/ppt, haline contraction coefficient
                          constant_temperature = nothing,
                          constant_salinity = nothing)

    densities = tupleit(densities)
    molar_masses = tupleit(molar_masses)
    constant_temperature = constant_temperature === true ? zero(FT) : constant_temperature
    constant_salinity = constant_salinity === true ? zero(FT) : constant_salinity

    constant_temperature = isnothing(constant_temperature) ? nothing : convert(FT, constant_temperature)
    constant_salinity = isnothing(constant_salinity) ? nothing : convert(FT, constant_salinity)

    return  TracerConcentrationBuoyancy{FT, typeof(densities), typeof(molar_masses), typeof(reference_density), typeof(thermal_expansion), typeof(haline_contraction), typeof(constant_temperature), typeof(constant_salinity)}(
                            densities, molar_masses, reference_density, thermal_expansion, haline_contraction, constant_temperature, constant_salinity)
end

required_tracers(::TracerConcentrationBuoyancy) = (:T, :S)
required_tracers(::TracerConcentrationBuoyancy{FT, <:Nothing, <:Number}) where {FT} = (:T,) # active temperature only
required_tracers(::TracerConcentrationBuoyancy{FT, <:Number, <:Nothing}) where {FT} = (:S,) # active salinity only

#####
##### Convinient aliases to dispatch on
#####

const ConcentrationBuoyancy = TracerConcentrationBuoyancy{FT} where FT
const TemperatureConcentrationBuoyancy = TracerConcentrationBuoyancy{FT, <:Nothing, <:Number} where FT
const SalinityConcentrationBuoyancy = TracerConcentrationBuoyancy{FT, <:Number, <:Nothing} where FT

#####
##### density calculation
#####

@inline function ρ_total(C, M, ρ_c, ρ_water)#must ignore salinity and temperature 
    ρ = ρ_water
    for c in C 
        ρ += (-ρ_water * C[c][i, j, k]/(M[c] * ρ_c[c]) + C[c]/M[c])
    end
    return @inbounds ρ
end 

#####
##### buoyancy perturbation
#####

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::ConcentrationBuoyancy, C)
    ρ = ρ_total(C, b.molar_masses, b.densities, b.reference_density)
    gravitational_acceleration = g_Earth * ρ / b.reference_density
    return @inbounds gravitational_acceleration * (b.thermal_expansion * C.T[i, j, k] -
                                              b.haline_contraction * C.S[i, j, k])
end

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::TemperatureConcentrationBuoyancy, C)
    ρ = ρ_total(C, b.molar_masses, b.densities, b.reference_density)
    gravitational_acceleration = g_Earth * ρ / b.reference_density
    return @inbounds gravitational_acceleration * b.thermal_expansion * C.T[i, j, k]
end

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::SalinityConcentrationBuoyancy, C)
    ρ = ρ_total(C, b.molar_masses, b.densities, b.reference_density)
    gravitational_acceleration = g_Earth * ρ / b.reference_density
    return @inbounds - gravitational_acceleration * b.haline_contraction * C.S[i, j, k]
end

#####
##### Buoyancy gradient components
#####
@inline function ∂x_b(i, j, k, grid, b::TracerConcentrationBuoyancy, C)
    ρ = ρ_total(C, b.molar_masses, b.densities, b.reference_density)
    gravitational_acceleration = g_Earth * ρ / b.reference_density
    return gravitational_acceleration * (
           b.thermal_expansion * ∂xᶠᶜᶜ(i, j, k, grid, T)
        - b.haline_contraction * ∂xᶠᶜᶜ(i, j, k, grid, S) )
end
@inline function ∂y_b(i, j, k, grid, b::TracerConcentrationBuoyancy, C)
    ρ = ρ_total(C, b.molar_masses, b.densities, b.reference_density)
    gravitational_acceleration = g_Earth * ρ / b.reference_density
    return gravitational_acceleration * (
           b.thermal_expansion * ∂yᶜᶠᶜ(i, j, k, grid, T)
        - b.haline_contraction * ∂yᶜᶠᶜ(i, j, k, grid, S))
end
@inline function ∂z_b(i, j, k, grid, b::TracerConcentrationBuoyancy, C)
    ρ = ρ_total(C, b.molar_masses, b.densities, b.reference_density)
    gravitational_acceleration = g_Earth * ρ / b.reference_density
    return gravitational_acceleration * (
           thermal_expansion * ∂zᶜᶜᶠ(i, j, k, grid, T)
        - haline_contraction * ∂zᶜᶜᶠ(i, j, k, grid, S))
end

#####
##### buoyancy flux
#####

@inline get_temperature_and_salinity_flux(::TracerConcentrationBuoyancy, bcs) = bcs.T, bcs.S
@inline get_temperature_and_salinity_flux(::TemperatureConcentrationBuoyancy, bcs) = bcs.T, NoFluxBoundaryCondition()
@inline get_temperature_and_salinity_flux(::SalinityConcentrationBuoyancy, bcs) = NoFluxBoundaryCondition(), bcs.S

@inline function top_bottom_buoyancy_flux(i, j, k, grid, b::TracerConcentrationBuoyancy, top_bottom_tracer_bcs, clock, fields)
    T, S = get_temperature_and_salinity(b, fields)
    T_flux_bc, S_flux_bc = get_temperature_and_salinity_flux(b, top_bottom_tracer_bcs)

    T_flux = getbc(T_flux_bc, i, j, grid, clock, fields)
    S_flux = getbc(S_flux_bc, i, j, grid, clock, fields)

    return b.gravitational_acceleration * (
              thermal_expansion * T_flux
           - haline_contraction * S_flux)
end
@inline    top_buoyancy_flux(i, j, grid, b::TracerConcentrationBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, grid.Nz+1, grid, b, args...)
@inline bottom_buoyancy_flux(i, j, grid, b::TracerConcentrationBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, 1, grid, b, args...)