using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition
import Oceananigans.BuoyancyFormulations: required_tracers, buoyancy_perturbationᶜᶜᶜ,  ∂x_b,  ∂y_b,  ∂z_b, top_buoyancy_flux, bottom_buoyancy_flux, 
                                        with_float_type, AbstractBuoyancyFormulation

struct MyBuoyancyTracer{FT, EOS, T, S} <: AbstractBuoyancyFormulation{EOS}
    equation_of_state :: EOS
    density :: FT
    c_sat :: FT
    constant_temperature :: T
    constant_salinity :: S
end

required_tracers(::MyBuoyancyTracer) = (:T, :S, :Alk)
required_tracers(::MyBuoyancyTracer{FT, EOS, <:Nothing, <:Number}) where {FT, EOS} = (:T, :Alk) # active temperature and alkalinity only
required_tracers(::MyBuoyancyTracer{FT, EOS, <:Number, <:Nothing}) where {FT, EOS} = (:S, :Alk) # active salinity and alkalinity only

function MyBuoyancyTracer(FT = Oceananigans.defaults.FloatType;
                          density = 1026.0, # kg m⁻³, average density at the surface of the world ocean
                          c_sat = 2300.0, # umol/kg, saturation concentration of dissolved inorganic carbon
                          equation_of_state = LinearEquationOfState(FT),
                          constant_temperature = nothing,
                          constant_salinity = nothing)
    constant_temperature = constant_temperature === true ? zero(FT) : constant_temperature
    constant_salinity = constant_salinity === true ? zero(FT) : constant_salinity
    equation_of_state = with_float_type(FT, equation_of_state)

    constant_temperature = isnothing(constant_temperature) ? nothing : convert(FT, constant_temperature)
    constant_salinity = isnothing(constant_salinity) ? nothing : convert(FT, constant_salinity)

    return MyBuoyancyTracer{FT, typeof(equation_of_state), typeof(constant_temperature), typeof(constant_salinity)}(
                            equation_of_state, density, c_sat, constant_temperature, constant_salinity)
end

const TemperatureMyBuoyancyTracer = MyBuoyancyTracer{FT, EOS, <:Nothing, <:Number} where {FT, EOS}
const SalinityMyBuoyancyTracer = MyBuoyancyTracer{FT, EOS, <:Number, <:Nothing} where {FT, EOS}
Base.nameof(::Type{TemperatureMyBuoyancyTracer}) = "TemperatureMyBuoyancyTracer"
Base.nameof(::Type{SalinityMyBuoyancyTracer}) = "SalinityMyBuoyancyTracer"

@inline get_temperature_and_salinity(::MyBuoyancyTracer, C) = C.T, C.S, C.Alk
@inline get_temperature_and_salinity(b::TemperatureMyBuoyancyTracer, C) = C.T, b.constant_salinity, C.Alk
@inline get_temperature_and_salinity(b::SalinityMyBuoyancyTracer, C) = b.constant_temperature, C.S, C.Alk

@inline grav(b, C) = g_Earth * (C.Alk - b.c_sat) / b.density

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::MyBuoyancyTracer, C)
    gravitational_acceleration = grav(b, C)
    return gravitational_acceleration * (b.equation_of_state.thermal_expansion * C.T[i, j, k] -
                                              b.equation_of_state.haline_contraction * C.S[i, j, k])
end 
@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::TemperatureMyBuoyancyTracer, C)
    gravitational_acceleration = grav(b, C)
    return gravitational_acceleration * b.equation_of_state.thermal_expansion * C.T[i, j, k]
end 
@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::SalinityMyBuoyancyTracer, C)
    gravitational_acceleration = grav(b, C)
    return - gravitational_acceleration * b.equation_of_state.haline_contraction * C.S[i, j, k]
end 
#####
##### Buoyancy gradient components
#####
@inline function ∂x_b(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, S, Alk = get_temperature_and_salinity(b, C)
    gravitational_acceleration = grav(b, C)
    return gravitational_acceleration * (
           thermal_expansionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂xᶠᶜᶜ(i, j, k, grid, T)
        - haline_contractionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂xᶠᶜᶜ(i, j, k, grid, S) )
end
@inline function ∂y_b(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, S, Alk = get_temperature_and_salinity(b, C)
    gravitational_acceleration = grav(b, C)
    return gravitational_acceleration * (
           thermal_expansionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂yᶜᶠᶜ(i, j, k, grid, T)
        - haline_contractionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂yᶜᶠᶜ(i, j, k, grid, S) )
end
@inline function ∂z_b(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, S, Alk = get_temperature_and_salinity(b, C)
    gravitational_acceleration = grav(b, C)
    return gravitational_acceleration * (
           thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * ∂zᶜᶜᶠ(i, j, k, grid, T)
        - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * ∂zᶜᶜᶠ(i, j, k, grid, S) )
end

#####
##### top buoyancy flux
#####

@inline get_temperature_and_salinity_flux(::MyBuoyancyTracer, bcs) = bcs.T, bcs.S, bcs.Alk
@inline get_temperature_and_salinity_flux(::TemperatureMyBuoyancyTracer, bcs) = bcs.T, NoFluxBoundaryCondition(), bcs.Alk
@inline get_temperature_and_salinity_flux(::SalinityMyBuoyancyTracer, bcs) = NoFluxBoundaryCondition(), bcs.S, bcs.Alk

@inline function top_bottom_buoyancy_flux(i, j, k, grid, b::MyBuoyancyTracer, top_bottom_tracer_bcs, clock, fields)
    T, S, Alk = get_temperature_and_salinity(b, C)
    T_flux_bc, S_flux_bc, Alk_flux_bc = get_temperature_and_salinity_flux(b, top_bottom_tracer_bcs)

    T_flux = getbc(T_flux_bc, i, j, grid, clock, fields)
    S_flux = getbc(S_flux_bc, i, j, grid, clock, fields)

    gravitational_acceleration = grav(b, C)
    return gravitational_acceleration * (
              thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * T_flux
           - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * S_flux)
end

@inline    top_buoyancy_flux(i, j, grid, b::MyBuoyancyTracer, args...) = top_bottom_buoyancy_flux(i, j, grid.Nz+1, grid, b, args...)
@inline bottom_buoyancy_flux(i, j, grid, b::MyBuoyancyTracer, args...) = top_bottom_buoyancy_flux(i, j, 1, grid, b, args...)