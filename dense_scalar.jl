using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition
import Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ,  ∂x_b,  ∂y_b,  ∂z_b, top_buoyancy_flux, bottom_buoyancy_flux, 
                                        with_float_type, AbstractBuoyancyFormulation

struct MyBuoyancyTracer{FT} <: AbstractBuoyancyFormulation{FT} 
    densities :: FT
    concentrations :: FT
    thermal_expansion :: FT
    haline_contraction :: FT
end

@inline grav(c_sat, density, Alk) = g_Earth * (Alk - c_sat) / density

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, Alk = C
    gravitational_acceleration = grav(b.c_sat, b.density, Alk)
    return @inbounds gravitational_acceleration * b.thermal_expansion * T[i, j, k]
end
#####
##### Buoyancy gradient components
#####
@inline function ∂x_b(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, Alk = C
    gravitational_acceleration = grav(b.c_sat, b.density, Alk)
    return gravitational_acceleration * (
           b.thermal_expansion * ∂xᶠᶜᶜ(i, j, k, grid, T))
end
@inline function ∂y_b(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, Alk = C
    gravitational_acceleration = grav(b.c_sat, b.density, Alk)
    return gravitational_acceleration * (b.thermal_expansion)
end
@inline function ∂z_b(i, j, k, grid, b::MyBuoyancyTracer, C)
    T, Alk = C
    gravitational_acceleration = grav(b.c_sat, b.density, Alk)
    return gravitational_acceleration * (b.thermal_expansion)
end

#####
##### buoyancy flux
#####
@inline    top_buoyancy_flux(i, j, grid, b::MyBuoyancyTracer, top_tracer_bcs, clock, fields) = getbc(top_tracer_bcs.T, i, j, grid, clock, fields) + getbc(top_tracer_bcs.Alk, i, j, grid, clock, fields)
@inline bottom_buoyancy_flux(i, j, grid, b::MyBuoyancyTracer, bottom_tracer_bcs, clock, fields) = getbc(bottom_tracer_bcs.T, i, j, grid, clock, fields) + getbc(bottom_tracer_bcs.Alk, i, j, grid, clock, fields)