using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.BuoyancyFormulations: g_Earth
#####
##### density calculation
#####

@inline function ρ_total(i, j, k, C, M, ρ_c, ρ_water)
    ρ = ρ_water
    m = 1
    for c in C
        ρ += (-ρ_water * c[i, j, k] * M[m] / ρ_c[m] + c[i, j, k] * M[m])
        m += 1
    end
    return ρ
end 

#####
##### buoyancy perturbation
#####
@inline function buoyancy_perturbation(i, j, k, grid, C, molar_masses, densities, reference_density, thermal_expansion)
    c_tracers = Base.structdiff(C, (T = nothing,))
    ρ = ρ_total(i, j, k, c_tracers, molar_masses, densities, reference_density)
    gravitational_acceleration = g_Earth * ρ / reference_density
    return @inbounds gravitational_acceleration * thermal_expansion * C.T[i, j, k]
end

@inline function densescalar(i, j, k, grid, clock, model_fields, parameters)
    molar_masses = parameters.molar_masses
    densities = parameters.densities
    reference_density = parameters.reference_density
    thermal_expansion  = parameters.thermal_expansion
    C = Base.structdiff(model_fields, (u = nothing, v = nothing, w = nothing,))
    return @inbounds -ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_perturbation, C, molar_masses, densities, reference_density, thermal_expansion)
end 
