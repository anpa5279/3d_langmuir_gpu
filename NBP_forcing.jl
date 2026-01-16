using Oceananigans.Operators: ℑzᵃᵃᶠ, ℑxzᶜᵃᶜ


using KernelAbstractions: @kernel, @index
#####
##### density calculation
#####

@inline function ρ_total(i, j, k, C, molar_mass, ρ_c, ρ_water)
    ρ = ρ_water
    m = 1
    for c in C
        @inbounds ρ += (-ρ_water * c[i, j, k] * molar_mass[m] / ρ_c[m] + c[i, j, k] * molar_mass[m])
        m += 1
    end
    return ρ
end 

#####
##### buoyancy perturbation
#####
@inline function buoyancy_perturbation(i, j, k, grid, C, molar_masses, densities, reference_density, thermal_expansion)
    g = Oceananigans.defaults.gravitational_acceleration
    c_tracers = Base.structdiff(C, (T = nothing,))
    ρ = ρ_total(i, j, k, c_tracers, molar_masses, densities, reference_density)
    gravitational_acceleration = g * (ρ / reference_density - 1.0)#because buoyancy is incorporated into the model. not double dipping
    if isnan(gravitational_acceleration)
        error("nans appearing from forcing")
    end
    return gravitational_acceleration
end

function densescalar(i, j, k, grid, clock, model_fields, parameters)
    molar_masses = parameters.molar_masses
    densities = parameters.densities
    reference_density = parameters.reference_density
    thermal_expansion  = parameters.thermal_expansion
    return @inbounds -ℑzᵃᵃᶠ(i, j, k, grid, buoyancy_perturbation, model_fields, molar_masses, densities, reference_density, thermal_expansion) #interpolation to get face values
end 
