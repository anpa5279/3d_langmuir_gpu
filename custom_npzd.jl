module NPZDcustom

export NPZDc

using OceanBioME: Biogeochemistry, ScaleNegativeTracers
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry

using Oceananigans.Units
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: AbstractGrid

using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation, default_surface_PAR
using OceanBioME: setup_velocity_fields, show_sinking_velocities
using OceanBioME.BoxModels: BoxModel

import Base: show, summary

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers,
                                     required_biogeochemical_auxiliary_fields,
                                     biogeochemical_drift_velocity

import OceanBioME: redfield, conserved_tracers, maximum_sinking_velocity, chlorophyll

import Adapt: adapt_structure, adapt

struct NPZDc{FT, W} <: AbstractContinuousFormBiogeochemistry
    # phytoplankton
    initial_photosynthetic_slope :: FT # α, 1/(W/m²)/s
    base_maximum_growth :: FT # μ₀, 1/s
    nutrient_half_saturation :: FT # kₙ, mmol N/m³
    base_respiration_rate :: FT # lᵖⁿ, 1/s
    phyto_base_mortality_rate :: FT # lᵖᵈ, 1/s

    # zooplankton
    maximum_grazing_rate :: FT # gₘₐₓ, 1/s
    grazing_half_saturation :: FT # kₚ, mmol N/m³
    assimulation_efficiency :: FT # β
    base_excretion_rate :: FT # lᶻⁿ, 1/s
    zoo_base_mortality_rate :: FT # lᶻᵈ, 1/s

    # detritus
    remineralization_rate :: FT # rᵈⁿ, 1/s

    # sinking
    sinking_velocities :: W

    function NPZDc(initial_photosynthetic_slope::FT,
                                                      base_maximum_growth::FT,
                                                      nutrient_half_saturation::FT,
                                                      base_respiration_rate::FT,
                                                      phyto_base_mortality_rate::FT,
    
                                                      maximum_grazing_rate::FT,
                                                      grazing_half_saturation::FT,
                                                      assimulation_efficiency::FT,
                                                      base_excretion_rate::FT,
                                                      zoo_base_mortality_rate::FT,
    
                                                      remineralization_rate::FT,
    
                                                      sinking_velocities::W) where {FT, W}
        return new{FT, W}(initial_photosynthetic_slope,
                          base_maximum_growth,
                          nutrient_half_saturation,
                          base_respiration_rate,
                          phyto_base_mortality_rate,

                          maximum_grazing_rate,
                          grazing_half_saturation,
                          assimulation_efficiency,
                          base_excretion_rate,
                          zoo_base_mortality_rate,

                          remineralization_rate,
                          
                          sinking_velocities)
    end
end

function NPZDc(; grid::AbstractGrid{FT},
                                                    initial_photosynthetic_slope::FT = 0.1953 / day, # 1/(W/m²)/s
                                                    base_maximum_growth::FT = 0.6989 / day, # 1/s
                                                    nutrient_half_saturation::FT = 2.3868, # mmol N/m³
                                                    base_respiration_rate::FT = 0.066 / day, # 1/s/(mmol N / m³)
                                                    phyto_base_mortality_rate::FT = 0.0101 / day, # 1/s/(mmol N / m³)
                                                    maximum_grazing_rate::FT = 2.1522 / day, # 1/s
                                                    grazing_half_saturation::FT = 0.5573, # mmol N/m³
                                                    assimulation_efficiency::FT = 0.9116, 
                                                    base_excretion_rate::FT = 0.0102 / day, # 1/s/(mmol N / m³)
                                                    zoo_base_mortality_rate::FT = 0.3395 / day, # 1/s/(mmol N / m³)²
                                                    remineralization_rate::FT = 0.1213 / day, # 1/s

                                                    surface_photosynthetically_active_radiation = default_surface_PAR,
                                                    light_attenuation_model::LA =
                                                        TwoBandPhotosyntheticallyActiveRadiation(; grid,
                                                                                                   surface_PAR = surface_photosynthetically_active_radiation),
                                                    sediment_model::S = nothing,
                
                                                    sinking_speeds = (P = 0.2551/day, D = 2.7489/day),
                                                    open_bottom::Bool = true,

                                                    scale_negatives = false,
                                                    invalid_fill_value = NaN,
                                                                                           
                                                    particles::P = nothing,
                                                    modifiers::M = nothing) where {FT, LA, S, P, M}

    sinking_velocities = setup_velocity_fields(sinking_speeds, grid, open_bottom)

    underlying_biogeochemistry = 
        NPZDc(initial_photosynthetic_slope,
                                                 base_maximum_growth,
                                                 nutrient_half_saturation,
                                                 base_respiration_rate,
                                                 phyto_base_mortality_rate,
                                                 maximum_grazing_rate,
                                                 grazing_half_saturation,
                                                 assimulation_efficiency,
                                                 base_excretion_rate,
                                                 zoo_base_mortality_rate,
                                                 remineralization_rate,
                                                 sinking_velocities)

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid; invalid_fill_value)
        modifiers = isnothing(modifiers) ? scaler : (modifiers..., scaler)
    end

    return Biogeochemistry(underlying_biogeochemistry;
                           light_attenuation = light_attenuation_model, 
                           sediment = sediment_model, 
                           particles,
                           modifiers)
end

required_biogeochemical_tracers(::NPZDc) = (:N, :P, :Z, :D, :T)
required_biogeochemical_auxiliary_fields(::NPZDc) = (:PAR, )

@inline nutrient_limitation(N, kₙ) = N / (kₙ + N)

@inline Q₁₀(T) = 1.88 ^ (T / 10) # T in °C

@inline light_limitation(PAR, α, μ₀) = α * PAR / sqrt(μ₀ ^ 2 + α ^ 2 * PAR ^ 2)

@inline function (bgc::NPZDc)(::Val{:N}, x, y, z, t, N, P, Z, D, T, PAR)
    μ₀ = bgc.base_maximum_growth
    kₙ = bgc.nutrient_half_saturation
    α = bgc.initial_photosynthetic_slope
    lᵖⁿ = bgc.base_respiration_rate
    lᶻⁿ = bgc.base_excretion_rate
    rᵈⁿ = bgc.remineralization_rate

    phytoplankton_consumption = μ₀ * Q₁₀(T) * nutrient_limitation(N, kₙ) * light_limitation(PAR, α, μ₀ * Q₁₀(T)) * P
    phytoplankton_metabolic_loss = lᵖⁿ * Q₁₀(T) * P
    zooplankton_metabolic_loss = lᶻⁿ * Q₁₀(T) * Z
    remineralization = rᵈⁿ * D

    return phytoplankton_metabolic_loss + zooplankton_metabolic_loss + remineralization - phytoplankton_consumption
end

@inline function (bgc::NPZDc)(::Val{:P}, x, y, z, t, N, P, Z, D, T, PAR)
    μ₀ = bgc.base_maximum_growth
    kₙ = bgc.nutrient_half_saturation
    α = bgc.initial_photosynthetic_slope
    gₘₐₓ = bgc.maximum_grazing_rate
    kₚ = bgc.grazing_half_saturation
    lᵖⁿ = bgc.base_respiration_rate
    lᵖᵈ = bgc.phyto_base_mortality_rate

    growth = μ₀ * Q₁₀(T) * nutrient_limitation(N, kₙ) * light_limitation(PAR, α, μ₀ * Q₁₀(T)) * P
    grazing = gₘₐₓ * nutrient_limitation(P ^ 2, kₚ ^ 2) * Z
    metabolic_loss = lᵖⁿ * Q₁₀(T) * P
    mortality_loss = lᵖᵈ * Q₁₀(T) * P

    return growth - grazing - metabolic_loss - mortality_loss
end

@inline function (bgc::NPZDc)(::Val{:Z}, x, y, z, t, N, P, Z, D, T, PAR)
    gₘₐₓ = bgc.maximum_grazing_rate
    kₚ = bgc.grazing_half_saturation
    lᶻⁿ = bgc.base_excretion_rate
    lᶻᵈ = bgc.zoo_base_mortality_rate
    β = bgc.assimulation_efficiency

    grazing = β * gₘₐₓ * nutrient_limitation(P ^ 2, kₚ ^ 2) * Z
    metabolic_loss = lᶻⁿ * Q₁₀(T) * Z
    mortality_loss = lᶻᵈ * Q₁₀(T) * Z ^ 2

    return grazing - metabolic_loss - mortality_loss 
end

@inline function (bgc::NPZDc)(::Val{:D}, x, y, z, t, N, P, Z, D, T, PAR)
    lᵖᵈ = bgc.phyto_base_mortality_rate
    gₘₐₓ = bgc.maximum_grazing_rate
    kₚ = bgc.grazing_half_saturation
    β = bgc.assimulation_efficiency
    lᶻᵈ = bgc.zoo_base_mortality_rate
    rᵈⁿ = bgc.remineralization_rate

    phytoplankton_mortality_loss = lᵖᵈ * Q₁₀(T) * P
    zooplankton_assimilation_loss = (1 - β) * gₘₐₓ * nutrient_limitation(P ^ 2, kₚ ^ 2) * Z
    zooplankton_mortality_loss = lᶻᵈ * Q₁₀(T) * Z ^ 2
    remineralization = rᵈⁿ * D

    return phytoplankton_mortality_loss + zooplankton_assimilation_loss + zooplankton_mortality_loss - remineralization
end

@inline function biogeochemical_drift_velocity(bgc::NPZDc, ::Val{tracer_name}) where tracer_name
    if tracer_name in keys(bgc.sinking_velocities)
        return (u = ZeroField(), v = ZeroField(), w = bgc.sinking_velocities[tracer_name])
    else
        return (u = ZeroField(), v = ZeroField(), w = ZeroField())
    end
end

summary(::NPZDc{FT, NamedTuple{K, V}}) where {FT, K, V} = string("NPZDc{$FT} model, with $K sinking")
show(io::IO, model::NPZDc{FT}) where {FT} = print(io, string("NPZDc{$FT} model \n",
                                                            "└── Sinking Velocities:", "\n", show_sinking_velocities(model.sinking_velocities)))

@inline maximum_sinking_velocity(bgc::NPZDc) = maximum(abs, bgc.sinking_velocities.D.w)

adapt_structure(to, NPZDc::NPZDc) = 
    NPZDc(adapt(to, NPZDc.initial_photosynthetic_slope),
                                             adapt(to, NPZDc.base_maximum_growth),
                                             adapt(to, NPZDc.nutrient_half_saturation),
                                             adapt(to, NPZDc.base_respiration_rate),
                                             adapt(to, NPZDc.phyto_base_mortality_rate),

                                             adapt(to, NPZDc.maximum_grazing_rate),
                                             adapt(to, NPZDc.grazing_half_saturation),
                                             adapt(to, NPZDc.assimulation_efficiency),
                                             adapt(to, NPZDc.base_excretion_rate),
                                             adapt(to, NPZDc.zoo_base_mortality_rate),

                                             adapt(to, NPZDc.remineralization_rate),

                                             adapt(to, NPZDc.sinking_velocities))

@inline redfield(i, j, k, val_tracer_name, bgc::NPZDc, tracers) = redfield(val_tracer_name, bgc)
@inline redfield(::Union{Val{:N}}, bgc::NPZDc{FT}) where FT = convert(FT, 0)
@inline redfield(::Union{Val{:P}, Val{:Z}, Val{:D}}, bgc::NPZDc{FT}) where FT = convert(FT, 6.56)

@inline conserved_tracers(::NPZDc) = (:N, :P, :Z, :D)
@inline sinking_tracers(bgc::NPZDc) = keys(bgc.sinking_velocities)

@inline chlorophyll(bgc::NPZDc{FT}, model) where FT = convert(FT, 1.31) * model.tracers.P

end # module
