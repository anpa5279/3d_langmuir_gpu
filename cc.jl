module CC #any edits made to this file, restart julia to see changes

export CarbonateChemistry

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

import OceanBioME: conserved_tracers, maximum_sinking_velocity

struct CarbonateChemistry{FT, W} <: AbstractContinuousFormBiogeochemistry
    #Zeebe and Wolf Gladrow 2001
    A1 :: FT # kg/umol/s
    E1 :: FT # kJ/mol
    A7 :: FT # kg/umol/s
    E7 :: FT # kJ/mol
    A8 :: FT # kg/umol/s
    E8 :: FT # kJ/mol
    #Dickson and Goyet 1994
    alpha3 :: FT # kg/umol/s
    alpha4 :: FT # kg/umol/s
    alpha5 :: FT # kg/umol/s
    sinking_velocities :: W
    function CarbonateChemistry(A1::FT, # kg/umol/s
                                E1::FT, # kJ/mol
                                A7::FT, # kg/umol/s
                                E7::FT, # kJ/mol
                                A8::FT, # kg/umol/s
                                E8::FT, # kJ/mol
                                alpha3::FT, # kg/umol/s
                                alpha4::FT, # kg/umol/s
                                alpha5::FT, # umol/kg/s
                                sinking_velocities::W) where {FT, W}
        return new{FT, W}(A1, E1, A7, E7, A8, E8, alpha3, alpha4, alpha5, sinking_velocities)
    end
end

function CarbonateChemistry(; grid::AbstractGrid{FT},
                            #Zeebe and Wolf Gladrow 2001
                            A1::FT = 4.70e7 / 1e6, # kg/umol/s
                            E1::FT = 23.2, # kJ/mol
                            A7::FT = 4.58e10 / 1e6, # kg/umol/s
                            E7::FT = 20.8, # kJ/mol
                            A8::FT = 3.05e10 / 1e6, # kg/umol/s
                            E8::FT = 20.8, # kJ/mol
                            #Dickson and Goyet 1994
                            alpha3::FT = 5e10 / 1e6, # kg/umol/s
                            alpha4::FT = 6.0e9 / 1e6, # kg/umol/s
                            alpha5::FT = 1.40e-3 * 1e6, # umol/kg/s

                            light_attenuation_model::LA = nothing, 
                            
                            sediment_model::S = nothing,

                            sinking_speeds = (CO₂ = 0.0, HCO₃ = 0.0, CO₃ = 0.0, H = 0.0, OH = 0.0, BOH₃ = 0.0, BOH₄ = 0.0),
                            open_bottom::Bool = true,

                            scale_negatives = false,
                            invalid_fill_value = NaN,
                                                                    
                            particles::P = nothing,
                            modifiers::M = nothing) where {FT, LA, S, P, M}
    sinking_velocities = setup_velocity_fields(sinking_speeds, grid, open_bottom)

    underlying_biogeochemistry = CarbonateChemistry(A1, E1, A7, E7, A8, E8, alpha3, alpha4, alpha5, sinking_velocities)

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

required_biogeochemical_tracers(::CarbonateChemistry) = (:CO₂, :HCO₃, :CO₃, :OH, :BOH₃, :BOH₄, :T, :S)
required_biogeochemical_auxiliary_fields(::CarbonateChemistry) = ()#(:stage)#(:H,)

const R = 0.00831446261815324 # kJ⋅K⁻1⋅mol⁻1

#Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
@inline K_1(T, S) = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.20760841) * sqrt(S) + 0.08468345 * S -
            0.00654208 * S^1.5 + log(1 - 0.001005 * S)) * 1.0e6 #umol/kg

@inline K_2(T, S) = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S -
            0.00846934 * S^1.5 + log(1 - 0.001005 * S)) * 1.0e6 #umol/kg

@inline K_w(T, S) = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) +
            (118.67 / (T + 273.15) - 5.977 + 1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)  * 1.0e6 * 1.0e6 #umol^2/kg^2

@inline K_b(T, S) = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 - 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15))
            + 0.053105 * sqrt(S) * (T + 273.15)) * 1.0e6 #umol/kg

#Zeebe and Wolf Gladrow 2001
@inline alpha1(T) = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15))  # 1/s
@inline beta1(alpha1, K1) = alpha1/ K1# kg/umol/s
@inline alpha2(A1, E1, T) = A1 * exp(-E1 / (R * (T + 273.15)))# kg/umol/s
@inline beta2(alpha2, Kw, K1) = alpha2 * Kw/ K1# 1/s
@inline beta3(alpha3, K2) = alpha3 * K2 # 1/s
@inline beta4(alpha4, Kw, K2) = alpha4 * Kw/ K2 # 1/s
@inline beta5(alpha5, Kw) = alpha5 / Kw # kg/umol/s
@inline alpha6(A7, E8, T) = A7 * exp(-E8 / (R * (T + 273.15)))# kg/umol/s
@inline beta6(alpha6, Kw, Kb) =  alpha6* Kw/ Kb # 1/s
@inline alpha7(A8, E8, T) = A8 * exp(-E8 / (R * (T + 273.15)))# kg/umol/s
@inline beta7(alpha7, K2, Kb) = alpha7* K2/ Kb# kg/umol/s

#QSS approximation
@inline H_qss(alpha1, beta1, alpha3, beta3, alpha5, beta5, c1, c2, c3, c5) = (alpha1*c1 + beta3*c2 + alpha5)/(beta1*c2 + alpha3*c3 + beta5*c5)
#updating tracers 
@inline function (bgc::CarbonateChemistry)(::Val{:CO₂}, x, y, z, t, CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄, T, S) #, H)
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(bgc.A1, bgc.E1, T)
    b2 = beta2(a2, Kw, K1)
    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a5 = bgc.alpha5
    b5 = beta5(a5, Kw)

    H = H_qss(a1, b1, a3, b3, a5, b5, CO₂, HCO₃, CO₃, OH)
    #println("a1 = ", a1, " b1 = ", b1, " a2 = ", a2, " b2 = ", b2)
    if isnan(CO₂) error("CO₂ concentration is NaN") end
    dcdt = - (a1 + a2 * OH) * CO₂ + (b1 * H + b2) * HCO₃
    return dcdt # converting to micromol/kg rate
end

@inline function (bgc::CarbonateChemistry)(::Val{:HCO₃}, x, y, z, t, CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄, T, S) #, H)
    
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(bgc.A1, bgc.E1, T)
    b2 = beta2(a2, Kw, K1)
    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a4 = bgc.alpha4
    b4 = beta4(a4, Kw, K2)
    a5 = bgc.alpha5
    b5 = beta5(a5, Kw)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)

    H = H_qss(a1, b1, a3, b3, a5, b5, CO₂, HCO₃, CO₃, OH)
    if isnan(HCO₃) error("HCO₃ concentration is NaN") end
    dcdt = (a1 + a2 * OH) * CO₂ - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH₄) * HCO₃ + (a3 * H + b4 + a7 * BOH₃) * CO₃
    return dcdt # converting to micromol/kg rate
end

@inline function (bgc::CarbonateChemistry)(::Val{:CO₃}, x, y, z, t, CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄, T, S) #, H)
    
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a4 = bgc.alpha4
    b4 = beta4(a4, Kw, K2)
    a5 = bgc.alpha5
    b5 = beta5(a5, Kw)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)

    H = H_qss(a1, b1, a3, b3, a5, b5, CO₂, HCO₃, CO₃, OH)
    if isnan(CO₃) error("CO₃ concentration is NaN") end
    dcdt = (b3 + a4 * OH + b7 * BOH₄) * HCO₃ - (a3 * H + b4 + a7 * BOH₃) * CO₃
    return dcdt # converting to micromol/kg rate
end

@inline function (bgc::CarbonateChemistry)(::Val{:OH}, x, y, z, t, CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄, T, S) #, H)
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(bgc.A1, bgc.E1, T)
    b2 = beta2(a2, Kw, K1)
    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a4 = bgc.alpha4
    b4 = beta4(a4, Kw, K2)
    a5 = bgc.alpha5
    b5 = beta5(a5, Kw)
    a6 = alpha6(bgc.A7, bgc.E8, T)
    b6 = beta6(a6, Kw, Kb)

    H = H_qss(a1, b1, a3, b3, a5, b5, CO₂, HCO₃, CO₃, OH)
    #println("a6 = ", a6, " b6 = ", b6)
    if isnan(OH) error("OH concentration is NaN") end
    dcdt = - a2 * OH * CO₂ + (b2 - a4 * OH) * HCO₃ + b4 * CO₃ + (a5 - b5 * H * OH) - (a6 * OH * BOH₃ - b6 * BOH₄)
    return dcdt # converting to micromol/kg rate
end

@inline function (bgc::CarbonateChemistry)(::Val{:BOH₃}, x, y, z, t, CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄, T, S) #, H)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a6 = alpha6(bgc.A7, bgc.E8, T)
    b6 = beta6(a6, Kw, Kb)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)
    if isnan(BOH₃) error("BOH₃ concentration is NaN") end
    if isnan(BOH₄) error("BOH₄ concentration is NaN") end
    dcdt = b7 * BOH₄ * HCO₃ - a7 * BOH₃ * CO₃ - (a6 * OH * BOH₃ - b6 * BOH₄)
    return dcdt # converting to micromol/kg rate
end

@inline (bgc::CarbonateChemistry)(::Val{:BOH₄}, args...) = -bgc(Val(:BOH₃), args...)

#default drift velocity 
@inline function biogeochemical_drift_velocity(bgc::CarbonateChemistry, ::Val{tracer_name}) where tracer_name
    if tracer_name in keys(bgc.sinking_velocities)
        return (u = ZeroField(), v = ZeroField(), w = bgc.sinking_velocities[tracer_name])
    else
        return (u = ZeroField(), v = ZeroField(), w = ZeroField())
    end
end

#conserving tracers
@inline conserved_tracers(::CarbonateChemistry) = (:CO₂, :HCO₃, :CO₃, :OH, :BOH₃, :BOH₄)

@inline maximum_sinking_velocity(bgc::CarbonateChemistry) = 0.0
@inline sinking_tracers(bgc::CarbonateChemistry) = keys(bgc.sinking_velocities)

end #end of module