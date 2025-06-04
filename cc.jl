module CC #any edits made to this file, restart julia to see changes

export CarbonateChemistry

using OceanBioME: setup_velocity_fields, Biogeochemistry, ScaleNegativeTracers
using OceanBioME.Light: default_surface_PAR

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: ZeroField
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers,
                                     required_biogeochemical_auxiliary_fields,
                                     biogeochemical_drift_velocity
import OceanBioME: conserved_tracers

struct CarbonateChemistry{FT, W} <: AbstractContinuousFormBiogeochemistry
    #Zeebe and Wolf Gladrow 2001
    A1:: FT
    E1:: FT 
    A7:: FT
    E7:: FT
    A8:: FT
    E8:: FT
    #Dickson and Goyet 1994
    alpha3 :: FT
    alpha4 :: FT
    alpha5 :: FT
    sinking_velocities::W
end

function CarbonateChemistry(; grid::AbstractGrid{FT},
                            #Zeebe and Wolf Gladrow 2001
                            A1:: FT = 4.70e7, # kg/mol/s
                            E1:: FT = 23.2, # kJ/mol
                            A7:: FT = 4.58e10, # kg/mol/s
                            E7:: FT = 20.8, # kJ/mol
                            A8:: FT = 3.05e10, # kg/mol/s
                            E8:: FT = 20.8, # kJ/mol
                            alpha3 :: FT = 5e10, # kg/mol/s
                            alpha4 :: FT = 6.0e9, # kg/mol/s
                            alpha5:: FT = 1.40e-3, # kg/mol/s

                            light_attenuation = default_surface_PAR, 
                            
                            sediment_model = nothing,

                            sinking_speeds = nothing,
                            open_bottom::Bool = true,

                            scale_negatives = false,
                            invalid_fill_value = NaN,
                                                                    
                            particles = nothing,
                            modifiers = nothing) where {FT}

    sinking_velocities = nothing
    underlying_biogeochemistry = CarbonateChemistry(A1, E1, A7, E7, A8, E8, alpha3, alpha4, alpha5, sinking_velocities)

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid; invalid_fill_value)
        modifiers = isnothing(modifiers) ? scaler : (modifiers..., scaler)
    end

    return Biogeochemistry(underlying_biogeochemistry;
                           modifiers)
end

required_biogeochemical_tracers(::CarbonateChemistry) = (:CO₂, :HCO₃, :CO₃, :H, :OH, :BOH₃, :BOH₄, :T, :S)
required_biogeochemical_auxiliary_fields(::CarbonateChemistry) = ()

const R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1

#Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
@inline K_1(T, S) = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.20760841) * sqrt(S) + 0.08468345 * S -
            0.00654208 * S^1.5 + log(1 - 0.001005 * S))

@inline K_2(T, S) = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S -
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))

@inline K_w(T, S) = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 + 1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)

@inline K_b(T, S) = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 - 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))

#Zeebe and Wolf Gladrow 2001
@inline alpha1(T) = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
@inline beta1(alpha1, K1) = alpha1/ K1 # 1/s
@inline alpha2(A1, E1, T) = A1 * exp(-E1 / (R * (T + 273.15))) # kg/mol/s
@inline beta2(alpha2, Kw, K2) = alpha2 * Kw/ K2# 1/s
@inline beta3(alpha3, K2) = alpha3 * K2 # 1/s
@inline beta4(alpha4, Kw, K2) = alpha4 * Kw/ K2 # 1/s
@inline beta5(alpha5, Kw) = alpha5 / Kw # kg/mol/s
@inline alpha6(A7, E8, T) = A7 * exp(-E8 / (R * (T + 273.15))) # kg/mol/s
@inline beta6(alpha6, Kw, Kb) =  alpha6* Kw/ Kb # 1/s
@inline alpha7(A8, E8, T) = A8 * exp(-E8 / (R * (T + 273.15))) # kg/mol/s
@inline beta7(alpha7, K2, Kb) = alpha7* K2/ Kb # kg/mol/s
#updating tracers 
@inline function (bgc::CarbonateChemistry)(::Val{:CO₂}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(bgc.A1, bgc.E1, T)
    b2 = beta2(a2, Kw, K2)
    return - (a1 + a2 * OH) * CO₂ + (b1 * H + b2) * HCO₃
end

@inline function (bgc::CarbonateChemistry)(::Val{:HCO₃}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(bgc.A1, bgc.E1, T)
    b2 = beta2(a2, Kw, K2)
    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a4 = bgc.alpha4
    b4 = beta4(a4, Kw, K2)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)

    return (a1 + a2 * OH) * CO₂ - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH₄) * HCO₃ + (a3 * H + b4 + a7 * BOH₃) * CO₃
end

@inline function (bgc::CarbonateChemistry)(::Val{:CO₃}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a4 = bgc.alpha4
    b4 = beta4(a4, Kw, K2)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)
    
    return (b3 + a4 * OH + b7 * BOH₄) * HCO₃ - (a3 * H + b4 + a7 * BOH₃) * CO₃
end

@inline function (bgc::CarbonateChemistry)(::Val{:H}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a3 = bgc.alpha3
    b3 = beta3(a3, K2)
    a5 = bgc.alpha5
    b5 = beta5(a5, Kw)

    return a1 * CO₂ - (b1 * H - b3) * HCO₃ - a3 * H * CO₃ + (a5 - b5 * H * OH)
end

@inline function (bgc::CarbonateChemistry)(::Val{:OH}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a2 = alpha2(bgc.A1, bgc.E1, T)
    b2 = beta2(a2, Kw, K2)
    a4 = bgc.alpha4
    b4 = beta4(a4, Kw, K2)
    a5 = bgc.alpha5
    b5 = beta5(a5, Kw)
    a6 = alpha6(bgc.A7, bgc.E8, T)
    b6 = beta6(a6, Kw, Kb)

    return - a2 * OH * CO₂ + (b2 - a4 * OH) * HCO₃ + b4 * CO₃ + (a5 - b5 * H * OH) - (a6 * OH * BOH₃ - b6 * BOH₄)
end

@inline function (bgc::CarbonateChemistry)(::Val{:BOH₃}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a6 = alpha6(bgc.A7, bgc.E8, T)
    b6 = beta6(a6, Kw, Kb)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)

    return b7 * BOH₄ * HCO₃ - a7 * BOH₃ * CO₃ - (a6 * OH * BOH₃ - b6 * BOH₄)
end

@inline function (bgc::CarbonateChemistry)(::Val{:BOH₄}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    a6 = alpha6(bgc.A7, bgc.E8, T)
    b6 = beta6(a6, Kw, Kb)
    a7 = alpha7(bgc.A8, bgc.E8, T)
    b7 = beta7(a7, K2, Kb)
    println("alpha6 = ", a6, " beta6 = ", b6, " alpha7 = ", a7, " beta7 = ", b7)
    println("T = ", T, " S = ", S)
    println("Kw = ", Kw, " Kb = ", Kb, " K2 = ", K2)
    println("BOH₃ = ", BOH₃, " BOH₄ = ", BOH₄, " HCO₃ = ", HCO₃, " CO₃ = ", CO₃)
    println(- b7 * BOH₄ * HCO₃ + a7 * BOH₃ * CO₃ + (a6 * OH * BOH₃ - b6 * BOH₄))
    if isnan(BOH₄) error("BOH₄ is NaN, check your inputs") end
    if isnan(BOH₃) error("BOH₃ is NaN, check your inputs") end
    if isnan(HCO₃) error("HCO₃ is NaN, check your inputs") end
    if isnan(CO₃) error("CO₃ is NaN, check your inputs") end
    if isnan(CO₂) error("CO₂ is NaN, check your inputs") end
    if isnan(H) error("H is NaN, check your inputs") end
    if isnan(OH) error("OH is NaN, check your inputs") end
    return - b7 * BOH₄ * HCO₃ + a7 * BOH₃ * CO₃ + (a6 * OH * BOH₃ - b6 * BOH₄)
end

#default drift velocity 
@inline function biogeochemical_drift_velocity(bgc::CarbonateChemistry, ::Val{tracer_name}) where tracer_name
    return (u = ZeroField(), v = ZeroField(), w = ZeroField())
end

#conserving tracers
@inline conserved_tracers(::CarbonateChemistry) = (:CO₂, :HCO₃, :CO₃, :H, :OH, :BOH₃, :BOH₄)

end