module CC
using OceanBioME: setup_velocity_fields, Biogeochemistry, ScaleNegativeTracers
using OceanBioME.Light: default_surface_PAR

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: ZeroField
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry

using KernelAbstractions

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers,
                                     required_biogeochemical_auxiliary_fields,
                                     biogeochemical_drift_velocity
import OceanBioME: conserved_tracers

struct CarbonateChemistry{FT, W} <: AbstractContinuousFormBiogeochemistry
    #Zeebe and Wolf Gladrow 2001
    A1:: FT
    E1:: FT 
    A6:: FT
    E6:: FT
    A7:: FT
    E7:: FT
    #Dickson and Goyet 1994
    alpha3 :: FT
    alpha4 :: FT
    alpha5 :: FT
    sinking_velocities::W
end

function CarbonateChemistry(; grid::AbstractGrid{FT},
                            #Zeebe and Wolf Gladrow 2001
                            A1:: FT = 4.70e7, # kg/mol/s
                            E1:: FT = 32.2, # kJ/mol
                            A6:: FT = 4.58e10, # kg/mol/s
                            E6:: FT = 20.8, # kJ/mol
                            A7:: FT = 3.05e10, # kg/mol/s
                            E7:: FT = 20.8, # kJ/mol
                            alpha3 :: FT = 5e10, # kg/mol/s
                            alpha4 :: FT = 6.0e9, # kg/mol/s
                            alpha5:: FT = 1.40e-3, # kg/mol/s

                            light_attenuation = default_surface_PAR, 
                            
                            sediment_model = nothing,

                            sinking_speeds = nothing,
                            open_bottom::Bool = true,

                            scale_negatives = true,
                            invalid_fill_value = NaN,
                                                                    
                            particles = nothing,
                            modifiers = nothing) where {FT}

    sinking_velocities = nothing
    underlying_biogeochemistry = CarbonateChemistry(A1, E1, A6, E6, A7, E7, alpha3, alpha4, alpha5, sinking_velocities)

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid; invalid_fill_value)
        modifiers = isnothing(modifiers) ? scaler : (modifiers..., scaler)
    end

    return Biogeochemistry(underlying_biogeochemistry;
                           light_attenuation = default_surface_PAR, 
                           sediment = nothing, 
                           particles,
                           modifiers)
end

required_biogeochemical_tracers(::CarbonateChemistry) = (:CO₂, :HCO₃, :CO₃, :H, :OH, :BOH₃, :BOH₄, :T, :S)
required_biogeochemical_auxiliary_fields(::CarbonateChemistry) = ()

#functions needed to update dc/dt
@kernel function _compute_CarbonateChemistry_vars!(grid, model_fields, bgc)
    i, j, k = @index(Global, NTuple)

    T = model_fields.T[i, j, k]
    S = model_fields.S[i, j, k]
    
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K1 = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.2) * sqrt(S) + 0.08468345 * S +
            0.00654208 * S^1.5 + log(1 - 0.001005 * S))
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 + 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    #Zeebe and Wolf Gladrow 2001
    alpha1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
    beta1 = alpha1/ K1# 1/s
    alpha2 = bgc.A1 * exp(-bgc.E1 / (R * (T + 273.15))) # kg/mol/s
    beta2 = alpha2* Kw/ K2# 1/s
    beta3 = bgc.alpha3 * K2 # 1/s
    beta4 = bgc.alpha4 * Kw/ K2# 1/s
    beta5 = bgc.alpha5 / Kw# kg/mol/s
    alpha6 = bgc.A6 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta6 =  alpha6* Kw/ Kb# 1/s
    alpha7 = bgc.A7 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta7 = alpha7* K2/ Kb# kg/mol/s

end 

#updating tracers 
@inline function (bgc::CarbonateChemistry)(::Val{:CO₂}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K1 = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.2) * sqrt(S) + 0.08468345 * S +
            0.00654208 * S^1.5 + log(1 - 0.001005 * S))
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    #Zeebe and Wolf Gladrow 2001
    alpha1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
    beta1 = alpha1/ K1# 1/s
    alpha2 = bgc.A1 * exp(-bgc.E1 / (R * (T + 273.15))) # kg/mol/s
    beta2 = alpha2* Kw/ K2# 1/s
    return - (alpha1 + alpha2 * OH) * CO₂ + (beta1 * H + beta2) * CO₂
end

@inline function (bgc::CarbonateChemistry)(::Val{:HCO₃}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K1 = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.2) * sqrt(S) + 0.08468345 * S +
            0.00654208 * S^1.5 + log(1 - 0.001005 * S))
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 + 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))
    #Zeebe and Wolf Gladrow 2001
    alpha1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
    beta1 = alpha1/ K1 # 1/s
    alpha2 = bgc.A1 * exp(-bgc.E1 / (R * (T + 273.15))) # kg/mol/s
    beta2 = alpha2* Kw/ K2 # 1/s
    alpha3 = bgc.alpha3
    beta3 = alpha3 * K2 # 1/s
    alpha4 = bgc.alpha4
    beta4 = alpha4 * Kw/ K2 # 1/s
    alpha7 = bgc.A7 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta7 = alpha7* K2/ Kb # kg/mol/s

    return (alpha1 + alpha2 * OH) * CO₂ - (beta1 * H + beta2 + beta3 + alpha4 * OH + beta7 * BOH₄) * HCO₃ +
            (alpha3 * H + beta4 + alpha7 * BOH₃) * CO₃
end

@inline function (bgc::CarbonateChemistry)(::Val{:CO₃}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 + 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))
    #Zeebe and Wolf Gladrow 2001
    alpha3 = bgc.alpha3
    beta3 = alpha3 * K2 # 1/s
    alpha4 = bgc.alpha4
    beta4 = alpha4 * Kw/ K2 # 1/s
    alpha7 = bgc.A7 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta7 = alpha7* K2/ Kb # kg/mol/s
    
    return (beta3 + alpha4 * OH + beta7 * BOH₄) * HCO₃ - (alpha3 * H + beta4 + alpha7 * BOH₃) * CO₃
end

@inline function (bgc::CarbonateChemistry)(::Val{:H}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K1 = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.2) * sqrt(S) + 0.08468345 * S +
            0.00654208 * S^1.5 + log(1 - 0.001005 * S))
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    #Zeebe and Wolf Gladrow 2001
    alpha1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
    beta1 = alpha1/ K1 # 1/s
    alpha3 = bgc.alpha3
    beta3 = alpha3 * K2 # 1/s
    alpha5 = bgc.alpha5
    beta5 = alpha5 / Kw # kg/mol/s

    return alpha1 * CO₂ - (beta1 * H - beta3) * HCO₃ - alpha3 * H * CO₃ + (alpha5 - beta5 * H * OH)
end

@inline function (bgc::CarbonateChemistry)(::Val{:OH}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 + 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))
    #Zeebe and Wolf Gladrow 2001
    alpha2 = bgc.A1 * exp(-bgc.E1 / (R * (T + 273.15))) # kg/mol/s
    beta2 = alpha2* Kw/ K2 # 1/s
    alpha4 = bgc.alpha4
    beta4 = alpha4 * Kw/ K2 # 1/s
    alpha5 = bgc.alpha5
    beta5 = alpha5 / Kw # kg/mol/s
    alpha6 = bgc.A6 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta6 =  alpha6* Kw/ Kb # 1/s

    return - alpha2 * OH * CO₂ + (beta2 - alpha4 * OH) * HCO₃ + beta4 * CO₃ + (alpha5 - beta5 * H * OH) -
            (alpha6 * OH * BOH₃ - beta6 * BOH₄)
end

@inline function (bgc::CarbonateChemistry)(::Val{:BOH₃}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 + 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))
    #Zeebe and Wolf Gladrow 2001
    alpha6 = bgc.A6 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta6 =  alpha6* Kw/ Kb # 1/s
    alpha7 = bgc.A7 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta7 = alpha7* K2/ Kb # kg/mol/s

    return beta7 * BOH₄ * HCO₃ - alpha7 * BOH₃ * CO₃ - (alpha6 * OH * BOH₃ - beta6 * BOH₄)
end

@inline function (bgc::CarbonateChemistry)(::Val{:BOH₄}, x, y, z, t, CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
    R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S + 
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
    Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
            (118.67 / (T + 273.15) - 5.977 -1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
    Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 + 
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
            1.62142 * S) + + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
            + 0.053105 * sqrt(S) * (T + 273.15))
    #Zeebe and Wolf Gladrow 2001
    alpha6 = bgc.A6 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta6 =  alpha6* Kw/ Kb # 1/s
    alpha7 = bgc.A7 * exp(-bgc.E7 / (R * (T + 273.15))) # kg/mol/s
    beta7 = alpha7* K2/ Kb # kg/mol/s
    
    return - beta7 * BOH₄ * HCO₃ + alpha7 * BOH₃ * CO₃ + (alpha6 * OH * BOH₃ - beta6 * BOH₄)
end

#default drift velocity 
@inline function biogeochemical_drift_velocity(bgc::CarbonateChemistry, ::Val{tracer_name}) where tracer_name
    return (u = ZeroField(), v = ZeroField(), w = ZeroField())
end

#conserving tracers
@inline conserved_tracers(::CarbonateChemistry) = (:CO₂, :HCO₃, :CO₃, :H, :OH, :BOH₃, :BOH₄)

summary(::CarbonateChemistry{FT}) where {FT} = string("CarbonateChemistryr model")
show(io::IO, model::CarbonateChemistry{FT}) where {FT} = print(io, string("CarbonateChemistryr model"))


end
