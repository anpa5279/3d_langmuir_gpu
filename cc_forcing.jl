using KernelAbstractions: @kernel, @index
using LinearAlgebra: norm

import Oceananigans.Utils: launch!
import Oceananigans.Architectures: child_architecture

function update_params(model, dt)
    grid = model.grid
    arch = child_architecture(grid)
    tracers = model.tracers

    CO2   = @inbounds tracers.CO2
    HCO3  = @inbounds tracers.HCO3
    CO3   = @inbounds tracers.CO3
    OH    = @inbounds tracers.OH
    BOH3  = @inbounds tracers.BOH3
    BOH4  = @inbounds tracers.BOH4
    T     = @inbounds tracers.T

    launch!(arch, grid, :xyz, intDriver!, grid, model.clock.time, model.clock.time+dt*0.5,
            CO2, HCO3, CO3, OH, BOH3, BOH4, T)
end

@kernel function intDriver!(grid, CO2, HCO3, CO3, OH, BOH3, BOH4, T)
    i, j, k = @index(Global, NTuple)
    CO2_val = @inbounds CO2[i,j,k]
    HCO3_val = @inbounds HCO3[i,j,k]
    CO3_val = @inbounds CO3[i,j,k]
    OH_val = @inbounds OH[i,j,k]
    BOH3_val = @inbounds BOH3[i,j,k]
    BOH4_val = @inbounds BOH4[i,j,k]
    c0 = [CO2_val, HCO3_val, CO3_val, OH_val, BOH3_val, BOH4_val]
    T_val = @inbounds T[i,j,k]

    # Compute the full RKC step with adaptive timestep inside
    c_new = dcdt(c0, T_val)

    @inbounds begin
        CO2[i,j,k]  = c_new[1]
        HCO3[i,j,k] = c_new[2]
        CO3[i,j,k]  = c_new[3]
        OH[i,j,k]   = c_new[4]
        BOH3[i,j,k] = c_new[5]
        BOH4[i,j,k] = c_new[6]
    end
end

# ------------------------
# dcdt for carbonate chemistry (6 variables)
# ------------------------
function dcdt(u::AbstractVector, T::Float64)
    CO2, HCO3, CO3, OH, BOH3, BOH4 = u
    du = similar(u)
    S = 35.0
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    # rate constants from Zeebe & Wolf-Gladrow, Dickson & Goyet
    # (define alphaX, betaX functions elsewhere)
    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(4.70e7/1e6, 23.2, T)
    b2 = beta2(a2, Kw, K1)
    a3 = 5e10/1e6
    b3 = beta3(a3, K2)
    a4 = 6e9/1e6
    b4 = beta4(a4, Kw, K2)
    a5 = 1.40e-3*1e6
    b5 = beta5(a5, Kw)
    a6 = alpha6(4.58e10/1e6, 20.8, T)
    b6 = beta6(a6, Kw, Kb)
    a7 = alpha7(3.05e10/1e6, 20.8, T)
    b7 = beta7(a7, K2, Kb)
    @show a1 a2 a3 a4 a5 a6 a7 b1 b2 b3 b4 b5 b6 b7
    H = H_qss(a1, b1, a3, b3, a5, b5, CO2, HCO3, CO3, OH)
    #@show H
    if isnan(H) error("H concentration is NaN") end
    p = (a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7)

    du[1] = CarbonateChemistry_CO2(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    du[2] = CarbonateChemistry_HCO3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    du[3] = CarbonateChemistry_CO3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    du[4] = CarbonateChemistry_OH(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    du[5] = CarbonateChemistry_BOH3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    du[6] = -du[5]  # BOH4
    return du
end


##### All chemical equations
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
@inline function CarbonateChemistry_CO2(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    a1 = p[1]
    a2 = p[2]
    b1 = p[8]
    b2 = p[9]
    if isnan(CO2) error("CO2 concentration is NaN") end
    dcdt = - (a1 + a2 * OH) * CO2 + (b1 * H + b2) * HCO3
    return dcdt # converting to micromol/kg rate
end

@inline function CarbonateChemistry_HCO3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    a4 = p[4]
    a7 = p[7]
    b1 = p[8]
    b2 = p[9]
    b3 = p[10]
    b4 = p[11]
    b7 = p[14]
    if isnan(HCO3) error("HCO3 concentration is NaN") end
    dcdt = (a1 + a2 * OH) * CO2 - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH4) * HCO3 + (a3 * H + b4 + a7 * BOH3) * CO3
    return dcdt # converting to micromol/kg rate
end

@inline function CarbonateChemistry_CO3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    a3 = p[3]
    a4 = p[4]
    a7 = p[7]
    b3 = p[10]
    b4 = p[11]
    b7 = p[14]
    if isnan(CO3) error("CO3 concentration is NaN") end
    dcdt = (b3 + a4 * OH + b7 * BOH4) * HCO3 - (a3 * H + b4 + a7 * BOH3) * CO3
    return dcdt # converting to micromol/kg rate
end

@inline function CarbonateChemistry_OH(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    a2 = p[2]
    a4 = p[4]
    a5 = p[5]
    a6 = p[6]
    b2 = p[9]
    b4 = p[11]
    b5 = p[12]
    b6 = p[13]
    if isnan(OH) error("OH concentration is NaN") end
    dcdt = - a2 * OH * CO2 + (b2 - a4 * OH) * HCO3 + b4 * CO3 + (a5 - b5 * H * OH) - (a6 * OH * BOH3 - b6 * BOH4)
    return dcdt # converting to micromol/kg rate
end

@inline function CarbonateChemistry_BOH3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, p)
    a6 = p[6]
    a7 = p[7]
    b6 = p[13]
    b7 = p[14]
    if isnan(BOH3) error("BOH3 concentration is NaN") end
    if isnan(BOH4) error("BOH4 concentration is NaN") end
    dcdt = b7 * BOH4 * HCO3 - a7 * BOH3 * CO3 - (a6 * OH * BOH3 - b6 * BOH4)
    return dcdt # converting to micromol/kg rate
end