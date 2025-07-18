using DifferentialEquations  # import the SciML ODE solver package
using JLD2
using ModelingToolkit

#----------------------------------------------------------------------------------
# Defining the carbonate chemistry system of ODEs
#----------------------------------------------------------------------------------
const T = 25.0
const S = 35.0
const R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1

#Zeebe and Wolf Gladrow 2001
const A1 = 4.70e7 # kg/mol/s
const E1 = 1000 * 23.2 # J/mol
const A7 = 4.58e10 # kg/mol/s
const E7 = 1000 * 20.8 # J/mol
const A8 = 3.05e10 # kg/mol/s
const E8 = 1000 * 20.8 # J/mol

#Dickson and Goyet 1994 (references Roy et al. 1993,  Dickson 1990, and Millero 1994)
const K1 = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
            (-4.0484 / (T + 273.15) - 0.20760841) * sqrt(S) + 0.08468345 * S -
            0.00654208 * S^1.5 + log(1 - 0.001005 * S))
const K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
            (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S -
            0.00846934 * S^1.5 + log(1 - 0.001005 * S))
const Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) +
            (118.67 / (T + 273.15) - 5.977 + 1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)
const Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 -
            0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) +
            1.62142 * S) + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15))
            + 0.053105 * sqrt(S) * (T + 273.15))

#Zeebe and Wolf Gladrow 2001
const a1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
const b1 = a1/ K1 # 1/s
const a2 = A1 * exp(-E1 / (R * (T + 273.15))) # kg/mol/s
const b2 = a2 * Kw/ K1# 1/s
const a3 = 5e10 # kg/mol/s
const b3 = a3 * K2 # 1/s
const a4 = 6.0e9 # kg/mol/s
const b4 = a4 * Kw/ K2 # 1/s
const a5 = 1.40e-3 # kg/mol/s
const b5 = a5 / Kw # kg/mol/s
const a6 = A7 * exp(-E8 / (R * (T + 273.15))) # kg/mol/s
const b6 =  a6* Kw/ Kb # 1/s
const a7 = A8 * exp(-E8 / (R * (T + 273.15))) # kg/mol/s
const b7 = a7* K2/ Kb # kg/mol/s

#@show a1
#@show b1
#@show a2
#@show b2
#@show a3
#@show b3
#@show a4
#@show b4
#@show a5
#@show b5
#@show a6
#@show b6
#@show a7
#@show b7

# individual reaction rates
@inline function dCO₂dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄)
    dcdt = - (a1 + a2 * OH) * CO₂ + (b1 * H + b2) * HCO₃
    return dcdt
end

@inline function dHCO₃dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄)
    dcdt = (a1 + a2 * OH) * CO₂ - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH₄) * HCO₃ + (a3 * H + b4 + a7 * BOH₃) * CO₃
    return dcdt
end

@inline function dCO₃dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄)
    dcdt = (b3 + a4 * OH + b7 * BOH₄) * HCO₃ - (a3 * H + b4 + a7 * BOH₃) * CO₃
    return dcdt
end

@inline function dHdt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄)
    dcdt = a1 * CO₂ - (b1 * H - b3) * HCO₃ - a3 * H * CO₃ + (a5 - b5 * H * OH)
    return dcdt
end

@inline function dOHdt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄)
    dcdt = - a2 * OH * CO₂ + (b2 - a4 * OH) * HCO₃ + b4 * CO₃ + (a5 - b5 * H * OH) - (a6 * OH * BOH₃ - b6 * BOH₄)
    return dcdt
end

@inline function dBOH₃dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄)
    dcdt = b7 * BOH₄ * HCO₃ - a7 * BOH₃ * CO₃ - (a6 * OH * BOH₃ - b6 * BOH₄)
    return dcdt
end

# complete RHS function. The `p` and `t` are unused arguments needed to match the DiffEq interface
function carbonate_rhs!(dc, c, p, t)
    dc[1] = dCO₂dt(c...)    # dCO₂
    dc[2] = dHCO₃dt(c...)   # dHCO₃
    dc[3] = dCO₃dt(c...)    # dCO₃
    dc[4] = dHdt(c...)      # dH
    dc[5] = dOHdt(c...)     # dOH
    dc[6] = dBOH₃dt(c...)   # dBOH₃
    dc[7] = - dc[6]         # dBOH₄
end

#----------------------------------------------------------------------------------
# Set up the ODE problem
f = ODEFunction(carbonate_rhs!)

#----------------------------------------------------------------------------------
# Run script
#----------------------------------------------------------------------------------
CO₂ = 7.57e-1
HCO₃ = 1.67e-3
CO₃ = 3.15e-4
BOH₃ = 2.97e-4
BOH₄ = 1.19e-4
OH = 9.6e-6
H = 6.31e-9

min = 60.0 # seconds per minute
hr = 60.0 * min # minutes per hour
day = 24hr # hours per day

t_final = 10# 1hr # final time, in seconds
dt_out = 0.1#2min # output rate, not solver timestep size, in seconds
c_0 = [CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄]
tspan = (0.0, t_final)
prob = ODEProblem(f, c_0, tspan)
sol = solve(prob, Rodas5(), reltol = 1e-8, abstol = 1e-8, saveat = dt_out)

@save "outputs/carbonate-diffeq-test.jld2" t=sol.t u=sol.u