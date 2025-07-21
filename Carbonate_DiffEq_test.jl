using DifferentialEquations
using ModelingToolkit
using JLD2

# ----------------------------------------------------------------------------------
# Constants and parameters (as before)
# ----------------------------------------------------------------------------------
const T = 25.0
const S = 35.0
const R = 8.31446261815324

const A1 = 4.70e7
const E1 = 1000 * 23.2
const A7 = 4.58e10
const E7 = 1000 * 20.8
const A8 = 3.05e10
const E8 = 1000 * 20.8

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

alpha1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15))
beta1 = alpha1/ K1
alpha2 = A1 * exp(-E1 / (R * (T + 273.15)))
beta2 = alpha2 * Kw/ K1
alpha3 = 5e10
beta3 = alpha3 * K2
alpha4 = 6.0e9
beta4 = alpha4 * Kw/ K2
alpha5 = 1.40e-3
beta5 = alpha5 / Kw
alpha6 = A7 * exp(-E8 / (R * (T + 273.15)))
beta6 =  alpha6* Kw/ Kb
alpha7 = A8 * exp(-E8 / (R * (T + 273.15)))
beta7 = alpha7* K2/ Kb
# ----------------------------------------------------------------------------------
# Simulation Setup
# ----------------------------------------------------------------------------------
CO₂₀ = 7.57e-1
HCO₃₀ = 1.67e-3
CO₃₀ = 3.15e-4
H₀ = 6.31e-9
OH₀ = 9.6e-6
BOH₃₀ = 2.97e-4
BOH₄₀ = 1.19e-4

c₀ = [CO₂₀, HCO₃₀, CO₃₀, H₀, OH₀, BOH₃₀, BOH₄₀]
# ----------------------------------------------------------------------------------
# Define the symbolic variables and system using ModelingToolkit
# ----------------------------------------------------------------------------------
@parameters t a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7
@variables CO2(t) HCO3(t) CO3(t) H(t) OH(t) BOH3(t) BOH4(t)
@derivatives D'~t

eqs = [
    D(CO2) ~ - (a1 + a2 * OH) * CO2 + (b1 * H + b2) * HCO3,
    D(HCO3) ~ (a1 + a2 * OH) * CO2 - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH4) * HCO3 + (a3 * H + b4 + a7 * BOH3) * CO3,
    D(CO3) ~ (b3 + a4 * OH + b7 * BOH4) * HCO3 - (a3 * H + b4 + a7 * BOH3) * CO3,
    D(H) ~ a1 * CO2 - (b1 * H - b3) * HCO3 - a3 * H * CO3 + (a5 - b5 * H * OH),
    D(OH) ~ - a2 * OH * CO2 + (b2 - a4 * OH) * HCO3 + b4 * CO3 + (a5 - b5 * H * OH) - (a6 * OH * BOH3 - b6 * BOH4),
    D(BOH3) ~ b7 * BOH4 * HCO3 - a7 * BOH3 * CO3 - (a6 * OH * BOH3 - b6 * BOH4),
    D(BOH4) ~ - (b7 * BOH4 * HCO3 - a7 * BOH3 * CO3 - (a6 * OH * BOH3 - b6 * BOH4))
]

@named sys = ODESystem(eqs, t)
sys_simplified = structural_simplify(sys)

# Generate ODEFunction and Jacobian
f = ODEFunction(sys_simplified; jac=true)
J_func = f.jac
p = [alpha1, beta1, alpha2, beta2, alpha3, beta3, alpha4, beta4, alpha5, beta5, alpha6, beta6, alpha7, beta7]
J = f.jac(0.0, c₀, p)
println("Jacobian matrix at t=0:")
println(J)

tspan = (0.0, 10.0)

prob = ODEProblem(f, c₀, tspan)
dt_out = 0.1
sol = solve(prob, Rodas5(), reltol=1e-8, abstol=1e-8, saveat=dt_out)

@save "outputs/carbonate-diffeq-test.jld2" t=sol.t u=sol.u