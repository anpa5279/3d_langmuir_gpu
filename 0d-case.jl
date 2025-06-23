using OceanBioME, Oceananigans
using Oceananigans.Units
using OceanBioME: Biogeochemistry, ScaleNegativeTracers
using Printf
using DifferentialEquations
using JLD2
using DiffEqCallbacks
include("cc.jl")
using .CC #: CarbonateChemistry #local module

grid = RectilinearGrid(size=(1, 1, 1), extent=(1.0, 1.0, 1.0))
bgc = CarbonateChemistry(; grid, scale_negatives = true)#$, 

#converting mol/kg to umol/kg
perturb = 1e1 # perturbation factor for initial conditions
CO₂ = 7.57e-6 * 1e6 * perturb
HCO₃ = 1.67e-3 * 1e6 #* perturb
CO₃ = 3.15e-4 * 1e6 #* perturb
BOH₃ = 2.97e-4 * 1e6 #* perturb
BOH₄ = 1.19e-4 * 1e6 #* perturb
OH = 9.6e-6 * 1e6 #* perturb

min = 60.0 # seconds per minute
hr = 60.0 * min # minutes per hour
day = 24hr # hours per day

# ODE system
function boxmodel_ode!(du, u, p, t)
    CO₂ = u[1]
    HCO₃ = u[2]
    CO₃ = u[3]
    OH = u[4]
    BOH₃ = u[5]
    BOH₄ = u[6]
    T = 25
    S = 35
    du[1] = bgc(Val(:CO₂),  0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[2] = bgc(Val(:HCO₃), 0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[3] = bgc(Val(:CO₃),  0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[4] = bgc(Val(:OH),   0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[5] = bgc(Val(:BOH₃), 0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[6] = bgc(Val(:BOH₄), 0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
end

t_final = 60 # final time, in seconds
dt_out = 0.05 # output rate, not solver timestep size, in seconds
c_0 = [CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄]
tspan = (0.0, t_final)
prob = ODEProblem(boxmodel_ode!, c_0, tspan)
sol = solve(prob, Rosenbrock23(), reltol = 1e-10, abstol = 1e-12, saveat = dt_out)

@save "outputs/0d-case.jld2" t=sol.t u=sol.u


