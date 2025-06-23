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
CO₂ = 7.57e-1 * 1e6 
HCO₃ = 1.67e-3 * 1e6 
CO₃ = 3.15e-4 * 1e6 
BOH₃ = 2.97e-4 * 1e6 
BOH₄ = 1.19e-4 * 1e6 
#H = 6.31e-9
OH = 9.6e-6 * 1e6 

min = 60.0 # seconds per minute
hr = 60.0 * min # minutes per hour
day = 24hr # hours per day

# ODE system
function boxmodel_ode!(du, u, p, t)
    CO₂ = u[1]
    HCO₃ = u[2]
    CO₃ = u[3]
    H = 0.0
    OH = u[4]
    BOH₃ = u[5]
    BOH₄ = u[6]
    T = 25
    S = 35
    du[1] =  bgc(Val(:CO₂),  0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S, H)
    du[2] =  bgc(Val(:HCO₃), 0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S, H)
    du[3] =  bgc(Val(:CO₃),  0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S, H)
    du[4] =  bgc(Val(:OH),   0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S, H)
    du[5] =  bgc(Val(:BOH₃), 0, 0, 0, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S, H)
    du[6] = -du[1]
end


function clip_negatives!(integrator)
    @. integrator.u = max(integrator.u, 1e-20)
end
condition(u, t, integrator) = any(<(0), u)
cb = DiscreteCallback(condition, clip_negatives!)

t_final = 60 # final time, in seconds
dt_out = 0.05 # output rate, not solver timestep size, in seconds
c_0 = [CO₂, HCO₃, CO₃, OH, BOH₃, BOH₄]
tspan = (0.0, t_final)
prob = ODEProblem(boxmodel_ode!, c_0, tspan)
sol = solve(prob, Rosenbrock23(), reltol = 1e-10, abstol = 1e-12, saveat = dt_out)#, maxiters = 10_000_000, callback = cb) #KenCarp47(linsolve = KrylovJL_GMRES()), solve(prob, alg_hints = [:stiff], reltol = 1e-10, abstol = 1e-15, saveat = dt_out)

@save "outputs/0d-case.jld2" t=sol.t u=sol.u


