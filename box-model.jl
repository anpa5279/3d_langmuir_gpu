import Pkg
Pkg.activate("/Users/annapauls/.julia/environments/carbonate chemsitry/")
Pkg.develop(path="/Users/annapauls/Documents/Github repositories/personal_oceananigans/Oceananigans.jl-main")
using Pkg
using OceanBioME, Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using MPI
using CUDA
using Oceananigans.BuoyancyFormulations: g_Earth
using Printf
#include("cc.jl")
#using .CC #: CarbonateChemistry #local module
#rank = MPI.Comm_rank(MPI.COMM_WORLD)
grid = BoxModelGrid()
clock = Clock(time = zero(grid))

model = BoxModel(; biogeochemistry = CarbonateChemistry(; grid), clock) #part of the issue is my latest version is not apart of the BGC model

perturb = 1e3
set!(model, T=25, BOH3 = 2.97e2, BOH4 = 1.19e2, CO2 = 7.57e0 * perturb, CO3 = 3.15e2, HCO3 = 1.67e3, OH = 9.6e0) 

simulation = Simulation(model, Δt=1e-7, stop_time = 3seconds) #0.05
@show simulation

output_interval = 0.00001seconds

BOH₃ = model.fields.BOH₃
BOH₄ = model.fields.BOH₄
CO₂ = model.fields.CO₂
CO₃ = model.fields.CO₃
HCO₃ = model.fields.HCO₃
OH = model.fields.OH

simulation.output_writers[:fields] = JLD2Writer(model, (; BOH₃, BOH₄, CO₂, CO₃, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "box_model.jld2", #$(rank)
                                                      overwrite_existing = true)

function progress(simulation)

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100000))
@info "Running the model..."
run!(simulation)