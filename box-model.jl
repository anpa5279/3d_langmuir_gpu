Pkg.develop(path="/Users/annapauls/.julia/dev/Oceananigans.jl-main") #this will call for my version of Oceananigans locally 
Pkg.develop(path="/Users/annapauls/.julia/dev/OceanBioME.jl-main") #this will call for my version of OceanBioME locally
using OceanBioME, Oceananigans
using Oceananigans.Units
#using MPI
#using CUDA
using Oceananigans.BuoyancyFormulations: g_Earth
using Printf
#include("cc.jl")
#using .CC #: CarbonateChemistry #local module
#rank = MPI.Comm_rank(MPI.COMM_WORLD)
grid = BoxModelGrid()
clock = Clock(time = zero(grid))

model = BoxModel(; biogeochemistry = CarbonateChemistry(; grid), clock)

perturb = 1e3
set!(model, BOH₃ = 2.97e-4 * 1e6, BOH₄ = 1.19e-4 * 1e6, CO₂ = 7.57e-6 * 1e6 * perturb, CO₃ = 3.15e-4 * 1e6, HCO₃ = 1.67e-3 * 1e6, OH = 9.6e-6 * 1e6, T=25, S = 35)

simulation = Simulation(model, Δt=1e-7, stop_time = 3seconds)
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