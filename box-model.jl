using OceanBioME, Oceananigans
using Oceananigans.Units
using Oceananigans.BuoyancyFormulations: g_Earth
using Printf
using MPI
include("cc.jl")
using .CC #: CarbonateChemistry #local module

grid = BoxModelGrid()
clock = Clock(time = zero(grid))

biogeochemistry = CarbonateChemistry(; grid)

model = BoxModel(; biogeochemistry, clock)

set!(model, BOH₃ = 2.97e-4, BOH₄ = 1.19e-4, CO₂ = 20.0, CO₃ = 3.15e-4, HCO₃ = 1.67e-3, OH = 9.6e-6, T=25, S = 35)#BOH₃ = 2.97e-4, BOH₄ = 1.19e-4, CO₂ = 7.57e-6, CO₃ = 3.15e-4, HCO₃ = 1.67e-3, OH = 9.6e-6, T=25, S = 35)

simulation = Simulation(model, Δt=1e-8, stop_time = 60seconds)
@show simulation

output_interval = 3seconds

BOH₃ = model.fields.BOH₃
BOH₄ = model.fields.BOH₄
CO₂ = model.fields.CO₂
CO₃ = model.fields.CO₃
HCO₃ = model.fields.HCO₃
OH = model.fields.OH

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; BOH₃, BOH₄, CO₂, CO₃, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "outputscpu/box_model.jld2", #$(rank)
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(3e7))
@info "Running the model..."
run!(simulation)