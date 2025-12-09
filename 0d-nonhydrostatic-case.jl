import Pkg
Pkg.activate("/Users/annapauls/.julia/environments/carbonate chemsitry/")
using Pkg
using Statistics
using Printf
using Random
Pkg.develop(path="/Users/annapauls/Documents/Github repositories/personal_oceananigans/Oceananigans.jl-main")
Pkg.status()
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
const Nx = 1        # number of points in each of x direction
const Ny = 1        # number of points in each of y direction
const Nz = 1        # number of points in the vertical direction
const Lx = 1    # (m) domain horizontal extents
const Ly = 1    # (m) domain horizontal extents
const Lz = 1    # (m) domain depth 
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) 
@show grid 
model = NonhydrostaticModel(; grid, 
                            tracers = (:BOH3, :BOH4, :CO2, :CO3, :HCO3, :OH, :T),
                            timestepper = :CCRungeKutta3, #chemical kinetics are embedded in this timestepper
                            )
@show model

perturb = 1e3
set!(model, T=25, BOH3 = 2.97e2, BOH4 = 1.19e2, CO2 = 7.57e0 * perturb, CO3 = 3.15e2, HCO3 = 1.67e3, OH = 9.6e0) 

day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 5*minutes)

output_interval = 5.0seconds

simulation.output_writers[:fields] = JLD2Writer(model, simulation.model.tracers,
                                                dir = "localoutputs/cc testing/",
                                                schedule = TimeInterval(output_interval),
                                                filename = "fields.jld2", #$(rank)
                                                overwrite_existing = true)

function progress(simulation)
    u, v, w = simulation.model.velocities 
    
    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n
    co2 = %.1e, co3 = %.1e, hco3 = %.1e, oh = %.1e, boh3 = %.1e, boh4 = %.1e",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time), 
                   mean(simulation.model.tracers.CO2),
                   mean(simulation.model.tracers.CO3),
                   mean(simulation.model.tracers.HCO3),
                   mean(simulation.model.tracers.OH),
                   mean(simulation.model.tracers.BOH3),
                   mean(simulation.model.tracers.BOH4))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

@info "Running the model..."
run!(simulation)