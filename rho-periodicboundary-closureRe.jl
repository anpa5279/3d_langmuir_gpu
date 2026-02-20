#using ThreadPinning
#using MPI
#MPI.Init()
#rank = MPI.Comm_rank(MPI.COMM_WORLD)
#nthreads = Threads.nthreads()
#mpi_pinthreads(:numa)
using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BoundaryConditions: fill_halo_regions!, OpenBoundaryCondition
using Oceananigans.Models: BoundaryAdjacentMean
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ

using Logging
global_logger(SimpleLogger(stdout, Logging.Info))
## simulation parameters
Lx = 320    # (m) domain horizontal extents
Ly = 320    # (m) domain horizontal extents
Lz = 96    # (m) domain depth 
dTdz = 0.01  # K m⁻¹, temperature gradient
β = 2.0e-4     # 1/K, thermal expansion coefficient
w_max = 0.10747783287769483

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
g = Oceananigans.defaults.gravitational_acceleration

u_f = 0.001
b0 = -4*10^(-1) # m s⁻²
Jᵇ = -u_f*b0 # m² s⁻³, surface buoyancy flux
@inline function bflux_t(x, y, t) 
    σ = 10.0 # m
    return Jᵇ/(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
end
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(bflux_t), 
                                    bottom = GradientBoundaryCondition(g*β*dTdz))

buoyancy = BuoyancyTracer()
## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
uᵢ(x, y, z) = u_f * r(x, y, z)
vᵢ(x, y, z) = -u_f * r(x, y, z)

# closure
Re = 3000
visc = w_max*Lz/Re # 1.0e-5 # m² s⁻¹
sgs = ScalarDiffusivity(ν=visc, κ=visc)
@show sgs
for MLD in (15, 45) # m, mixed layer depth
    path = "with closure Re $Re/MLD = $MLD m/"
    bᵢ(x, y, z) = z > - MLD ? g*β*dTdz * Lz * 1e-6 * r(x, y, z) : #random noise in the mixed layer
                g*β*dTdz * (z + MLD) + g*β*dTdz * Lz * 1e-6 * r(x, y, z)
    for N in (256, )
        Nx = N
        Ny = N
        for Nz in (256, )#16, 32, 64, 128
            println("Running simulation with Nx = $Nx, Ny = $Ny, Nz = $Nz")
            ## referring to files with desiraed functions
            grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
            @show grid
            ## defining model
            model = NonhydrostaticModel(grid;
                                        buoyancy, 
                                        advection = WENO(),
                                        tracers = (:b,),
                                        timestepper = :RungeKutta3,
                                        boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs),
                                        closure = sgs
                                        )
            @show model
            set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

            # defining simulation
            simulation = Simulation(model, Δt=30, stop_time = 12hours) 
            @show simulation
            ## progress function
            function progress(simulation)
                u, v, w = simulation.model.velocities
                # Print a progress message
                msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                            iteration(simulation),
                            prettytime(time(simulation)),
                            prettytime(simulation.Δt),
                            maximum(abs, u), maximum(abs, v), maximum(abs, w),
                            prettytime(simulation.run_wall_time))
                @info msg
                return nothing
            end
            simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))
            ## updating cfl every time step
            conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, min_Δt = 1.0, max_Δt=30seconds) #ensrues cfl is updated ever iteration
            ## output files
            output_interval = 0.2hours

            rel_path = "$path/flux b tracer Nx = $Nx, Ny = $Ny, Nz = $Nz/"
            u, v, w = model.velocities
            b = model.tracers.b
            P_static = model.pressures.pHY′
            P_dynamic = model.pressures.pNHS
            simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, b, P_static, P_dynamic),
                                                                dir = rel_path,  with_halos=false,
                                                                array_type = Array{Float64},
                                                                schedule = TimeInterval(output_interval),
                                                                filename = "fields.jld2", #$(rank)
                                                                overwrite_existing = true)

            # running the simulation
            run!(simulation)
        end
    end
end 