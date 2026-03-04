using ThreadPinning
using MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nthreads = Threads.nthreads()
#mpi_pinthreads(:numa)
using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds

using Logging
global_logger(SimpleLogger(stdout, Logging.Info))
## simulation parameters
Nx = 256
Ny = 256
Nz = 256
Lx = 320            # (m) domain horizontal extents
Ly = 320            # (m) domain horizontal extents
Lz = 96             # (m) domain depth 
MLD = 30.0          # m, mixed layer depth
dTdz = 0.01         # K m⁻¹, temperature gradient
alpha = 2.0e-4      # 1/K, thermal expansion coefficient
rp = 10.0           # m, radius of surface buoyancy flux
rho0 = 1025.0       # kg m⁻³, seawater density
rho_tracer = 1300.0 # kg m⁻³, reference density for tracer
u₁₀ = 5.75          # (m s⁻¹) wind speed at 10 meters above the ocean

# defining grid
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
@show grid

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))
g = Oceananigans.defaults.gravitational_acceleration
wp = -0.001
b0 = -4*10^(-1) # m s⁻²
rho_ratio = (rho_tracer-rho0) / (rho0) 
Sj = -(g*rho_ratio)/b0 # amount of tracer flux needed to achieve the desired buoyancy flux
Jᵇ = wp*Sj
@inline function sflux(x, y, t) 
    σ = 10.0 # m
    return wp*Sj/(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
end
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(sflux), 
                                bottom = GradientBoundaryCondition(0.0))

# closure
Re = 3000
w_max = 0.10747783287769483
visc = w_max*Lz/Re # 1.0e-5 # m² s⁻¹
sgs = ScalarDiffusivity(ν=visc, κ=visc)

# buoyancy
beta = (rho_tracer - rho0) / (rho0 * Sj) 
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha, haline_contraction = beta))

## defining model
model = NonhydrostaticModel(grid;
                            buoyancy, 
                            advection = WENO(),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                            closure = sgs
                            )
@show model
## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
uᵢ(x, y, z) = wp * r(x, y, z)
vᵢ(x, y, z) = -wp * r(x, y, z)
Tᵢ(x, y, z) = z > - MLD ? dTdz * Lz * 1e-6 * r(x, y, z) : #random noise in the mixed layer
            dTdz * (z + MLD) + dTdz * Lz * 1e-6 * r(x, y, z)
set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, S=0.0)

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
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_Δt = 1.0, max_Δt=30seconds) #ensrues cfl is updated ever iteration
## output files
output_interval = 0.2hours
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
P_static = model.pressures.pHY′
P_dynamic = model.pressures.pNHS
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S, P_static, P_dynamic),
                                                    with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "fields.jld2",
                                                    init = save_IC!,
                                                    overwrite_existing = true)

# running the simulation
run!(simulation)