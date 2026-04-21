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
using Oceananigans.Units: minute, minutes, hours, seconds, stokes_velocity, dstokes_dz
using Oceananigans.BoundaryConditions: fill_halo_regions!, OpenBoundaryCondition
using Oceananigans.Models: BoundaryAdjacentMean
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ

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
wp = -0.001 # m/s, vertical velocity for surface buoyancy flux
Sj = 0.1 # g/kg, tracer mass 

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

# stokes drift
g = Oceananigans.defaults.gravitational_acceleration
dusdz_top = dstokes_dz(grid.z.cᵃᵃᶜ[Nz]/2, u₁₀)
dusdz_bot = dstokes_dz(grid.z.cᵃᵃᶜ[0], u₁₀)
dusdz_bcs = FieldBoundaryConditions(grid, (nothing, nothing, Center()), top = ValueBoundaryCondition(dusdz_top), 
                                bottom = ValueBoundaryCondition(dusdz_bot))
dusdz = Field{Nothing, Nothing, Center}(grid; boundary_conditions = dusdz_bcs)
dusdz_1d = dstokes_dz.(grid.z.cᵃᵃᶜ[1:Nz], u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
us = stokes_velocity.(grid.z.cᵃᵃᶜ[1:Nz], u₁₀)

## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
uᵢ(x, y, z) = wp * r(x, y, z)
vᵢ(x, y, z) = -wp * r(x, y, z)
Tᵢ(x, y, z) = z > - MLD ? dTdz * Lz * 1e-6 * r(x, y, z) : #random noise in the mixed layer
            dTdz * (z + MLD) + dTdz * Lz * 1e-6 * r(x, y, z)

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
## output file inputs
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = us
    file["IC/wind_speed"] = u₁₀
    return nothing
end

# for loop variations 
# Langmuir number ---> changes wind stress (u flux BC)
#for La_t in [0.2, 0.3, 0.4]
#uf = La_t^2 * us[Nz]
#τx = -(uf^2)
#u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), bottom = GradientBoundaryCondition(0.0))
# turn stokes on or off
# for stokes in [nothing, UniformStokesDrift(∂z_uˢ=dusdz)]
# turn coriolis on or off
# for coriolis in [nothing, FPlane(f=1e-4)]

## defining model
model = NonhydrostaticModel(grid;
                            coriolis,
                            stokes_drift = stokes,
                            buoyancy, 
                            advection = WENO(),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                            closure = sgs
                            )
@show model
if stokes !== nothing
    uᵢ(x, y, z) = wp * r(x, y, z) + stokes_velocity(z, u₁₀)
end
set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, S=0.0)
# defining simulation
simulation = Simulation(model, Δt=30, stop_time = 12hours) 
@show simulation
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
                                                    init = save_IC!
                                                    overwrite_existing = true, init = save_grid!)

# running the simulation
run!(simulation)