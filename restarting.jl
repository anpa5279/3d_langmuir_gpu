using Pkg
using MPI
using CUDA
using JLD2
using Statistics
using Printf
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.Fields: interior, set!
const Nx = 32        # number of points in each of x direction
const Ny = 32        # number of points in each of y direction
const Nz = 128        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const N² = 5.3e-9    # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 30.0 # m 
const Q = 1e11     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const ρ_calcite = 2710.0 # kg m⁻³, dummy density of CaCO3
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S₀ = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number
# Automatically distribute among available processors
arch = Distributed(GPU())
@show arch
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))
@show grid

#stokes drift
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz

u_f = p.La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, p.u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), #FluxBoundaryCondition(p.Q / (p.cᴾ * p.ρₒ * p.Lx * p.Ly)),
                                bottom = GradientBoundaryCondition(p.dTdz))
#coriolis = FPlane(f=1e-4) # s⁻¹
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=new_dUSDdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))

simulation = Simulation(model, Δt=30.0, stop_time = 96hours)
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
output_interval = 10minutes

fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2Writer(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields_$(rank).jld2",
                                                      overwrite_existing = false)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(Int(5.0e4)), prefix="model_checkpoint_$(rank)")
# Load the checkpoint file manually
file = jldopen("model_checkpoint_$(rank)_iteration50000.jld2", "r")
t = file["NonhydrostaticModel/clock"]

# Now manually set them in your simulation
simulation.model.clock.time = t.time
simulation.model.clock.iteration = t.iteration

u_full = Array(file["NonhydrostaticModel/u/data"])
v_full = Array(file["NonhydrostaticModel/v/data"])
w_full = Array(file["NonhydrostaticModel/w/data"])
T_full = Array(file["NonhydrostaticModel/T/data"])

u_interior = u_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]
v_interior = v_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]
w_interior = w_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]
T_interior = T_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]

set!(simulation.model.velocities.u, CuArray(reshape(u_interior, :)))
set!(simulation.model.velocities.v, CuArray(reshape(v_interior, :)))
set!(simulation.model.velocities.w, CuArray(reshape(w_interior, :)))
set!(simulation.model.tracers.T, CuArray(reshape(T_interior, :)))
close(file)

simulation.Δt = 30.0
run!(simulation) #; pickup = true) #"model_checkpoint_$(rank)_iteration50000.jld2") to run with the file, comment out the file you read in