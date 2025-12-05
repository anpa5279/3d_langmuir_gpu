using Pkg
using MPI
using CUDA
@show MPI.has_cuda()
@show CUDA.has_cuda()
MPI.Init() # Initialize MPI
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
#using Oceananigans.BuoyancyFormulations: g_Earth #
using Oceananigans.DistributedComputations
using Oceananigans.TurbulenceClosures: Smagorinsky
const Nx = 128        # number of points in each of x direction
const Ny = 128        # number of points in each of y direction
const Nz = 128        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const initial_mixed_layer_depth = 30.0 # m 
const Q = 5.0     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S0 = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number
# Automatically distribute among available processors
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch
# other forcing
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)

coriolis = FPlane(f=1e-4) # s⁻¹

# stokes drift
g_Earth = buoyancy.gravitational_acceleration
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz

# BCs
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(dTdz))
@show T_bcs
us = stokes_velocity(z_d, u₁₀)
u_f = La_t^2 * us[end]
const τx = -(u_f^2)# m² s⁻², surface kinematic momentum flux
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), 
                                bottom = GradientBoundaryCondition(0.0))
@show u_bcs
v_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0.0), #GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
@show v_bcs


model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            tracers = :T,
                            buoyancy = buoyancy,
                            closure = Smagorinsky(coefficient=0.1),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs)
                            )
@show model
@show model.pressure_solver
@show model.pressure_solver.grid.architecture
# ICs
r_z(z) = z > - initial_mixed_layer_depth ? randn(Xoshiro()) : 0.0 
ampv = 1.0e-3 # m s⁻¹
ue(x, y, z) = r_z(z) * ampv 
uᵢ(x, y, z) = -ue(x, y, z) + stokes_velocity(z, u₁₀)
vᵢ(x, y, z) = ue(x, y, z)
wᵢ(x, y, z) = 0.0
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? (T0 + dTdz * model.grid.Lz * 1e-6 * r_z(z)) : T0 + dTdz * (z + initial_mixed_layer_depth) 
@show uᵢ, vᵢ, Tᵢ
set!(model, u=uᵢ, w=wᵢ, v=vᵢ, T=Tᵢ)
@show model
simulation = Simulation(model, Δt=30.0, stop_time=240*hours)
@show simulation

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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

#output files
function save_IC!(file, model)
    if (rank == 0 || Nranks == 1)# && iteration(model.simulation) == 1
        file["IC/friction_velocity"] = u_f
        file["IC/stokes_velocity"] = us
        file["IC/wind_speed"] = u₁₀
    end
    return nothing
end

output_interval = 2.4*hours

u, v, w = model.velocities
T = model.tracers.T
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T_avg = Average(T, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    with_halos = false,
                                                    array_type = Array{Float64},
                                                    init = save_IC!)
                                                      
T = Average(T, dims=(1, 2))
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "langmuir_turbulence_averages.jld2",
                                                    overwrite_existing = true,
                                                    with_halos = false,
                                                    array_type = Array{Float64})
simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(48hours), prefix="model_checkpoint")

run!(simulation)#; pickup = true)