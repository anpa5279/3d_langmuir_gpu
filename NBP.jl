using Pkg
using MPI
MPI.Init() # Initialize MPI
using CUDA
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.DistributedComputations
using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: launch!
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: update_state!
using Oceananigans: UpdateStateCallsite
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Operators: volume
const Nx = 128        # number of points in each of x direction
const Ny = 128        # number of points in each of y direction
const Nz = 128        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const N² = 5.3e-9    # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 30.0 # m 
const Q = 0.0     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const ρ_calcite = 2710.0 # kg m⁻³, dummy density of CaCO3
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 17.0    # C, temperature at the surface  
const S₀ = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number
const calcite0 = 10.0e6 # kg
const r_plume = 1e-4 # [m] "Fine sand"
# Automatically distribute among available processors
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch

#stokes drift
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz

#BCs
u_f = La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx)) 
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)), bottom = GradientBoundaryCondition(0.0))
calcite_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(0.0))
coriolis = FPlane(f=1e-4) # s⁻¹
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S₀)

w_plume = CenterField(grid, boundary_conditions=FieldBoundaryConditions(grid, (Center, Center, Center)))
sinking = AdvectiveForcing(w=w_plume)
#defining model
model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            tracers = (:T, :calcite, ),
                            timestepper = :RungeKutta3,
                            closure = Smagorinsky(), 
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs, calcite=calcite_bcs), 
                            forcing = (calcite=sinking,))

@show model

# ICs
r_z(z) = randn(Xoshiro()) * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+ dTdz * model.grid.Lz * 1e-6 * r_z(z) 
uᵢ(x, y, z) = u_f * 1e-1 * r_z(z) 
vᵢ(x, y, z) = -u_f * 1e-1 * r_z(z) 
σ = 10.0 # m
calciteᵢ(x, y, z) = calcite0/sqrt(2*pi* σ^2) * exp(-z^2 / (2 * σ^2)) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, calcite=calciteᵢ)

day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 2*day) #stop_time = 96hours,
@show simulation
# updating dense plume
function update_plume_velocity(simulation)
    arch = model.architecture
    calcite = model.tracers.calcite
    ν = model.diffusivity_fields.νₑ #1.05e-6 # [m² s⁻¹] molecular kinematic viscosity of water
    grid = model.grid
    fill_halo_regions!(calcite)
    launch!(arch, grid, :xyz, plume_velocity!, grid, g_Earth, calcite, w_plume)
    fill_halo_regions!(w_plume)
end 
@kernel function plume_velocity!(grid, g, calcite, w_plume)
    i, j, k = @index(Global, NTuple)
    #defining dense plume
    ρ_plume = calcite[i, j, k] / Vᶜᶜᶜ(i, j, k, grid) #local density of the plume
    Δb = g * (ρₒ - ρ_plume) / ρₒ # m s⁻²
    @inbounds w_plume[i, j, k] = -2 * Δb * (r_plume)^2 / 9 / 1.05e6# m s⁻¹

end

simulation.callbacks[:sink] = Callback(update_plume_velocity, IterationInterval(1), callsite=UpdateStateCallsite())
# outputs and running
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

conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30.0seconds)

#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    file["IC/wind_speed"] = u₁₀
    return nothing
end

output_interval = 4hours

u, v, w = model.velocities
T = model.tracers.T
calcite = model.tracers.calcite
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, calcite),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "NBP_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)

W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T = Average(T, dims=(1, 2))
                                                      
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "NBP_averages.jld2",
                                                    overwrite_existing = true)

run!(simulation)#; pickup = true)